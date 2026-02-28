import os
import time
import yaml
import torch
import pickle
import dm_env
import logging
import collections
import datetime
import numpy as np
import tracemalloc
from einops import rearrange
import matplotlib.pyplot as plt
from torchvision import transforms
from shadow_rm_robot.realman_arm import RmArm
from shadow_camera.realsense import RealSenseCamera
from shadow_act.models.latent_model import Latent_Model_Transformer
from shadow_act.network.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from shadow_act.utils.utils import set_seed
import sys
import cv2

# --- Updated Import: Use the new coordinated gripper model file ---
# Add the project root to sys.path if not already present
sys.path.append('/home/rm/aloha/shadow_rm_act/src/shadow_act')
# Assuming EDIL_GLTSmodel_coordinated.py is in src/shadow_act/train/traingripper/
from train.traingripper.EDIL_GLTSmodel_coordinated import CoordinatedMultiModalGripperNet

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- 2. Data Preprocessing: Deterministic Transform for Inference ---
# Training used ColorJitter/RandomAffine which is BAD for inference consistency.
# Here we define a strict deterministic transform (Resize + Norm).
def get_inference_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class RmActEvaluator:
    def __init__(self, config, save_episode=True, num_rollouts=50):
        self.config = config
        self._seed = config["seed"]
        self.robot_env = config["robot_env"]
        self.checkpoint_dir = config["checkpoint_dir"]
        self.checkpoint_name = config["checkpoint_name"]
        self.save_episode = save_episode
        self.num_rollouts = num_rollouts
        self.state_dim = config["state_dim"]
        self.real_robot = config["real_robot"]
        self.policy_class = config["policy_class"]
        self.onscreen_render = config["onscreen_render"]
        self.camera_names = config["camera_names"]
        self.max_timesteps = config["episode_len"]
        self.task_name = config["task_name"]
        self.temporal_agg = config["temporal_agg"]
        self.onscreen_cam = "angle"
        self.policy_config = config["policy_config"]
        self.vq = config["policy_config"]["vq"]
        
        # --- Updated: Load gripper model config ---
        self.gripper_model_path = config.get("gripper_model_path")
        self.gl_model_sequence_length = config.get("gl_model_sequence_length", 10)
        
        self.stats = None
        self.env = None
        self.env_max_reward = 0
        self.gripper_model = None

    def _make_policy(self, policy_class, policy_config):
        if policy_class == "ACT":
            return ACTPolicy(policy_config)
        elif policy_class == "CNNMLP":
            return CNNMLPPolicy(policy_config)
        elif policy_class == "Diffusion":
            return DiffusionPolicy(policy_config)
        else:
            raise NotImplementedError(f"Policy class {policy_class} is not implemented")

    def load_policy_and_stats(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)
        logging.info(f"Loading policy from: {checkpoint_path}")
        self.policy = self._make_policy(self.policy_class, self.policy_config)
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cuda:0")))
        self.policy.cuda()
        self.policy.eval()

        # --- Updated: Load Coordinated Gripper Model ---
        if self.gripper_model_path:
            logging.info(f"Loading Coordinated Gripper model from: {self.gripper_model_path}")
            # Ensure model architecture matches training (default params usually fine, or load from config)
            self.gripper_model = CoordinatedMultiModalGripperNet()
            self.gripper_model.load_state_dict(torch.load(self.gripper_model_path, map_location=torch.device("cuda:0")))
            self.gripper_model.cuda()
            self.gripper_model.eval()
            logging.info("Coordinated Gripper model loaded successfully.")
        else:
            logging.warning("`gripper_model_path` not found in config, gripper model will not be used.")

        stats_path = os.path.join(self.checkpoint_dir, "dataset_stats.pkl")
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)

    def pre_process(self, state_qpos):
        return (state_qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]

    def post_process(self, action):
        return action * self.stats["action_std"] + self.stats["action_mean"]

    def get_image_torch(self, timestep, camera_names):
        current_images = []
        for cam_name in camera_names:
            current_image = rearrange(timestep.observation["images"][cam_name], "h w c -> c h w")
            current_images.append(current_image)
        current_image = np.stack(current_images, axis=0)
        current_image = torch.from_numpy(current_image / 255.0).float().cuda().unsqueeze(0)
        return current_image

    def load_environment(self):
        if self.real_robot:
            self.env = DeviceAloha(self.robot_env)
        else:
            self.env = None

    def evaluate(self, checkpoint_name=None):
        if checkpoint_name is not None:
            self.checkpoint_name = checkpoint_name
        set_seed(self._seed)
        self.load_policy_and_stats()
        self.load_environment()

        if self.env is None:
            logging.error("Environment not loaded. Aborting evaluation.")
            return 0, 0
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join('result_GLTS', timestamp)
        os.makedirs(results_dir, exist_ok=True)
        logging.info(f"Saving evaluation results to: {os.path.abspath(results_dir)}")
        
        query_frequency = self.policy_config["num_queries"]
        max_timesteps = int(self.max_timesteps * 1)
 
        evaluation_interrupted = False
        for rollout_id in range(2):
            timestep = self.env.reset()

            DT = 1 / 8
            fps = int(1 / DT)
            video_filename = os.path.join(results_dir, f'rollout_{rollout_id}_cam_left.mp4')
            h, w, _ = timestep.observation['images']['cam_left'].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (w, h))

            qpos_history_raw = np.zeros((max_timesteps, self.state_dim))
            actions_history = []
            
            action_joint_left_sequence = collections.deque(maxlen=self.gl_model_sequence_length)
            action_joint_right_sequence = collections.deque(maxlen=self.gl_model_sequence_length)
            
            # --- Updated: Use deterministic transform for inference ---
            gripper_transform = get_inference_transforms()
            
            try:
                with torch.inference_mode():
                    start_time = time.time()
                    for t in range(max_timesteps):
                        obs = timestep.observation
                        # Record raw observation (BGR for OpenCV writer)
                        video_writer.write(cv2.cvtColor(obs['images']['cam_left'], cv2.COLOR_RGB2BGR))
                        
                        qpos_numpy = np.array(obs["qpos"])
                        qpos_history_raw[t] = qpos_numpy
                        qpos = self.pre_process(qpos_numpy)
                        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                        current_image = self.get_image_torch(timestep, self.camera_names)

                        if t == 0:
                            # Warm up main policy
                            for _ in range(10): self.policy(qpos, current_image)
                            logging.info("Main policy warm up done")
                            # Warm up coordinated gripper model
                            if self.gripper_model is not None:
                                dummy_img = torch.randn(1, 3, 224, 224).cuda()
                                dummy_seq = torch.randn(1, self.gl_model_sequence_length, 6).cuda()
                                for _ in range(10):
                                    self.gripper_model(dummy_img, dummy_seq, dummy_img, dummy_seq)
                                logging.info("Coordinated gripper model warm up done")

                        if t % query_frequency == 0:
                            all_actions = self.policy(qpos, current_image)
                        raw_action = all_actions[:, t % query_frequency]
                        
                        raw_action = raw_action.squeeze(0).cpu().numpy()
                        action = self.post_process(raw_action)
                        
                        action_joint_left = action[:6]
                        action_joint_right = action[7:13]
                        
                        # --- Updated: Robust sequence initialization ---
                        # If sequence is empty (start of episode), pad with current action to avoid errors
                        if not action_joint_left_sequence:
                             for _ in range(self.gl_model_sequence_length):
                                 action_joint_left_sequence.append(action_joint_left)
                                 action_joint_right_sequence.append(action_joint_right)
                        else:
                             action_joint_left_sequence.append(action_joint_left)
                             action_joint_right_sequence.append(action_joint_right)
                        
                        gripper_pred_left, gripper_pred_right = 0, 0

                        if self.gripper_model is not None:
                            # 1. Prepare Left Input
                            # Ensure image is RGB before transform (RealSense usually returns RGB, check if needed)
                            img_left_raw = obs['images']['cam_left'] 
                            # Convert to BGR for display/saving, but keep RGB for model if trained on RGB
                            # Assuming model was trained on RGB images.
                            img_left_tensor = gripper_transform(img_left_raw).unsqueeze(0).cuda()
                            
                            # Prepare Sequence (Already padded/filled above)
                            current_seq_left = list(action_joint_left_sequence)
                            seq_left_tensor = torch.from_numpy(np.array(current_seq_left)).float().unsqueeze(0).cuda()

                            # 2. Prepare Right Input
                            img_right_raw = obs['images']['cam_right']
                            img_right_tensor = gripper_transform(img_right_raw).unsqueeze(0).cuda()

                            current_seq_right = list(action_joint_right_sequence)
                            seq_right_tensor = torch.from_numpy(np.array(current_seq_right)).float().unsqueeze(0).cuda()
                            
                            # 3. Model Inference (Single call, dual output)
                            pred_l_tensor, pred_r_tensor = self.gripper_model(
                                img_left_tensor, seq_left_tensor, img_right_tensor, seq_right_tensor
                            )
                            # Sigmoid output is naturally in [0, 1]
                            gripper_pred_left = pred_l_tensor.item()
                            gripper_pred_right = pred_r_tensor.item()

                        # --- 3. Gripper Logic: Continuous Control with optional filtering ---
                        # Use raw continuous values directly if they are clean.
                        # Sigmoid outputs [0, 1].
                        # If you need a hard threshold for 'closing', you can keep it, 
                        # but if you want full continuous control, just pass the value.
                        
                        # Simple noise filter: If value is very low (likely open), clamp to 0 (Open)
                        # This prevents the gripper from jittering near the open state.
                        # Adjust 0.35/0.2 based on real-world sensitivity.
                        
                        print(f"t={t}, Raw L:{gripper_pred_left:.4f}, Raw R:{gripper_pred_right:.4f}")

                        gripper_confirm_left = gripper_pred_left if gripper_pred_left > 0.35 else 0.0
                        gripper_confirm_right = gripper_pred_right if gripper_pred_right > 0.2 else 0.0
                        
                        # Note: If your gripper uses 1.0 for Closed and 0.0 for Open (or vice versa),
                        # ensure this matches your training labels. 
                        # Assuming labels were: 0=Open, 1=Close (or Max).
                        
                        # 4. Assemble Final Action
                        action_good_left = np.append(action_joint_left, gripper_confirm_left)
                        action_good_right = np.append(action_joint_right, gripper_confirm_right)
                        final_action = np.concatenate([action_good_left, action_good_right])
                        actions_history.append(final_action)

                        # 5. Execute Action
                        timestep = self.env.step(final_action)
                        
                        # Sleep to maintain control frequency
                        expected_end_time = start_time + (t + 1) * DT
                        sleep_duration = expected_end_time - time.time()
                        if sleep_duration > 0:
                            time.sleep(sleep_duration)
            except KeyboardInterrupt:
                logging.warning(f"Rollout {rollout_id} interrupted by user.")
                evaluation_interrupted = True

            if actions_history:
                try:
                    logging.info(f"Saving {len(actions_history)} recorded actions for rollout {rollout_id}...")
                    actions_history_np = np.array(actions_history)
                    csv_filename = os.path.join(results_dir, f'rollout_{rollout_id}_actions.csv')
                    np.savetxt(csv_filename, actions_history_np, delimiter=',', fmt='%f')
                    logging.info(f"Successfully saved actions to {csv_filename}")
                except Exception as e:
                    logging.error(f"Failed to save actions for rollout {rollout_id}: {e}")
            else:
                logging.warning(f"No actions recorded for rollout {rollout_id}. Nothing to save.")

            video_writer.release()
            logging.info(f"Video saving complete for rollout {rollout_id}.")
            
            if evaluation_interrupted:
                logging.info("Evaluation interrupted by user, stopping.")
                break
        return 0, 0

class DeviceAloha:
    def __init__(self, aloha_config):
        """
        Initialize Device
        """
        config_left_arm = aloha_config["rm_left_arm"]
        config_right_arm = aloha_config["rm_right_arm"]
        config_head_camera = aloha_config["head_camera"]
        config_bottom_camera = aloha_config["bottom_camera"]
        config_left_camera = aloha_config["left_camera"]
        config_right_camera = aloha_config["right_camera"]
        self.init_left_arm_angle = aloha_config["init_left_arm_angle"]
        self.init_right_arm_angle = aloha_config["init_right_arm_angle"]
        self.arm_axis = aloha_config["arm_axis"]
        self.arm_left = RmArm(config_left_arm)
        self.arm_right = RmArm(config_right_arm)
        
        def _init_camera(camera_config):
            logging.info(f"Initializing camera with config: {camera_config}")
            return RealSenseCamera(camera_config, False)

        self.camera_top = _init_camera(config_head_camera) 
        self.camera_bottom = _init_camera(config_bottom_camera) 
        self.camera_left = _init_camera(config_left_camera) 
        self.camera_right = _init_camera(config_right_camera) 

        try:
            self.camera_top.start_camera()
            self.camera_bottom.start_camera()
            self.camera_left.start_camera()
            self.camera_right.start_camera()
        except Exception as e:
            logging.error(f"Failed to start cameras: {e}")
            self.stop_camera() 
            raise

    def stop_camera(self):
        """
        Stop cameras
        """
        if hasattr(self, 'camera_top') and self.camera_top and self.camera_top.camera_on:
            self.camera_top.stop_camera()
        if hasattr(self, 'camera_bottom') and self.camera_bottom and self.camera_bottom.camera_on:
            self.camera_bottom.stop_camera()
        if hasattr(self, 'camera_left') and self.camera_left and self.camera_left.camera_on:
            self.camera_left.stop_camera()
        if hasattr(self, 'camera_right') and self.camera_right and self.camera_right.camera_on:
            self.camera_right.stop_camera()
        logging.info("Cameras closed successfully.")

    def get_qps(self):
        """
        Get joint angles
        """
        left_slave_arm_angle = self.arm_left.get_joint_angle()
        left_joint_angles_array = np.array(list(left_slave_arm_angle.values()))
        right_slave_arm_angle = self.arm_right.get_joint_angle()
        right_joint_angles_array = np.array(list(right_slave_arm_angle.values()))
        return np.concatenate([left_joint_angles_array, right_joint_angles_array])

    def get_qvel(self):
        """
        Get joint velocities
        """
        left_slave_arm_velocity = self.arm_left.get_joint_velocity()
        left_joint_velocity_array = np.array(list(left_slave_arm_velocity.values()))
        right_slave_arm_velocity = self.arm_right.get_joint_velocity()
        right_joint_velocity_array = np.array(list(right_slave_arm_velocity.values()))
        return np.concatenate([left_joint_velocity_array, right_joint_velocity_array])

    def get_effort(self):
        """
        Get joint efforts
        """
        left_slave_arm_effort = self.arm_left.get_joint_effort()
        left_joint_effort_array = np.array(list(left_slave_arm_effort.values()))
        right_slave_arm_effort = self.arm_right.get_joint_effort()
        right_joint_effort_array = np.array(list(right_slave_arm_effort.values()))
        return np.concatenate([left_joint_effort_array, right_joint_effort_array])

    def get_images(self):
        """
        Get images
        """
        self.top_image, _, _, _ = self.camera_top.read_frame(True, False, False, False)
        self.bottom_image, _, _, _ = self.camera_bottom.read_frame(
            True, False, False, False
        )
        self.left_image, _, _, _ = self.camera_left.read_frame(
            True, False, False, False
        )
        self.right_image, _, _, _ = self.camera_right.read_frame(
            True, False, False, False
        )
        return {
            "cam_high": self.top_image,
            "cam_low": self.bottom_image,
            "cam_left": self.left_image,
            "cam_right": self.right_image,
        }
        

    def get_observation(self):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qps()
        obs["qvel"] = self.get_qvel()
        obs["effort"] = self.get_effort()
        obs["images"] = self.get_images()
        return obs

    def reset(self):
        logging.info("Resetting the environment")
        self.arm_left.set_joint_position(self.init_left_arm_angle[0:self.arm_axis])
        self.arm_right.set_joint_position(self.init_right_arm_angle[0:self.arm_axis])
        self.arm_left.set_gripper_position(0)
        self.arm_right.set_gripper_position(0)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=0,
            discount=None,
            observation=self.get_observation(),
        )

    def step(self, target_angle):
        self.arm_left.set_joint_canfd_position(target_angle[0:6])
        self.arm_right.set_joint_canfd_position(target_angle[7:13])
        self.arm_left.set_gripper_position(target_angle[6])
        self.arm_right.set_gripper_position(target_angle[13])
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=0,
            discount=None,
            observation=self.get_observation(),
        )


if __name__ == "__main__":
    # Load configuration
    # Update this path to your actual config location
    with open("/home/rm/collect_data/iWatch/dpkt/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    evaluator = RmActEvaluator(config)
    try:
        success_rate, avg_return = evaluator.evaluate()
    finally:
        if evaluator.real_robot and evaluator.env is not None:
            logging.info("Closing environment...")
            evaluator.env.stop_camera()
            logging.info("Environment closed.")
