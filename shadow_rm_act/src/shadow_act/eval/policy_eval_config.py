import yaml
import torch
import numpy as np
import os, time, logging, pickle, inspect
from typing import Dict
from tqdm import tqdm
#from utils.utils import set_seed, save_eval_results
#from utils.utils import set_seed
from policies.common.maker import make_policy
from envs.common_env import get_image, CommonEnv
import dm_env
import cv2
import os
import numpy as np
import h5py
import yaml
import logging
import time
import os
import yaml
import torch
import pickle
import logging
import collections
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
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
class DataValidator:
    def __init__(self, config):
        self.dataset_dir = config['dataset_dir']
        self.episode_idx = config['episode_idx']
        self.dataset_name = f'episode_{self.episode_idx}'
        self.dataset_path = os.path.join(self.dataset_dir, self.dataset_name + '.hdf5')
        self.joint_names = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
        self.state_names = self.joint_names + ["gripper"]
        self.left_arm = RmArm('/home/rm/aloha/shadow_rm_aloha/config/rm_left_arm.yaml')
        self.right_arm = RmArm('/home/rm/aloha/shadow_rm_aloha/config/rm_right_arm.yaml')

    def load_hdf5(self):
        if not os.path.isfile(self.dataset_path):
            logging.error(f'Dataset does not exist at {self.dataset_path}')
            exit()

        with h5py.File(self.dataset_path, 'r') as root:
            self.is_sim = root.attrs['sim']
            self.qpos = root['/observations/qpos'][()]
            self.qvel = root['/observations/qvel'][()]
            self.effort = root['/observations/effort'][()]
            self.action = root['/action'][()]
            self.image_dict = {cam_name: root[f'/observations/images/{cam_name}'][()]
                               for cam_name in root[f'/observations/images/'].keys()}

    def validate_data(self):
        # 验证位置数据
        if not self.qpos.shape[1] == 14:
            logging.error('qpos shape does not match expected number of joints')
            return False

        logging.info('Data validation passed')
        return True

    def control_robot(self):
        time0 = time.time()
        # 设置初始关节位置
        self.left_arm.set_joint_position(self.qpos[0][0:6])
        self.right_arm.set_joint_position(self.qpos[0][7:13])

        # 遍历qpos数组，控制机器人
        for qpos in self.qpos:
            logging.debug(f'qpos: {qpos}')
            # 控制左臂
            self.left_arm.set_joint_canfd_position(qpos[0:6])
            self.left_arm.set_gripper_position(qpos[6])
            # 控制右臂
            self.right_arm.set_joint_canfd_position(qpos[7:13])
            self.right_arm.set_gripper_position(qpos[13])
            logging.info(f'control time: {time.time() - time0}')

    def run(self):
        self.load_hdf5()
        if self.validate_data():
            self.control_robot()
def load_config_from_yaml(yaml_path):
    """从 YAML 文件加载配置"""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config
def load_environment(self):
    """
    加载环境
    """
    if self.real_robot:
        self.env = DeviceAloha(self.robot_env)
        self.env_max_reward = 0
    else:
        from sim_env import make_sim_env

        self.env = make_sim_env(self.task_name)
        self.env_max_reward = self.env.task.max_reward


def main(config):
    # 从 YAML 文件中加载所有配置
    #all_config = get_all_config(config, "eval")
    #set_seed(all_config["seed"])
   
    #ckpt_names = all_config["ckpt_names"]
    config = config
    _seed = config["seed"]
    robot_env = config["robot_env"]
    ckpt_dir = config["ckpt_dir"]
    ckpt_names = config["ckpt_names"]
    #save_episode = save_episode
    num_rollouts = config["num_rollouts"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    camera_names = config["camera_names"]
    max_timesteps = config["max_timesteps"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    onscreen_cam = "angle"
    policy_config = config["policy_config"]
    vq = config["policy_config"]["vq"]
    set_seed(_seed)  
        # self.actuator_config = config["actuator_config"]
        # self.use_actuator_net = self.actuator_config["actuator_network_dir"] is not None
    stats = None
    env = None
    env_max_reward = 0
    # 创建环境
    robot_env = config["robot_env"]
    #env_config = all_config["environments"]
    #env_maker = env_config.pop("environment_maker")
    #env = env_maker(all_config)  # 使用 all_config 提供更多灵活性
    env = DeviceAloha(robot_env)
    assert env is not None, "Environment is not created..."

    results = []
    # 多个检查点评估
    for ckpt_name in ckpt_names:
        success_rate, avg_return = eval_bc(config, ckpt_name, env)
        results.append([ckpt_name, success_rate, avg_return])

    for ckpt_name, success_rate, avg_return in results:
        logger.info(f"{ckpt_name}: {success_rate=} {avg_return=}")

    print()


def get_ckpt_path(ckpt_dir, ckpt_name, stats_path):
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    raw_ckpt_path = ckpt_path
    if not os.path.exists(ckpt_path):
        ckpt_dir = os.path.dirname(ckpt_dir)  # 检查上级目录
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        logger.warning(
            f"Warning: not found ckpt_path: {raw_ckpt_path}, try {ckpt_path}..."
        )
        if not os.path.exists(ckpt_path):
            ckpt_dir = os.path.dirname(stats_path)
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            logger.warning(
                f"Warning: also not found ckpt_path: {ckpt_path}, try {ckpt_path}..."
            )
    return ckpt_path

#原本
#def eval_bc(config, ckpt_name, env: CommonEnv):
def eval_bc(config, ckpt_name, env):
    # 显式获得配置
    ckpt_dir = config["ckpt_dir"]
    stats_path = config["stats_path"]
    save_dir = config["save_dir"]
    max_timesteps = config["max_timesteps"]
    camera_names = config["camera_names"]
    max_rollouts = config["num_rollouts"]
    policy_config: dict = config["policy_config"]
    state_dim = policy_config["state_dim"]
    action_dim = policy_config["action_dim"]
    #temporal_agg = policy_config["temporal_agg"]
    temporal_agg = False
    num_queries = policy_config["num_queries"]  # i.e. chunk_size
    dt = 1 / config["fps"]
    image_mode = config.get("image_mode", 0)
    save_all = config.get("save_all", False)
    save_time_actions = config.get("save_time_actions", False)
    filter_type = config.get("filter", None)
    ensemble: dict = config.get("ensemble", None)
    save_dir = save_dir if save_dir != "AUTO" else ckpt_dir
    result_prefix = "result_" + ckpt_name.split(".")[0]
    debug = config.get("debug", False)
    if debug:
        logger.setLevel(logging.DEBUG)
        from utils.visualization.ros1_logger import LoggerROS1

        ros1_logger = LoggerROS1("eval_debuger")

    # 获取检查点路径
    ckpt_path = get_ckpt_path(ckpt_dir, ckpt_name, stats_path)
    policy_config["ckpt_path"] = ckpt_path

    # 创建和配置策略
    policies: Dict[str, list] = {}
    if ensemble is None:
        logger.info("policy_config:", policy_config)
        policy_config["max_timesteps"] = max_timesteps  # TODO: remove this
        policy = make_policy(policy_config, "eval")
        policies["Group1"] = (policy,)
    else:
        logger.info("ensemble config:", ensemble)
        ensembler = ensemble.pop("ensembler")
        for gr_name, gr_cfgs in ensemble.items():
            policies[gr_name] = []
            for index, gr_cfg in enumerate(gr_cfgs):
                policies[gr_name].append(
                    make_policy(
                        gr_cfg["policies"][index]["policy_class"],
                    )
                )

    # 添加动作滤波器
    if filter_type is not None:
        from OneEuroFilter import OneEuroFilter

        config = {
            "freq": config["fps"],  # Hz
            "mincutoff": 0.01,  # Hz
            "beta": 0.05,
            "dcutoff": 0.5,  # Hz
        }
        filters = [OneEuroFilter(**config) for _ in range(action_dim)]

    # 初始化预处理和后处理函数
    use_stats = True
    if use_stats:
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
        post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
    else:
        pre_process = lambda s_qpos: s_qpos
        post_process = lambda a: a

    showing_images = config.get("show_images", False)

    def show_images(ts1_zjy):
        images: dict = ts1_zjy.observation["images"]
        for name, value in images.items():
            cv2.imshow(name, value)
        cv2.waitKey(1)

    # 评估循环
    if hasattr(policy, "eval"):
        policy.eval()
    env_max_reward = 0
    episode_returns = []
    highest_rewards = []
    num_rollouts = 0
    policy_sig = inspect.signature(policy).parameters
    prediction_freq = 100000
    for rollout_id in range(max_rollouts):
        all_time_actions = torch.zeros(
            [max_timesteps, max_timesteps + num_queries, action_dim]
        ).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []  # for visualization
        qpos_list = []
        action_list = []
        rewards = []
        with torch.inference_mode():
            logger.info("Reset environment...")
            ts1_zjy = env.reset(sleep_time=1)
            if showing_images:
                for _ in range(10):
                    show_images(ts1_zjy)
            logger.info(f"Current rollout: {rollout_id} for {ckpt_name}.")
            v = input(f"Press Enter to start evaluation or z and Enter to exit...")
            if v == "z":
                break
            ts1_zjy = env.reset()
            if hasattr(policy, "reset"):
                policy.reset()
            try:
                for t in tqdm(range(max_timesteps)):
                    start_time = time.time()
                    image_list.append(ts1_zjy.observation["images"])
                    if showing_images:
                        show_images(ts1_zjy)
                    obs = ts1_zjy.observation
                    curr_image = get_image(ts1_zjy, camera_names, image_mode)
                    qpos_numpy = np.array(ts1_zjy.observation["qpos"])

                    logger.debug(f"raw qpos: {qpos_numpy}")
                    qpos = pre_process(qpos_numpy)  # normalize qpos
                    logger.debug(f"pre qpos: {qpos}")
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    qpos_history[:, t] = qpos

                    logger.debug(f"observe time: {time.time() - start_time}")
                    start_time = time.time()
                    target_t = t % num_queries
                    if temporal_agg or target_t == 0:
                        all_actions: torch.Tensor = policy(qpos, curr_image)
                    all_time_actions[[t], t : t + num_queries] = all_actions
                    index = 0 if temporal_agg else target_t
                    raw_action = all_actions[:, index]

                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    logger.debug(f"raw action: {raw_action}")
                    action = post_process(raw_action)  # de-normalize action
                    if filter_type is not None:  # filt action
                        for i, filter in enumerate(filters):
                            action[i] = filter(action[i], time.time())
                    time.sleep(max(0, 1 / prediction_freq - (time.time() - start_time)))
                    logger.debug(f"prediction time: {time.time() - start_time}")
                    #0-5:右臂 6：右夹爪，7-12：左臂，13：左夹爪
                    ts1_zjy: dm_env.TimeStep = env.step(action)

                    qpos_list.append(qpos_numpy)
                    action_list.append(action)
                    rewards.append(ts1_zjy.reward)
            except KeyboardInterrupt:
                logger.info(f"Current roll out: {rollout_id} interrupted by CTRL+C...")
                continue
            else:
                num_rollouts += 1

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        logger.info(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
        )

        '''if save_dir != "":
            dataset_name = f"{result_prefix}_{rollout_id}"
            save_eval_results(
                save_dir,
                dataset_name,
                rollout_id,
                image_list,
                qpos_list,
                action_list,
                camera_names,
                dt,
                all_time_actions,
                save_all=save_all,
                save_time_actions=save_time_actions,
            )
	'''
    if num_rollouts > 0:
        success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
        avg_return = np.mean(episode_returns)
        summary_str = (
            f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
        )
        for r in range(env_max_reward + 1):
            more_or_equal_r = (np.array(highest_rewards) >= r).sum()
            more_or_equal_r_rate = more_or_equal_r / num_rollouts
            summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

        logger.info(summary_str)

        if save_dir != "":
            with open(os.path.join(save_dir, dataset_name + ".txt"), "w") as f:
                f.write(summary_str)
                f.write(repr(episode_returns))
                f.write("\n\n")
                f.write(repr(highest_rewards))
            logger.info(
                f'Success rate and average return saved to {os.path.join(save_dir, dataset_name + ".txt")}'
            )
    else:
        success_rate = 0
        avg_return = 0
    if showing_images:
        cv2.destroyAllWindows()
    return success_rate, avg_return

class DeviceAloha:
    def __init__(self, aloha_config):
        """
        初始化设备

        Args:
            device_name (str): 设备名称
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
        self.camera_left = RealSenseCamera(config_head_camera, False)
        self.camera_right = RealSenseCamera(config_bottom_camera, False)
        self.camera_bottom = RealSenseCamera(config_left_camera, False)
        self.camera_top = RealSenseCamera(config_right_camera, False)
        self.camera_left.start_camera()
        self.camera_right.start_camera()
        self.camera_bottom.start_camera()
        self.camera_top.start_camera()
        self.left_arm = RmArm('/home/rm/aloha/shadow_rm_aloha/config/rm_left_arm.yaml')
        self.right_arm = RmArm('/home/rm/aloha/shadow_rm_aloha/config/rm_right_arm.yaml')

    def close(self):
        """
        关闭摄像头
        """
        self.camera_left.close()
        self.camera_right.close()
        self.camera_bottom.close()
        self.camera_top.close()

    def get_qps(self):
        """
        获取关节角度

        Returns:
            np.array: 关节角度
        """
        left_slave_arm_angle = self.arm_left.get_joint_angle()
        left_joint_angles_array = np.array(list(left_slave_arm_angle.values()))
        right_slave_arm_angle = self.arm_right.get_joint_angle()
        right_joint_angles_array = np.array(list(right_slave_arm_angle.values()))
        return np.concatenate([left_joint_angles_array, right_joint_angles_array])

    def get_qvel(self):
        """
        获取关节速度

        Returns:
            np.array: 关节速度
        """
        left_slave_arm_velocity = self.arm_left.get_joint_velocity()
        left_joint_velocity_array = np.array(list(left_slave_arm_velocity.values()))
        right_slave_arm_velocity = self.arm_right.get_joint_velocity()
        right_joint_velocity_array = np.array(list(right_slave_arm_velocity.values()))
        return np.concatenate([left_joint_velocity_array, right_joint_velocity_array])

    def get_effort(self):
        """
        获取关节力

        Returns:
            np.array: 关节力
        """
        left_slave_arm_effort = self.arm_left.get_joint_effort()
        left_joint_effort_array = np.array(list(left_slave_arm_effort.values()))
        right_slave_arm_effort = self.arm_right.get_joint_effort()
        right_joint_effort_array = np.array(list(right_slave_arm_effort.values()))
        return np.concatenate([left_joint_effort_array, right_joint_effort_array])

    def get_images(self):
        """
        获取图像

        Returns:
            dict: 图像字典
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
            "cam_left": self.left_image,
            "cam_low": self.bottom_image,
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

    def step(self, action):
        #先注释掉，防止机器人撞
        #self.left_arm.set_joint_canfd_position(action[0:6])
        #self.left_arm.set_gripper_position(action[6])
            # 控制右臂
        #self.right_arm.set_joint_canfd_position(action[7:13])
        #self.right_arm.set_gripper_position(action[13])
        #logging.info(f'control time: {time.time() - time0}')

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=0,
            discount=None,
            observation=self.get_observation(),
        )


if __name__ == "__main__":
    # 从 YAML 文件加载配置
    yaml_path = "/home/rm/aloha/shadow_rm_act/config/config_eval.yaml"  # 替换为你的 YAML 文件路径
    config = load_config_from_yaml(yaml_path)
    main(config)
