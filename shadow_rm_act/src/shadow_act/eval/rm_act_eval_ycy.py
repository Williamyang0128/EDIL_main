import os
import time
import yaml
import torch
import pickle
import dm_env
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
import csv

# 配置logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# # 隐藏h5py的警告Creating converter from 7 to 5
# logging.getLogger("h5py").setLevel(logging.WARNING)


class RmActEvaluator:
    def __init__(self, config, save_episode=True, num_rollouts=50):
        """
        初始化Evaluator类

        Args:
            config (dict): 配置字典
            checkpoint_name (str): 检查点名称
            save_episode (bool): 是否保存每个episode
            num_rollouts (int): 滚动次数
        """
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
        # self.actuator_config = config["actuator_config"]
        # self.use_actuator_net = self.actuator_config["actuator_network_dir"] is not None
        self.stats = None
        self.env = None
        self.env_max_reward = 0

    def _make_policy(self, policy_class, policy_config):
        """
        根据策略类和配置创建策略对象

        Args:
            policy_class (str): 策略类名称
            policy_config (dict): 策略配置字典

        Returns:
            policy: 创建的策略对象
        """
        if policy_class == "ACT":
            return ACTPolicy(policy_config)
        elif policy_class == "CNNMLP":
            return CNNMLPPolicy(policy_config)
        elif policy_class == "Diffusion":
            return DiffusionPolicy(policy_config)
        else:
            raise NotImplementedError(f"Policy class {policy_class} is not implemented")

    def load_policy_and_stats(self):
        """
        加载策略和统计数据
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)
        logging.info(f"Loading policy from: {checkpoint_path}")
        self.policy = self._make_policy(self.policy_class, self.policy_config)
        # 加载模型并设置为评估模式
        #self.policy.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cuda:0")))
        self.policy.cuda()
        self.policy.eval()

        if self.vq:
            vq_dim = self.config["policy_config"]["vq_dim"]
            vq_class = self.config["policy_config"]["vq_class"]
            self.latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
            latent_model_checkpoint_path = os.path.join(
                self.checkpoint_dir, "latent_model_last.ckpt"
            )
            self.latent_model.deserialize(torch.load(latent_model_checkpoint_path))
            self.latent_model.eval()
            self.latent_model.cuda()
            logging.info(
                f"Loaded policy from: {checkpoint_path}, latent model from: {latent_model_checkpoint_path}"
            )
        else:
            logging.info(f"Loaded: {checkpoint_path}")

        stats_path = os.path.join(self.checkpoint_dir, "dataset_stats.pkl")
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)

    def pre_process(self, state_qpos):
        """
        预处理状态位置

        Args:
            state_qpos (np.array): 状态位置数组

        Returns:
            np.array: 预处理后的状态位置
        """
        if self.policy_class == "Diffusion":
            return ((state_qpos + 1) / 2) * (
                self.stats["action_max"] - self.stats["action_min"]
            ) + self.stats["action_min"]
        # 标准化处理，均值为 0，标准差为 1

        return (state_qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]

    def post_process(self, action):
        """
        后处理动作

        Args:
            action (np.array): 动作数组

        Returns:
            np.array: 后处理后的动作
        """
        # 反标准化处理
        return action * self.stats["action_std"] + self.stats["action_mean"]

    def get_image_torch(self, timestep, camera_names, random_crop_resize=False):
        """
        获取图像

        Args:
            timestep (object): 时间步对象
            camera_names (list): 相机名称列表
            random_crop_resize (bool): 是否随机裁剪和调整大小

        Returns:
            torch.Tensor: 处理后的图像，归一化(num_cameras, channels, height, width)
        """
        current_images = []
        for cam_name in camera_names:
            current_image = rearrange(
                timestep.observation["images"][cam_name], "h w c -> c h w"
            )
            current_images.append(current_image)
        current_image = np.stack(current_images, axis=0)
        current_image = (
            torch.from_numpy(current_image / 255.0).float().cuda().unsqueeze(0)
        )

        if random_crop_resize:
            logging.info("Random crop resize is used!")
            original_size = current_image.shape[-2:]
            ratio = 0.95
            current_image = current_image[
                ...,
                int(original_size[0] * (1 - ratio) / 2) : int(
                    original_size[0] * (1 + ratio) / 2
                ),
                int(original_size[1] * (1 - ratio) / 2) : int(
                    original_size[1] * (1 + ratio) / 2
                ),
            ]
            current_image = current_image.squeeze(0)
            resize_transform = transforms.Resize(original_size, antialias=True)
            current_image = resize_transform(current_image)
            current_image = current_image.unsqueeze(0)

        return current_image

    def load_environment(self):
        """
        加载环境
        """
        if self.real_robot:
            self.env = DeviceAloha(self.robot_env)
            self.env_max_reward = 0
        else:
            from sim_env import make_sim_envi

            self.env = make_sim_env(self.task_name)
            self.env_max_reward = self.env.task.max_reward

    def get_auto_index(self, checkpoint_dir):
        max_idx = 1000
        for i in range(max_idx + 1):
            if not os.path.isfile(os.path.join(checkpoint_dir, f"qpos_{i}.npy")):
                return i
        raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

    def evaluate(self, checkpoint_name=None):
        """
        评估策略

        Returns:
            tuple: 成功率和平均回报
        """
        if checkpoint_name is not None:
            self.checkpoint_name = checkpoint_name
        set_seed(self._seed)  # np与torch的随机种子
        self.load_policy_and_stats()
        self.load_environment()

        # 创建用于存储结果的目录
        results_dir = '/home/rm/aloha/shadow_rm_act/src/shadow_act/eval/result_gripper'
        os.makedirs(results_dir, exist_ok=True)
        logging.info(f"动作数据将保存到: {results_dir}")

        query_frequency = self.policy_config["num_queries"]

        # 时间聚合时，每个时间步只有1个查询
        if self.temporal_agg:
            query_frequency = 1
            num_queries = self.policy_config["num_queries"]

        # # 真实机器人时，基础延迟为13？？？
        # if self.real_robot:
        #     BASE_DELAY = 13
        #     # query_frequency -= BASE_DELAY

        max_timesteps = int(self.max_timesteps * 1)  # may increase for real-world tasks
        episode_returns = []
        highest_rewards = []

        for rollout_id in range(2):
            # 为每个rollout设置CSV文件
            csv_filename = os.path.join(results_dir, f'actions_rollout_{rollout_id}.csv')
            csv_header = [
                'timestep', 
                'left_joint_1', 'left_joint_2', 'left_joint_3', 'left_joint_4', 'left_joint_5', 'left_joint_6', 'left_gripper',
                'right_joint_1', 'right_joint_2', 'right_joint_3', 'right_joint_4', 'right_joint_5', 'right_joint_6', 'right_gripper'
            ]
            
            with open(csv_filename, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(csv_header)

                timestep = self.env.reset()

                if self.onscreen_render:
                    # TODO 画图
                    pass
                if self.temporal_agg:
                    all_time_actions = torch.zeros(
                        [max_timesteps, max_timesteps + num_queries, self.state_dim]
                    ).cuda()
                qpos_history_raw = np.zeros((max_timesteps, self.state_dim))
                rewards = []

                with torch.inference_mode():
                    time_0 = time.time()
                    DT = 1 / 8
                    culmulated_delay = 0
                    for t in range(max_timesteps):
                        time_1 = time.time()
                        if self.onscreen_render:
                            # TODO 显示图像
                            pass
                        # process previous timestep to get qpos and image_list
                        obs = timestep.observation
                        qpos_numpy = np.array(obs["qpos"])
                        qpos_history_raw[t] = qpos_numpy
                        qpos = self.pre_process(qpos_numpy)
                        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

                        logging.info(f"t{t}")

                        if t % query_frequency == 0:
                            current_image = self.get_image_torch(
                                timestep,
                                self.camera_names,
                                random_crop_resize=(
                                    self.config["policy_class"] == "Diffusion"
                                ),
                            )

                        if t == 0:
                            # 网络预热
                            for _ in range(10):
                                self.policy(qpos, current_image)
                            logging.info("Network warm up done")

                        if self.config["policy_class"] == "ACT":
                            if t % query_frequency == 0:
                                if self.vq:
                                    if rollout_id == 0:
                                        for _ in range(10):
                                            vq_sample = self.latent_model.generate(
                                                1, temperature=1, x=None
                                            )
                                            logging.info(
                                                torch.nonzero(vq_sample[0])[:, 1]
                                                .cpu()
                                                .numpy()
                                            )
                                    vq_sample = self.latent_model.generate(
                                        1, temperature=1, x=None
                                    )
                                    all_actions = self.policy(
                                        qpos, current_image, vq_sample=vq_sample
                                    )
                                else:
                                    all_actions = self.policy(qpos, current_image)
                                # if self.real_robot:
                                #     all_actions = torch.cat(
                                #         [
                                #             all_actions[:, :-BASE_DELAY, :-2],
                                #             all_actions[:, BASE_DELAY:, -2:],
                                #         ],
                                #         dim=2,
                                #     )
                            if self.temporal_agg:
                                all_time_actions[[t], t : t + num_queries] = all_actions
                                actions_for_curr_step = all_time_actions[:, t]
                                actions_populated = torch.all(
                                    actions_for_curr_step != 0, axis=1
                                )
                                actions_for_curr_step = actions_for_curr_step[
                                    actions_populated
                                ]
                                k = 0.01
                                exp_weights = np.exp(
                                    -k * np.arange(len(actions_for_curr_step))
                                )
                                exp_weights = exp_weights / exp_weights.sum()
                                exp_weights = (
                                    torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                                )
                                raw_action = (actions_for_curr_step * exp_weights).sum(
                                    dim=0, keepdim=True
                                )
                            else:
                                raw_action = all_actions[:, t % query_frequency]
                        elif self.config["policy_class"] == "Diffusion":
                            if t % query_frequency == 0:
                                all_actions = self.policy(qpos, current_image)
                                # if self.real_robot:
                                #     all_actions = torch.cat(
                                #         [
                                #             all_actions[:, :-BASE_DELAY, :-2],
                                #             all_actions[:, BASE_DELAY:, -2:],
                                #         ],
                                #         dim=2,
                                #     )
                            raw_action = all_actions[:, t % query_frequency]
                        elif self.config["policy_class"] == "CNNMLP":
                            raw_action = self.policy(qpos, current_image)
                            all_actions = raw_action.unsqueeze(0)
                        else:
                            raise NotImplementedError

                        ### post-process actions
                        raw_action = raw_action.squeeze(0).cpu().numpy()
                        action = self.post_process(raw_action)

                        # 将action写入CSV文件
                        row_to_write = [t] + action.tolist()
                        csv_writer.writerow(row_to_write)

                        ### step the environment
                        if self.real_robot:
                            logging.info(f"timestep: {t}, action = {action}")
                            timestep = self.env.step(action)

                        rewards.append(timestep.reward)
                        duration = time.time() - time_1
                        sleep_time = max(0, DT - duration)
                        time.sleep(sleep_time)
                        if duration >= DT:
                            culmulated_delay += duration - DT
                            logging.warning(
                                f"Warning: step duration: {duration:.3f} s at step {t} longer than DT: {DT} s, culmulated delay: {culmulated_delay:.3f} s"
                            )

                    logging.info(f"Avg fps {max_timesteps / (time.time() - time_0)}")
                    plt.close()

            if self.real_robot:
                log_id = self.get_auto_index(self.checkpoint_dir)
                np.save(
                    os.path.join(self.checkpoint_dir, f"qpos_{log_id}.npy"),
                    qpos_history_raw,
                )
                plt.figure(figsize=(10, 20))
                for i in range(self.state_dim):
                    plt.subplot(self.state_dim, 1, i + 1)
                    plt.plot(qpos_history_raw[:, i])
                    if i != self.state_dim - 1:
                        plt.xticks([])
                plt.tight_layout()
                plt.savefig(os.path.join(self.checkpoint_dir, f"qpos_{log_id}.png"))
                plt.close()

            rewards = np.array(rewards)
            episode_return = np.sum(rewards[rewards != None])
            episode_returns.append(episode_return)
            episode_highest_reward = np.max(rewards)
            highest_rewards.append(episode_highest_reward)
            logging.info(
                f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {self.env_max_reward=}, Success: {episode_highest_reward == self.env_max_reward}"
            )

        success_rate = np.mean(np.array(highest_rewards) == self.env_max_reward)
        avg_return = np.mean(episode_returns)
        summary_str = (
            f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
        )
        for r in range(self.env_max_reward + 1):
            more_or_equal_r = (np.array(highest_rewards) >= r).sum()
            more_or_equal_r_rate = more_or_equal_r / self.num_rollouts
            summary_str += f"Reward >= {r}: {more_or_equal_r}/{self.num_rollouts} = {more_or_equal_r_rate * 100}%\n"

        logging.info(summary_str)

        result_file_name = "result_" + self.checkpoint_name.split(".")[0] + ".txt"
        with open(os.path.join(self.checkpoint_dir, result_file_name), "w") as f:
            f.write(summary_str)
            f.write(repr(episode_returns))
            f.write("\n\n")
            f.write(repr(highest_rewards))

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

# 在DeviceAloha中，step方法
# target_angle是14维的，前7维是左臂，后7维是右臂
    def step(self, target_angle):
        self.arm_left.set_joint_canfd_position(target_angle[0:self.arm_axis])
        self.arm_right.set_joint_canfd_position(target_angle[self.arm_axis+1:self.arm_axis*2+1])
        self.arm_left.set_gripper_position(target_angle[self.arm_axis])
        self.arm_right.set_gripper_position(target_angle[(self.arm_axis*2 + 1)])
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=0,
            discount=None,
            observation=self.get_observation(),
        )


if __name__ == "__main__":
    # with open("/home/rm/code/shadow_act/config/config.yaml", "r") as f:
    #     config = yaml.safe_load(f)
    # aloha_config = config["robot_env"]
    # device = DeviceAloha(aloha_config)
    # device.reset()
    # while True:
    #     init_angle = np.concatenate([device.init_left_arm_angle, device.init_right_arm_angle])
    #     time_step = time.time()
    #     timestep = device.step(init_angle)
    #     logging.info(f"Time: {time.time() - time_step}")
    #     obs = timestep.observation

    with open("/home/rm/collect_data/Tools/dpkt/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # logging.info(f"Config: {config}")
    evaluator = RmActEvaluator(config)
    try:
        success_rate, avg_return = evaluator.evaluate()
    finally:
        if evaluator.real_robot and evaluator.env is not None:
            logging.info("Closing environment...")
            evaluator.env.close()
            logging.info("Environment closed.")

