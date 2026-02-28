import time
import logging
import numpy as np
import pyrealsense2 as rs
from shadow_camera import base_camera

# 设置日志配置
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class RealSenseCamera(base_camera.BaseCamera):
    """Intel RealSense相机类"""

    def __init__(self, serial_num=None, is_depth_frame=False):
        """
        初始化相机对象
        :param serial_num: 相机序列号，默认为None
        """
        super().__init__()
        self._color_resolution = [640, 480]
        self._depth_resolution = [640, 480]
        self._color_frames_rate = 30
        self._depth_frames_rate = 15
        self.timestamp = 0
        self.color_timestamp = 0
        self.depth_timestamp = 0
        self._colorizer = rs.colorizer()
        self._config = rs.config()
        self.is_depth_frame = is_depth_frame
        self.camera_on = False
        self.serial_num = serial_num

    def get_serial_num(self):
        serial_num = {}
        context = rs.context()
        devices = context.query_devices()  # 获取所有设备
        if len(context.devices) > 0:
            for i, device in enumerate(devices):
                serial_num[i] = device.get_info(rs.camera_info.serial_number)

        logging.info(f"Detected serial numbers: {serial_num}")
        return serial_num

    def _set_config(self):
        if self.serial_num is not None:
            logging.info(f"Setting device with serial number: {self.serial_num}")
            self._config.enable_device(self.serial_num)

        self._config.enable_stream(
            rs.stream.color,
            self._color_resolution[0],
            self._color_resolution[1],
            rs.format.rgb8,
            self._color_frames_rate,
        )
        if self.is_depth_frame:
            self._config.enable_stream(
                rs.stream.depth,
                self._depth_resolution[0],
                self._depth_resolution[1],
                rs.format.z16,
                self._depth_frames_rate,
            )

    def start_camera(self):
        """
        启动相机并获取内参信息,如果后续调用帧对齐,则内参均为彩色内参
        """
        self._pipeline = rs.pipeline()
        
        if self.is_depth_frame:
            self.point_cloud = rs.pointcloud()
            self._align = rs.align(rs.stream.color)
        self._set_config()

        self.profile = self._pipeline.start(self._config)

        if self.is_depth_frame:
            self._depth_intrinsics = (
                self.profile.get_stream(rs.stream.depth)
                .as_video_stream_profile()
                .get_intrinsics()
            )
 
        self._color_intrinsics = (
            self.profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.camera_on = True
        logging.info("Camera started successfully")
        logging.info(
            f"Camera started with color resolution: {self._color_resolution}, depth resolution: {self._depth_resolution}"
        )
        logging.info(
            f"Color FPS: {self._color_frames_rate}, Depth FPS: {self._depth_frames_rate}"
        )

    def stop_camera(self):
        """
        停止相机
        """
        self._pipeline.stop()
        self.camera_on = False
        logging.info("Camera stopped")

    def set_resolution(self, color_resolution, depth_resolution):
        self._color_resolution = color_resolution
        self._depth_resolution = depth_resolution
        logging.info(
            "Optional color resolution:"
            "[320, 180] [320, 240] [424, 240] [640, 360] [640, 480]"
            "[848, 480] [960, 540] [1280, 720] [1920, 1080]"
        )
        logging.info(
            "Optional depth resolution:"
            "[256, 144] [424, 240] [480, 270] [640, 360] [640, 400]"
            "[640, 480] [848, 100] [848, 480] [1280, 720] [1280, 800]"
        )
        logging.info(f"Set color resolution to: {color_resolution}")
        logging.info(f"Set depth resolution to: {depth_resolution}")

    def set_frame_rate(self, color_fps, depth_fps):
        self._color_frames_rate = color_fps
        self._depth_frames_rate = depth_fps
        logging.info("Optional color fps: 6 15 30 60 ")
        logging.info("Optional depth fps: 6 15 30 60 90 100 300")
        logging.info(f"Set color FPS to: {color_fps}")
        logging.info(f"Set depth FPS to: {depth_fps}")

    # TODO: 调节白平衡进行补偿
    # def set_exposure(self, exposure):

    def read_frame(self, is_color=True, is_depth=True, is_colorized_depth=False, is_point_cloud=False):
        """
        读取一帧彩色图像和深度图像
        :return: 彩色图像和深度图像的NumPy数组
        """
        while not self.camera_on:
            time.sleep(0.5)
        color_image = None
        depth_image = None
        colorized_depth = None
        point_cloud = None
        try:
            frames = self._pipeline.wait_for_frames()
            if is_color:
                color_frame = frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
            else:
                color_image = None

            if is_depth:
                depth_frame = frames.get_depth_frame()
                depth_image = np.asanyarray(depth_frame.get_data())
            else:
                depth_image = None

            colorized_depth = (
                np.asanyarray(self._colorizer.colorize(depth_frame).get_data())
                if is_colorized_depth
                else None
            )
            point_cloud = (
                np.asanyarray(self.point_cloud.calculate(depth_frame).get_vertices())
                if is_point_cloud
                else None
            )
            # 获取时间戳单位为ms，对齐后color时间戳 > depth = aligned，选择color
            self.color_timestamp = color_frame.get_timestamp()
            if self.is_depth_frame:
                self.depth_timestamp = depth_frame.get_timestamp()

        except Exception as e:
            logging.warning(e)
            if "Frame didn't arrive within 5000" in str(e):
                logging.warning("Frame didn't arrive within 5000ms, resetting device")
                self.stop_camera()
                self.start_camera()

        return color_image, depth_image, colorized_depth, point_cloud

    def read_align_frame(self, is_color=True, is_depth=True, is_colorized_depth=False, is_point_cloud=False):
        """
        读取一帧对齐的彩色图像和深度图像
        :return: 彩色图像和深度图像的NumPy数组
        """
        while not self.camera_on:
            time.sleep(0.5)
        try:
            frames = self._pipeline.wait_for_frames()
            aligned_frames = self._align.process(frames)
            aligned_color_frame = aligned_frames.get_color_frame()
            self._aligned_depth_frame = aligned_frames.get_depth_frame()

            color_image = np.asanyarray(aligned_color_frame.get_data())
            depth_image = np.asanyarray(self._aligned_depth_frame.get_data())
            colorized_depth = (
                np.asanyarray(
                    self._colorizer.colorize(self._aligned_depth_frame).get_data()
                )
                if is_colorized_depth
                else None
            )

            if is_point_cloud:
                points = self.point_cloud.calculate(self._aligned_depth_frame)
                # 将元组数据转换为 NumPy 数组
                point_cloud = np.array(
                    [[point[0], point[1], point[2]] for point in points.get_vertices()]
                )
            else:
                point_cloud = None

            # 获取时间戳单位为ms，对齐后color时间戳 > depth = aligned，选择color
            self.timestamp = aligned_color_frame.get_timestamp()

            return color_image, depth_image, colorized_depth, point_cloud

        except Exception as e:
            if "Frame didn't arrive within 5000" in str(e):
                logging.warning("Frame didn't arrive within 5000ms, resetting device")
                self.stop_camera()
                self.start_camera()
                # device = self.profile.get_device()
                # device.hardware_reset()

    def get_camera_intrinsics(self):
        """
        获取彩色图像和深度图像的内参信息
        :return: 彩色图像和深度图像的内参信息
        """
        # 宽高：.width, .height; 焦距：.fx, .fy; 像素坐标：.ppx, .ppy; 畸变系数：.coeffs
        logging.info("Getting camera intrinsics")
        logging.info(
            "Width and height: .width, .height; Focal length: .fx, .fy; Pixel coordinates: .ppx, .ppy; Distortion coefficient: .coeffs"
        )
        return self._color_intrinsics, self._depth_intrinsics

    def get_3d_camera_coordinate(self, depth_pixel, align=False):
        """
        获取深度相机坐标系下的三维坐标
        :param depth_pixel:深度像素坐标
        :param align: 是否对齐

        :return: 深度值和相机坐标
        """
        if not hasattr(self, "_aligned_depth_frame"):
            raise AttributeError(
                "Aligned depth frame not set. Call read_align_frame() first."
            )

        distance = self._aligned_depth_frame.get_distance(
            depth_pixel[0], depth_pixel[1]
        )
        intrinsics = self._color_intrinsics if align else self._depth_intrinsics
        camera_coordinate = rs.rs2_deproject_pixel_to_point(
            intrinsics, depth_pixel, distance
        )
        return distance, camera_coordinate


if __name__ == "__main__":
    camera = RealSenseCamera(is_depth_frame=False)
    camera.get_serial_num()
    camera.start_camera()
    # camera.set_frame_rate(60, 60)
    color_image, depth_image, colorized_depth, point_cloud = camera.read_frame()
    camera.stop_camera()
    logging.info(f"Color image shape: {color_image.shape}")
    # logging.info(f"Depth image shape: {depth_image.shape}")
    # logging.info(f"Colorized depth image shape: {colorized_depth.shape}")
    # logging.info(f"Point cloud shape: {point_cloud.shape}")
    logging.info(f"Color timestamp: {camera.timestamp}")
    # logging.info(f"Depth timestamp: {camera.depth_timestamp}")
    logging.info(f"Color timestamp: {camera.color_timestamp}")
    # logging.info(f"Depth timestamp: {camera.depth_timestamp}")
    logging.info("Test passed")
