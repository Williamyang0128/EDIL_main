import pytest
import pyrealsense2 as rs
import cv2
import matplotlib.pyplot as plt
from shadow_camera.realsense import RealSenseCamera

class TestRealSenseCamera:
    @pytest.fixture(autouse=True)
    def setup_camera(self):
        self.camera = RealSenseCamera()

    def test_get_serial_num(self):
        serial_nums = self.camera.get_serial_num()
        assert isinstance(serial_nums, dict)
        assert len(serial_nums) > 0

    def test_start_stop_camera(self):
        self.camera.start_camera()
        assert self.camera.camera_on is True
        self.camera.stop_camera()
        assert self.camera.camera_on is False

    def test_set_resolution(self):
        color_resolution = [1280, 720]
        depth_resolution = [1280, 720]
        self.camera.set_resolution(color_resolution, depth_resolution)
        assert self.camera._color_resolution == color_resolution
        assert self.camera._depth_resolution == depth_resolution

    def test_set_frame_rate(self):
        color_fps = 60
        depth_fps = 60
        self.camera.set_frame_rate(color_fps, depth_fps)
        assert self.camera._color_frames_rate == color_fps
        assert self.camera._depth_frames_rate == depth_fps

    def test_read_frame(self):
        self.camera.start_camera()
        color_image, depth_image, colorized_depth, point_cloud = self.camera.read_frame()
        assert color_image is not None
        assert depth_image is not None

        # 使用 matplotlib 显示颜色图像
        plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        plt.title('Color Image')
        plt.show()

        self.camera.stop_camera()

    def test_read_align_frame(self):
        self.camera.start_camera()
        color_image, depth_image, colorized_depth, point_cloud = self.camera.read_align_frame()
        assert color_image is not None
        assert depth_image is not None

        # 使用 matplotlib 显示颜色图像
        plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        plt.title('Aligned Color Image')
        plt.show()

        self.camera.stop_camera()

    def test_get_camera_intrinsics(self):
        self.camera.start_camera()
        color_intrinsics, depth_intrinsics = self.camera.get_camera_intrinsics()
        assert color_intrinsics is not None
        assert depth_intrinsics is not None
        self.camera.stop_camera()

    def test_get_3d_camera_coordinate(self):
        self.camera.start_camera()
        # 先调用 read_align_frame 方法以确保 _aligned_depth_frame 被设置
        self.camera.read_align_frame()
        depth_pixel = [320, 240]
        distance, camera_coordinate = self.camera.get_3d_camera_coordinate(depth_pixel, align=True)
        assert distance > 0
        assert len(camera_coordinate) == 3
        self.camera.stop_camera()

# 运行测试
if __name__ == '__main__':
    pytest.main([__file__])
