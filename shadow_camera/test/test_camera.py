import os
import cv2
import time
import numpy as np
from os import path
import pyrealsense2 as rs
from shadow_camera import realsense
import logging



def test_camera():
    camera = realsense.RealSenseCamera('241122071186')
    camera.start_camera()

    while True:
        # result = camera.read_align_frame()
        # if result is None:
        #     print('is None')
        #     continue
        # start_time = time.time()
        color_image, depth_image, colorized_depth, vtx = camera.read_frame()
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        print(f"color_image: {color_image.shape}")
        # print(f"Time: {end_time - start_time}")
        cv2.imshow("bgr_image", color_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    camera.stop_camera()


def test_get_serial_num():
    camera = realsense.RealSenseCamera()
    device = camera.get_serial_num()


class CameraCapture:
    def __init__(self, camera_serial_num=None, save_dir="./save"):
        self._camera_serial_num = camera_serial_num
        self._color_save_dir = path.join(save_dir, "color")
        self._depth_save_dir = path.join(save_dir, "depth")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(self._color_save_dir, exist_ok=True)
        os.makedirs(self._depth_save_dir, exist_ok=True)

    def get_serial_num(self):
        self._camera_serial_num = {}
        camera_names = ["left", "right", "head", "table"]
        context = rs.context()
        devices = context.query_devices()  # 获取所有设备
        if len(context.devices) > 0:
            for i, device in enumerate(devices):
                self._camera_serial_num[camera_names[i]] = device.get_info(
                    rs.camera_info.serial_number
                )
        print(self._camera_serial_num)

        return self._camera_serial_num

    def start_camera(self):
        if self._camera_serial_num is None:
            self.get_serial_num()
        self._camera_left = realsense.RealSenseCamera(self._camera_serial_num["left"])
        self._camera_right = realsense.RealSenseCamera(self._camera_serial_num["right"])
        self._camera_head = realsense.RealSenseCamera(self._camera_serial_num["head"])

        self._camera_left.start_camera()
        self._camera_right.start_camera()
        self._camera_head.start_camera()

    def stop_camera(self):
        self._camera_left.stop_camera()
        self._camera_right.stop_camera()
        self._camera_head.stop_camera()

    def _save_datas(self, timestamp, color_image, depth_image, camera_name):
        color_filename = path.join(
            self._color_save_dir, f"{timestamp}" + camera_name + ".jpg"
        )
        depth_filename = path.join(
            self._depth_save_dir, f"{timestamp}" + camera_name + ".png"
        )
        cv2.imwrite(color_filename, color_image)
        cv2.imwrite(depth_filename, depth_image)

    def capture_images(self):
        while True:
            (
                color_image_left,
                depth_image_left,
                _,
                _,
            ) = self._camera_left.read_align_frame()
            (
                color_image_right,
                depth_image_right,
                _,
                _,
            ) = self._camera_right.read_align_frame()
            (
                color_image_head,
                depth_image_head,
                _,
                point_cloud3,
            ) = self._camera_head.read_align_frame()

            bgr_color_image_left = cv2.cvtColor(color_image_left, cv2.COLOR_RGB2BGR)
            bgr_color_image_right = cv2.cvtColor(color_image_right, cv2.COLOR_RGB2BGR)
            bgr_color_image_head = cv2.cvtColor(color_image_head, cv2.COLOR_RGB2BGR)

            timestamp = time.time() * 1000

            cv2.imshow("Camera left", bgr_color_image_left)
            cv2.imshow("Camera right", bgr_color_image_right)
            cv2.imshow("Camera head", bgr_color_image_head)

            # self._save_datas(
            #     timestamp, bgr_color_image_left, depth_image_left, "left"
            # )
            # self._save_datas(
            #     timestamp, bgr_color_image_right, depth_image_right, "right"
            # )
            # self._save_datas(
            #     timestamp, bgr_color_image_head, depth_image_head, "head"
            # )
            self._save_datas(
                 timestamp, bgr_color_image_left, depth_image_left, "left"
             )
            self._save_datas(
                 timestamp, bgr_color_image_right, depth_image_right, "right"
            )
            self._save_datas(
                timestamp, bgr_color_image_head, depth_image_head, "head"
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    #test_camera()
    test_get_serial_num()
    """
    输入相机序列号制定左右相机：
    dict：{'left': '241222075132', 'right': '242322076532', 'head': '242322076532'}
    保存路径：
    str：./save
    输入为空，自动分配相机序列号（不指定左、右、头部），保存路径为./save
    """

    capture = CameraCapture()
    capture.get_serial_num()
    test_get_serial_num()

    capture.start_camera()
    capture.capture_images()
    capture.stop_camera()
    
    
    
    
