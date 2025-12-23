"""
全景图处理模块
功能：从全景图中提取局部视口，转换为标准针孔相机视图
适用于：参考图为360°全景图的场景
"""
import cv2
import numpy as np
import math
import os
from typing import Tuple


class PanoramaProcessor:
    """全景图处理器"""

    def __init__(self):
        """初始化"""
        pass

    def extract_equirectangular_viewport(
        self,
        pano_image: np.ndarray,
        hfov_deg: float = 90.0,
        vfov_deg: float = 60.0,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        output_width: int = None,
        output_height: int = None
    ) -> np.ndarray:
        """
        从等距柱状投影（Equirectangular）全景图中提取一个透视投影视口

        参数:
            pano_image: 全景图像 (H, W, C)，等距柱状投影格式
            hfov_deg: 输出视口的水平视场角（度）
            vfov_deg: 输出视口的垂直视场角（度）
            yaw_deg: 视口中心的偏航角/方位角（度，0-360，0=正前方）
            pitch_deg: 视口中心的俯仰角（度，-90到90，0=水平）
            output_width: 输出图像宽度，None则自动计算
            output_height: 输出图像高度，None则自动计算

        返回:
            viewport: 提取的透视视口图像
        """
        pano_h, pano_w = pano_image.shape[:2]

        # 自动计算输出尺寸
        if output_width is None:
            # 根据HFOV和全景图宽度估算合理的输出宽度
            output_width = int(pano_w * hfov_deg / 360.0)

        if output_height is None:
            # 根据HFOV和VFOV的比例计算高度
            output_height = int(output_width * math.tan(math.radians(vfov_deg / 2))
                              / math.tan(math.radians(hfov_deg / 2)))

        # 创建输出图像的像素坐标网格
        x = np.arange(output_width)
        y = np.arange(output_height)
        xx, yy = np.meshgrid(x, y)

        # 归一化到 [-0.5, 0.5] 范围
        px = (xx - output_width / 2.0) / output_width
        py = (yy - output_height / 2.0) / output_height

        # 计算针孔相机的射线方向（透视投影）
        # 假设焦距 f = W / (2 * tan(hfov/2))
        f = output_width / (2.0 * math.tan(math.radians(hfov_deg / 2)))

        # 射线方向（相机坐标系）
        ray_x = px * output_width
        ray_y = py * output_height
        ray_z = f * np.ones_like(ray_x)

        # 归一化射线
        norm = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
        ray_x /= norm
        ray_y /= norm
        ray_z /= norm

        # 应用旋转（yaw 和 pitch）
        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)

        # 绕Y轴旋转（yaw）
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        ray_x_rot = ray_x * cos_yaw + ray_z * sin_yaw
        ray_z_rot = -ray_x * sin_yaw + ray_z * cos_yaw

        # 绕X轴旋转（pitch）
        cos_pitch = math.cos(pitch_rad)
        sin_pitch = math.sin(pitch_rad)

        ray_y_rot = ray_y * cos_pitch - ray_z_rot * sin_pitch
        ray_z_final = ray_y * sin_pitch + ray_z_rot * cos_pitch
        ray_x_final = ray_x_rot

        # 转换为球坐标（全景图的坐标）
        # 经度 (longitude): atan2(x, z)
        # 维度 (latitude): asin(y)
        lon = np.arctan2(ray_x_final, ray_z_final)
        lat = np.arcsin(np.clip(ray_y_rot, -1.0, 1.0))

        # 将球坐标映射到全景图的像素坐标
        # 等距柱状投影：x = (lon + π) / (2π) * W, y = (π/2 - lat) / π * H
        map_x = ((lon + math.pi) / (2 * math.pi) * pano_w) % pano_w
        map_y = (math.pi / 2 - lat) / math.pi * pano_h

        # 限制在有效范围内
        map_y = np.clip(map_y, 0, pano_h - 1)

        # 使用 remap 进行重采样
        viewport = cv2.remap(
            pano_image,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP  # 水平方向循环
        )

        return viewport

    def extract_stereo_viewports_from_pano(
        self,
        pano_left: np.ndarray,
        pano_right: np.ndarray,
        hfov_deg: float = 90.0,
        vfov_deg: float = 60.0,
        yaw_deg: float = 180.0,
        pitch_deg: float = 0.0,
        output_width: int = 640,
        output_height: int = 480
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从左右全景图中提取对应的立体视口对

        参数:
            pano_left: 左相机的全景图
            pano_right: 右相机的全景图
            hfov_deg: 视口水平视场角
            vfov_deg: 视口垂直视场角
            yaw_deg: 观看方向（度）
            pitch_deg: 俯仰角（度）
            output_width: 输出宽度
            output_height: 输出高度

        返回:
            (viewport_left, viewport_right): 左右视口图像
        """
        viewport_left = self.extract_equirectangular_viewport(
            pano_left, hfov_deg, vfov_deg, yaw_deg, pitch_deg,
            output_width, output_height
        )

        viewport_right = self.extract_equirectangular_viewport(
            pano_right, hfov_deg, vfov_deg, yaw_deg, pitch_deg,
            output_width, output_height
        )

        return viewport_left, viewport_right

    def detect_optimal_viewport(
        self,
        pano_image: np.ndarray,
        foreground_center_ratio: Tuple[float, float] = (0.5, 0.5),
        hfov_deg: float = 90.0,
        vfov_deg: float = 60.0
    ) -> dict:
        """
        根据前景位置自动检测最佳视口参数

        参数:
            pano_image: 全景图像
            foreground_center_ratio: 前景中心位置在全景图中的归一化坐标 (x_ratio, y_ratio)
            hfov_deg: 期望的水平视场角
            vfov_deg: 期望的垂直视场角

        返回:
            viewport_params: 包含 yaw_deg, pitch_deg 的字典
        """
        pano_h, pano_w = pano_image.shape[:2]

        # 从全景图的归一化坐标计算yaw和pitch
        x_ratio, y_ratio = foreground_center_ratio

        # yaw: 0度=正前方(图像中心), 向右为正
        yaw_deg = (x_ratio - 0.5) * 360.0

        # pitch: 0度=水平, 向上为正
        pitch_deg = (0.5 - y_ratio) * 180.0

        return {
            'yaw_deg': yaw_deg,
            'pitch_deg': pitch_deg,
            'hfov_deg': hfov_deg,
            'vfov_deg': vfov_deg
        }


def test_panorama_extraction():
    """测试全景图视口提取"""
    print("=" * 80)
    print("测试全景图视口提取")
    print("=" * 80)

    # 创建处理器
    processor = PanoramaProcessor()

    # 测试：从全景图中提取视口
    pano_path = "test_data/images/2_00402.jpg"  # 假设这是全景图

    if not os.path.exists(pano_path):
        print(f"测试图像不存在: {pano_path}")
        return

    pano = cv2.imread(pano_path)
    print(f"全景图尺寸: {pano.shape}")

    # 提取不同方向的视口
    test_configs = [
        {"yaw": 0, "pitch": 0, "name": "front"},
        {"yaw": 90, "pitch": 0, "name": "right"},
        {"yaw": 180, "pitch": 0, "name": "back"},
        {"yaw": 270, "pitch": 0, "name": "left"},
    ]

    os.makedirs("output/pano_test", exist_ok=True)

    for config in test_configs:
        viewport = processor.extract_equirectangular_viewport(
            pano,
            hfov_deg=90.0,
            vfov_deg=60.0,
            yaw_deg=config["yaw"],
            pitch_deg=config["pitch"],
            output_width=640,
            output_height=480
        )

        output_path = f"output/pano_test/viewport_{config['name']}.jpg"
        cv2.imwrite(output_path, viewport)
        print(f"✓ 提取视口 [{config['name']}]: {output_path}")

    print("\n" + "=" * 80)
    print("测试完成！请查看 output/pano_test/ 目录")
    print("=" * 80)


if __name__ == "__main__":
    test_panorama_extraction()

