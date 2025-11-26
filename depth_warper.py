"""
基于深度的背景重投影
适用于有较强三维结构的背景
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict


class DepthWarper:
    """深度引导的背景扭曲器"""

    def __init__(self):
        """初始化深度扭曲器"""
        self.K_source = None  # 源相机内参
        self.K_left = None    # 左目标相机内参
        self.K_right = None   # 右目标相机内参
        self.R_left = None    # 左相机旋转
        self.t_left = None    # 左相机平移
        self.R_right = None   # 右相机旋转
        self.t_right = None   # 右相机平移

    def set_source_camera(self, K: np.ndarray):
        """
        设置源背景图的相机内参

        Args:
            K: 3x3 内参矩阵
        """
        self.K_source = K.copy()
        print(f"Source camera intrinsics set: f={K[0,0]:.1f}px")

    def set_target_cameras(
        self,
        K_left: np.ndarray,
        K_right: np.ndarray,
        R_left: np.ndarray,
        t_left: np.ndarray,
        R_right: np.ndarray,
        t_right: np.ndarray
    ):
        """
        设置左右目标相机的内外参

        Args:
            K_left: 左相机 3x3 内参
            K_right: 右相机 3x3 内参
            R_left: 左相机 3x3 旋转矩阵
            t_left: 左相机 3x1 平移向量
            R_right: 右相机 3x3 旋转矩阵
            t_right: 右相机 3x1 平移向量
        """
        self.K_left = K_left.copy()
        self.K_right = K_right.copy()
        self.R_left = R_left.copy()
        self.t_left = t_left.reshape(3, 1)
        self.R_right = R_right.copy()
        self.t_right = t_right.reshape(3, 1)
        print("Target cameras intrinsics and extrinsics set")

    def set_target_cameras_symmetric(
        self,
        K: np.ndarray,
        baseline: float,
        rotation_y_deg: float = 0.0
    ):
        """
        设置对称的左右相机（简化版）

        Args:
            K: 共享的 3x3 内参
            baseline: 基线距离（世界坐标单位）
            rotation_y_deg: 左右相机绕Y轴的对称旋转角度（度）
        """
        self.K_left = K.copy()
        self.K_right = K.copy()

        # 对称配置：左相机向左平移 baseline/2，右相机向右平移 baseline/2
        angle_rad = np.radians(rotation_y_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # 左相机：向左平移，可选向右旋转
        self.R_left = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ], dtype=np.float64)
        self.t_left = np.array([[-baseline / 2], [0], [0]], dtype=np.float64)

        # 右相机：向右平移，可选向左旋转
        self.R_right = np.array([
            [cos_a, 0, -sin_a],
            [0, 1, 0],
            [sin_a, 0, cos_a]
        ], dtype=np.float64)
        self.t_right = np.array([[baseline / 2], [0], [0]], dtype=np.float64)

        print(f"Symmetric cameras set: baseline={baseline}, rotation_y={rotation_y_deg}°")

    def unproject_depth(
        self,
        depth_map: np.ndarray,
        K: np.ndarray
    ) -> np.ndarray:
        """
        将深度图反投影到3D点云

        Args:
            depth_map: HxW 深度图
            K: 3x3 内参矩阵

        Returns:
            points_3d: (H*W)x3 点云，列向量 [X, Y, Z]
        """
        H, W = depth_map.shape[:2]

        # 生成像素坐标网格
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.astype(np.float32)
        v = v.astype(np.float32)

        # 归一化坐标
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy

        # 3D坐标
        Z = depth_map.flatten()
        X = x_norm.flatten() * Z
        Y = y_norm.flatten() * Z

        points_3d = np.stack([X, Y, Z], axis=1)  # (H*W, 3)

        return points_3d

    def project_points(
        self,
        points_3d: np.ndarray,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        image_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将3D点投影到目标相机

        Args:
            points_3d: Nx3 点云
            K: 3x3 内参
            R: 3x3 旋转
            t: 3x1 平移
            image_size: (width, height)

        Returns:
            uv: Nx2 像素坐标
            depth: N 深度值
            valid_mask: N bool掩码（是否在图像内且深度>0）
        """
        # 变换到目标相机坐标系
        points_cam = (R @ points_3d.T + t).T  # (N, 3)

        # 投影
        Z = points_cam[:, 2]
        valid_depth = Z > 0

        uv_homo = (K @ points_cam.T).T  # (N, 3)
        u = uv_homo[:, 0] / (uv_homo[:, 2] + 1e-8)
        v = uv_homo[:, 1] / (uv_homo[:, 2] + 1e-8)

        # 检查是否在图像内
        W, H = image_size
        valid_u = (u >= 0) & (u < W)
        valid_v = (v >= 0) & (v < H)
        valid_mask = valid_depth & valid_u & valid_v

        uv = np.stack([u, v], axis=1)

        return uv, Z, valid_mask

    def warp_background_with_depth(
        self,
        background_img: np.ndarray,
        depth_map: np.ndarray,
        target_size: Tuple[int, int],
        camera: str = "left",
        fill_holes: bool = True,
        bilateral_filter: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用深度图对背景进行重投影

        Args:
            background_img: 源背景图 HxWx3
            depth_map: 源深度图 HxW（与背景图对应）
            target_size: 目标尺寸 (width, height)
            camera: "left" 或 "right"
            fill_holes: 是否填补空洞
            bilateral_filter: 是否对深度图进行双边滤波

        Returns:
            warped_img: 变换后的图像
            valid_mask: 有效像素掩码
        """
        if self.K_source is None:
            raise RuntimeError("Source camera intrinsics not set")

        # 选择目标相机参数
        if camera == "left":
            K_target = self.K_left
            R_target = self.R_left
            t_target = self.t_left
        elif camera == "right":
            K_target = self.K_right
            R_target = self.R_right
            t_target = self.t_right
        else:
            raise ValueError(f"Unknown camera: {camera}")

        if K_target is None or R_target is None or t_target is None:
            raise RuntimeError(f"Target {camera} camera not set")

        # 深度图预处理
        depth_processed = depth_map.copy()
        if bilateral_filter:
            # 双边滤波平滑深度，保留边缘
            depth_processed = cv2.bilateralFilter(
                depth_processed.astype(np.float32), 9, 75, 75
            )

        # 反投影到3D
        points_3d = self.unproject_depth(depth_processed, self.K_source)

        # 投影到目标相机
        uv_target, depth_target, valid_mask = self.project_points(
            points_3d, K_target, R_target, t_target, target_size
        )

        # 前向投影 + z-buffer
        H_src, W_src = background_img.shape[:2]
        W_tgt, H_tgt = target_size

        warped_img = np.zeros((H_tgt, W_tgt, 3), dtype=background_img.dtype)
        z_buffer = np.full((H_tgt, W_tgt), np.inf, dtype=np.float32)
        pixel_mask = np.zeros((H_tgt, W_tgt), dtype=bool)

        # 源像素坐标
        v_src, u_src = np.divmod(np.arange(H_src * W_src), W_src)

        valid_indices = np.where(valid_mask)[0]

        for idx in valid_indices:
            u_t = int(round(uv_target[idx, 0]))
            v_t = int(round(uv_target[idx, 1]))
            z = depth_target[idx]

            # 边界检查（双重保险）
            if 0 <= u_t < W_tgt and 0 <= v_t < H_tgt:
                # z-buffer测试
                if z < z_buffer[v_t, u_t]:
                    z_buffer[v_t, u_t] = z
                    warped_img[v_t, u_t] = background_img[v_src[idx], u_src[idx]]
                    pixel_mask[v_t, u_t] = True

        # 填补空洞
        if fill_holes:
            hole_mask = ~pixel_mask
            if np.any(hole_mask):
                # 使用inpaint填补
                warped_img = cv2.inpaint(
                    warped_img,
                    hole_mask.astype(np.uint8) * 255,
                    inpaintRadius=3,
                    flags=cv2.INPAINT_TELEA
                )
                pixel_mask = np.ones((H_tgt, W_tgt), dtype=bool)

        coverage = np.sum(pixel_mask) / (H_tgt * W_tgt) * 100
        print(f"Warped {camera} camera: {coverage:.1f}% coverage")

        return warped_img, pixel_mask


if __name__ == "__main__":
    print("DepthWarper class loaded. Use in main pipeline.")

