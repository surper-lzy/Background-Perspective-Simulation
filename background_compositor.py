"""
背景合成主类
整合单应变换和深度重投影，完成前景背景合成
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
from enum import Enum

from intrinsics_estimator import IntrinsicsEstimator
from homography_warper import HomographyWarper
from depth_warper import DepthWarper


class WarpMethod(Enum):
    """背景扭曲方法"""
    HOMOGRAPHY = "homography"
    DEPTH = "depth"


class BackgroundCompositor:
    """背景合成器主类"""

    def __init__(self, method: WarpMethod = WarpMethod.HOMOGRAPHY):
        """
        Args:
            method: 背景扭曲方法
        """
        self.method = method
        self.intrinsics_estimator = IntrinsicsEstimator()

        if method == WarpMethod.HOMOGRAPHY:
            self.warper = HomographyWarper()
        elif method == WarpMethod.DEPTH:
            self.warper = DepthWarper()
        else:
            raise ValueError(f"Unknown method: {method}")

        self.background_img = None
        self.foreground_mask_left = None
        self.foreground_mask_right = None

    def load_background(self, image_path: str):
        """加载背景图"""
        self.background_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.background_img is None:
            raise RuntimeError(f"Failed to load background image: {image_path}")
        print(f"Background loaded: {self.background_img.shape}")

    def load_foreground_masks(
        self,
        mask_left_path: Optional[str] = None,
        mask_right_path: Optional[str] = None,
        mask_left: Optional[np.ndarray] = None,
        mask_right: Optional[np.ndarray] = None
    ):
        """
        加载前景掩码

        Args:
            mask_left_path: 左相机前景掩码路径
            mask_right_path: 右相机前景掩码路径
            mask_left: 左相机前景掩码数组
            mask_right: 右相机前景掩码数组
        """
        if mask_left_path:
            self.foreground_mask_left = cv2.imread(mask_left_path, cv2.IMREAD_GRAYSCALE)
            if self.foreground_mask_left is None:
                raise RuntimeError(f"Failed to load left mask: {mask_left_path}")
        elif mask_left is not None:
            self.foreground_mask_left = mask_left

        if mask_right_path:
            self.foreground_mask_right = cv2.imread(mask_right_path, cv2.IMREAD_GRAYSCALE)
            if self.foreground_mask_right is None:
                raise RuntimeError(f"Failed to load right mask: {mask_right_path}")
        elif mask_right is not None:
            self.foreground_mask_right = mask_right

        print("Foreground masks loaded")

    def setup_homography_method(
        self,
        left_reference_img: Optional[np.ndarray] = None,
        right_reference_img: Optional[np.ndarray] = None,
        left_points: Optional[List[Tuple[float, float]]] = None,
        right_points: Optional[List[Tuple[float, float]]] = None,
        background_points_left: Optional[List[Tuple[float, float]]] = None,
        background_points_right: Optional[List[Tuple[float, float]]] = None,
        auto_method: Optional[str] = None
    ):
        """
        设置单应变换方法的参数

        Args:
            left_reference_img: 左相机参考图（用于特征匹配）
            right_reference_img: 右相机参考图（用于特征匹配）
            left_points: 左相机目标点
            right_points: 右相机目标点
            background_points_left: 背景图对应左相机的点
            background_points_right: 背景图对应右相机的点
            auto_method: 自动特征匹配方法 ("sift", "orb")
        """
        if self.method != WarpMethod.HOMOGRAPHY:
            raise RuntimeError("Current method is not HOMOGRAPHY")

        if self.background_img is None:
            raise RuntimeError("Background image not loaded")

        self.warper.setup_stereo_homographies(
            self.background_img,
            left_reference_img=left_reference_img,
            right_reference_img=right_reference_img,
            left_points=left_points,
            right_points=right_points,
            background_points_left=background_points_left,
            background_points_right=background_points_right,
            auto_method=auto_method
        )

        print("Homography method setup complete")

    def setup_two_stage_homography(
        self,
        left_reference_img: np.ndarray,
        right_reference_img: np.ndarray,
        plane_corners_in_left: List[Tuple[float, float]],
        auto_method: str = "sift"
    ):
        """
        设置两阶段单应变换方法
        【阶段1】从左右参考对学习相机几何关系
        【阶段2】将新背景作为平面映射到左右视角

        Args:
            left_reference_img: 左相机参考图
            right_reference_img: 右相机参考图
            plane_corners_in_left: 背景平面在左视角中的四个角点（顺时针或逆时针）
            auto_method: 特征匹配方法 ("sift", "orb")
        """
        if self.method != WarpMethod.HOMOGRAPHY:
            raise RuntimeError("Current method is not HOMOGRAPHY")

        if self.background_img is None:
            raise RuntimeError("Background image not loaded")

        self.warper.setup_two_stage_homographies(
            left_reference_img,
            right_reference_img,
            self.background_img,
            plane_corners_in_left,
            auto_method
        )

        print("Two-stage homography method setup complete")

    def setup_depth_method(
        self,
        depth_map: np.ndarray,
        background_image_shape: Tuple[int, int],
        target_image_shape: Tuple[int, int],
        hfov_deg: float = 70.0,
        baseline: float = 0.065,
        rotation_y_deg: float = 5.0,
        background_exif_path: Optional[str] = None
    ):
        """
        设置深度重投影方法的参数

        Args:
            depth_map: 背景图的深度图
            background_image_shape: 背景图尺寸 (H, W)
            target_image_shape: 目标图尺寸 (H, W)
            hfov_deg: 水平视场角
            baseline: 左右相机基线距离
            rotation_y_deg: 左右相机对称旋转角度
            background_exif_path: 背景图EXIF路径（用于估计内参）
        """
        if self.method != WarpMethod.DEPTH:
            raise RuntimeError("Current method is not DEPTH")

        # 估计源相机内参
        K_source, _, meta_src = self.intrinsics_estimator.estimate(
            background_image_shape,
            exif_image_path=background_exif_path,
            hfov_deg_override=hfov_deg
        )
        self.warper.set_source_camera(K_source)

        # 估计目标相机内参（假设与源相机相同）
        K_target, _, meta_tgt = self.intrinsics_estimator.estimate(
            target_image_shape,
            hfov_deg_override=hfov_deg
        )

        # 设置对称相机
        self.warper.set_target_cameras_symmetric(
            K_target, baseline=baseline, rotation_y_deg=rotation_y_deg
        )

        self.depth_map = depth_map
        print("Depth method setup complete")

    def generate_warped_backgrounds(
        self,
        target_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成左右相机的扭曲背景

        Args:
            target_size: 目标尺寸 (width, height)

        Returns:
            warped_left: 左相机扭曲背景
            warped_right: 右相机扭曲背景
        """
        if self.background_img is None:
            raise RuntimeError("Background image not loaded")

        if self.method == WarpMethod.HOMOGRAPHY:
            warped_left = self.warper.warp_background(
                self.background_img, target_size, camera="left"
            )
            warped_right = self.warper.warp_background(
                self.background_img, target_size, camera="right"
            )

        elif self.method == WarpMethod.DEPTH:
            if not hasattr(self, 'depth_map') or self.depth_map is None:
                raise RuntimeError("Depth map not set")

            warped_left, mask_left = self.warper.warp_background_with_depth(
                self.background_img, self.depth_map, target_size, camera="left"
            )
            warped_right, mask_right = self.warper.warp_background_with_depth(
                self.background_img, self.depth_map, target_size, camera="right"
            )
        else:
            raise RuntimeError(f"Unknown method: {self.method}")

        print("Warped backgrounds generated")
        return warped_left, warped_right

    def composite_with_foreground(
        self,
        warped_background: np.ndarray,
        foreground_image: np.ndarray,
        foreground_mask: Optional[np.ndarray] = None,
        feather_radius: int = 5
    ) -> np.ndarray:
        """
        将扭曲背景与前景合成

        Args:
            warped_background: 扭曲后的背景
            foreground_image: 前景图像
            foreground_mask: 前景掩码（0-255，255为前景）
            feather_radius: 边缘羽化半径

        Returns:
            合成后的图像
        """
        if foreground_mask is None:
            # 如果没有掩码，直接返回背景
            print("No foreground mask provided, returning warped background")
            return warped_background

        # 确保掩码尺寸匹配
        if foreground_mask.shape[:2] != warped_background.shape[:2]:
            foreground_mask = cv2.resize(
                foreground_mask,
                (warped_background.shape[1], warped_background.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        # 归一化掩码到 [0, 1]
        mask_norm = foreground_mask.astype(np.float32) / 255.0

        # 边缘羽化
        if feather_radius > 0:
            mask_norm = cv2.GaussianBlur(mask_norm, (feather_radius*2+1, feather_radius*2+1), 0)

        # 扩展掩码维度以匹配RGB
        mask_3ch = np.stack([mask_norm] * 3, axis=2)

        # Alpha混合
        composite = (foreground_image.astype(np.float32) * mask_3ch +
                    warped_background.astype(np.float32) * (1 - mask_3ch))
        composite = np.clip(composite, 0, 255).astype(np.uint8)

        return composite

    # background_compositor.py

    def process_stereo_pair(
            self,
            foreground_left: np.ndarray,
            foreground_right: np.ndarray,
            target_size: tuple[int, int],
            feather_radius: int = 5,
            color_matching: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        处理完整的立体图像对，将前景合成到扭曲后的背景上。

        Args:
            foreground_left: 左前景图。
            foreground_right: 右前景图。
            target_size: 最终输出图像尺寸 (W, H)。
            feather_radius: 掩码羽化半径，用于平滑边缘。
            color_matching: 是否尝试进行颜色匹配。

        Returns:
            (result_left, result_right): 合成后的左右视图。
        """
        if self.mask_left is None or self.mask_right is None:
            raise RuntimeError("Foreground masks have not been loaded.")

        # 步骤1: 生成扭曲的背景
        warped_left, warped_right = self.generate_warped_backgrounds(target_size)

        # 步骤2: 羽化掩码以获得平滑边缘
        # 确保羽化半径是奇数
        if feather_radius > 0:
            k_size = feather_radius * 2 + 1
            # 将掩码转换为浮点数以便进行精确计算
            mask_left_float = cv2.GaussianBlur(self.mask_left, (k_size, k_size), 0).astype(np.float32) / 255.0
            mask_right_float = cv2.GaussianBlur(self.mask_right, (k_size, k_size), 0).astype(np.float32) / 255.0
        else:
            mask_left_float = self.mask_left.astype(np.float32) / 255.0
            mask_right_float = self.mask_right.astype(np.float32) / 255.0

        # 将单通道掩码扩展为三通道，以便与彩色图像相乘
        mask_left_3ch = cv2.cvtColor(mask_left_float, cv2.COLOR_GRAY2BGR)
        mask_right_3ch = cv2.cvtColor(mask_right_float, cv2.COLOR_GRAY2BGR)

        # 步骤3: 执行 Alpha Blending
        # 将图像转换为浮点数以进行精确计算
        fg_left_float = foreground_left.astype(np.float32)
        fg_right_float = foreground_right.astype(np.float32)
        bg_left_float = warped_left.astype(np.float32)
        bg_right_float = warped_right.astype(np.float32)

        # 应用 Alpha Blending 公式: Result = Fg * alpha + Bg * (1 - alpha)
        result_left_float = fg_left_float * mask_left_3ch + bg_left_float * (1.0 - mask_left_3ch)
        result_right_float = fg_right_float * mask_right_3ch + bg_right_float * (1.0 - mask_right_3ch)

        # 步骤4: 将结果转换回 uint8
        result_left = np.clip(result_left_float, 0, 255).astype(np.uint8)
        result_right = np.clip(result_right_float, 0, 255).astype(np.uint8)

        # (可选) 颜色匹配逻辑可以放在这里
        if color_matching:
            # ... 颜色匹配实现 ...
            pass

        return result_left, result_right

    def _match_color(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        简单的颜色匹配（直方图均衡化）

        Args:
            source: 源图像
            reference: 参考图像

        Returns:
            颜色匹配后的源图像
        """
        # 转换到LAB空间
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)

        # 对每个通道进行匹配
        matched_lab = np.zeros_like(source_lab)
        for i in range(3):
            # 计算累积直方图
            src_hist, src_bins = np.histogram(source_lab[:,:,i].flatten(), 256, [0, 256])
            ref_hist, ref_bins = np.histogram(reference_lab[:,:,i].flatten(), 256, [0, 256])

            src_cdf = src_hist.cumsum()
            src_cdf = src_cdf / src_cdf[-1]

            ref_cdf = ref_hist.cumsum()
            ref_cdf = ref_cdf / ref_cdf[-1]

            # 映射查找表
            lut = np.interp(src_cdf, ref_cdf, np.arange(256))
            matched_lab[:,:,i] = lut[source_lab[:,:,i]]

        # 转回BGR
        matched = cv2.cvtColor(matched_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return matched


if __name__ == "__main__":
    print("BackgroundCompositor class loaded. Use in main pipeline.")
