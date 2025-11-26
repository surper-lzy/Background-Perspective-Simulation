"""
基于单应变换的背景扭曲
适用于平面或远景背景
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


class HomographyWarper:
    """单应变换背景扭曲器"""

    def __init__(self, ransac_reproj_threshold: float = 5.0):
        """
        Args:
            ransac_reproj_threshold: RANSAC重投影误差阈值（像素）
        """
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.H_left = None
        self.H_right = None
        self.H_ref_left_to_right = None  # 参考对的相对单应
        self.plane_to_left = None  # 从标准平面到左视角的单应
        self.plane_to_right = None  # 从标准平面到右视角的单应
        self.meta = {}

    def compute_homography_manual(
        self,
        pts_source: List[Tuple[float, float]],
        pts_target: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, Dict]:
        """
        从手动标注的对应点计算单应矩阵

        Args:
            pts_source: 源图像中的点 [(x, y), ...]
            pts_target: 目标图像中的对应点 [(x, y), ...]

        Returns:
            H: 3x3 单应矩阵
            meta: 元信息 {method, inliers, reproj_error}
        """
        pts_src = np.array(pts_source, dtype=np.float32)
        pts_dst = np.array(pts_target, dtype=np.float32)

        if len(pts_src) < 4:
            raise ValueError(f"Need at least 4 points, got {len(pts_src)}")

        # 使用RANSAC计算单应矩阵
        H, mask = cv2.findHomography(
            pts_src, pts_dst,
            cv2.RANSAC,
            self.ransac_reproj_threshold
        )

        if H is None:
            raise RuntimeError("Failed to compute homography")

        inliers = int(np.sum(mask)) if mask is not None else len(pts_src)

        # 计算重投影误差
        pts_src_homo = np.concatenate([pts_src, np.ones((len(pts_src), 1))], axis=1)
        pts_proj = (H @ pts_src_homo.T).T
        pts_proj = pts_proj[:, :2] / pts_proj[:, 2:3]
        reproj_error = np.mean(np.linalg.norm(pts_proj - pts_dst, axis=1))

        meta = {
            "method": "manual_points_ransac",
            "total_points": len(pts_src),
            "inliers": inliers,
            "reproj_error": float(reproj_error)
        }

        print(f"Homography computed: {inliers}/{len(pts_src)} inliers, reproj_error={reproj_error:.2f}px")

        return H, meta

    def compute_homography_features(
        self,
        img_source: np.ndarray,
        img_target: np.ndarray,
        method: str = "sift",
        max_features: int = 5000
    ) -> Tuple[np.ndarray, Dict]:
        """
        从特征匹配自动计算单应矩阵

        Args:
            img_source: 源图像
            img_target: 目标图像
            method: 特征提取方法 ("sift", "orb")
            max_features: 最大特征点数量

        Returns:
            H: 3x3 单应矩阵
            meta: 元信息
        """
        # 转为灰度图
        if len(img_source.shape) == 3:
            gray_src = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
        else:
            gray_src = img_source

        if len(img_target.shape) == 3:
            gray_dst = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
        else:
            gray_dst = img_target

        # 特征检测与描述
        if method.lower() == "sift":
            detector = cv2.SIFT_create(nfeatures=max_features)
        elif method.lower() == "orb":
            detector = cv2.ORB_create(nfeatures=max_features)
        else:
            raise ValueError(f"Unknown method: {method}")

        kp1, desc1 = detector.detectAndCompute(gray_src, None)
        kp2, desc2 = detector.detectAndCompute(gray_dst, None)

        if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
            raise RuntimeError(f"Insufficient features detected: {len(kp1)} vs {len(kp2)}")

        # 特征匹配
        if method.lower() == "sift":
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = matcher.knnMatch(desc1, desc2, k=2)
            # Lowe's ratio test
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
        else:  # ORB
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            good_matches = matcher.match(desc1, desc2)
            good_matches = sorted(good_matches, key=lambda x: x.distance)[:500]

        if len(good_matches) < 4:
            raise RuntimeError(f"Insufficient good matches: {len(good_matches)}")

        # 提取匹配点
        pts_src = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts_dst = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # 计算单应矩阵
        H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, self.ransac_reproj_threshold)

        if H is None:
            raise RuntimeError("Failed to compute homography from features")

        inliers = int(np.sum(mask))

        # 计算重投影误差
        pts_src_homo = np.concatenate([pts_src, np.ones((len(pts_src), 1))], axis=1)
        pts_proj = (H @ pts_src_homo.T).T
        pts_proj = pts_proj[:, :2] / pts_proj[:, 2:3]
        reproj_error = np.mean(np.linalg.norm(pts_proj - pts_dst, axis=1))

        meta = {
            "method": f"auto_features_{method}",
            "features_detected": (len(kp1), len(kp2)),
            "matches": len(good_matches),
            "inliers": inliers,
            "reproj_error": float(reproj_error)
        }

        print(f"Feature-based homography: {len(kp1)}/{len(kp2)} features, "
              f"{len(good_matches)} matches, {inliers} inliers, reproj_error={reproj_error:.2f}px")

        return H, meta

    def compute_geometry_from_reference_pair(
        self,
        ref_left_img: np.ndarray,
        ref_right_img: np.ndarray,
        method: str = "sift",
        max_features: int = 5000
    ) -> Tuple[np.ndarray, Dict]:
        """
        【阶段1】从左右参考对学习相机几何关系

        Args:
            ref_left_img: 左参考图
            ref_right_img: 右参考图
            method: 特征提取方法
            max_features: 最大特征点数量

        Returns:
            H_left_to_right: 左→右的单应矩阵
            meta: 元信息
        """
        print("\n[阶段1] 从参考对学习相机几何关系...")

        # 转为灰度图
        if len(ref_left_img.shape) == 3:
            gray_left = cv2.cvtColor(ref_left_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = ref_left_img

        if len(ref_right_img.shape) == 3:
            gray_right = cv2.cvtColor(ref_right_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_right = ref_right_img

        # 特征检测
        if method.lower() == "sift":
            detector = cv2.SIFT_create(nfeatures=max_features)
        elif method.lower() == "orb":
            detector = cv2.ORB_create(nfeatures=max_features)
        else:
            raise ValueError(f"Unknown method: {method}")

        kp_left, desc_left = detector.detectAndCompute(gray_left, None)
        kp_right, desc_right = detector.detectAndCompute(gray_right, None)

        if desc_left is None or desc_right is None or len(kp_left) < 4 or len(kp_right) < 4:
            raise RuntimeError(f"Insufficient features: {len(kp_left)} vs {len(kp_right)}")

        # 特征匹配
        if method.lower() == "sift":
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = matcher.knnMatch(desc_left, desc_right, k=2)
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            good_matches = matcher.match(desc_left, desc_right)
            good_matches = sorted(good_matches, key=lambda x: x.distance)[:500]

        if len(good_matches) < 4:
            raise RuntimeError(f"Insufficient matches: {len(good_matches)}")

        # 提取匹配点
        pts_left = np.float32([kp_left[m.queryIdx].pt for m in good_matches])
        pts_right = np.float32([kp_right[m.trainIdx].pt for m in good_matches])

        # 计算左→右单应矩阵
        H, mask = cv2.findHomography(pts_left, pts_right, cv2.RANSAC, self.ransac_reproj_threshold)

        if H is None:
            raise RuntimeError("Failed to compute homography from reference pair")

        inliers = int(np.sum(mask))

        # 计算重投影误差
        pts_left_homo = np.concatenate([pts_left, np.ones((len(pts_left), 1))], axis=1)
        pts_proj = (H @ pts_left_homo.T).T
        pts_proj = pts_proj[:, :2] / pts_proj[:, 2:3]
        reproj_error = np.mean(np.linalg.norm(pts_proj - pts_right, axis=1))

        self.H_ref_left_to_right = H

        meta = {
            "method": f"reference_pair_{method}",
            "features_detected": (len(kp_left), len(kp_right)),
            "matches": len(good_matches),
            "inliers": inliers,
            "reproj_error": float(reproj_error)
        }

        print(f"   ✓ 参考对几何学习完成: {len(kp_left)}/{len(kp_right)} 特征点, "
              f"{len(good_matches)} 匹配, {inliers} 内点, 误差={reproj_error:.2f}px")

        return H, meta

    def calibrate_plane_homographies(
        self,
        background_img: np.ndarray,
        plane_corners: List[Tuple[float, float]],
        ref_img_size: Tuple[int, int]
    ):
        """
        【阶段1.5】标定平面：定义背景图作为标准平面，建立到左右视角的映射

        Args:
            background_img: 背景图（作为纹理平面）
            plane_corners: 平面在左视角中的四个角点（用户标注或自动检测）
            ref_img_size: 参考图尺寸 (width, height)
        """
        H_bg, W_bg = background_img.shape[:2]

        # 背景图的四角作为标准平面坐标
        bg_corners = np.float32([
            [0, 0],
            [W_bg - 1, 0],
            [W_bg - 1, H_bg - 1],
            [0, H_bg - 1]
        ])

        # 平面在左视角中的位置
        left_corners = np.float32(plane_corners)

        # 计算：标准平面 → 左视角
        self.plane_to_left = cv2.getPerspectiveTransform(bg_corners, left_corners)

        # 计算：标准平面 → 右视角
        # 利用 H_ref_left_to_right：right = H_ref * left
        # 所以 plane_to_right = H_ref * plane_to_left
        if self.H_ref_left_to_right is None:
            raise RuntimeError("Must call compute_geometry_from_reference_pair first")

        self.plane_to_right = self.H_ref_left_to_right @ self.plane_to_left

        print(f"   ✓ 平面标定完成: 背景({W_bg}x{H_bg}) → 左右视角")

    def setup_two_stage_homographies(
        self,
        ref_left_img: np.ndarray,
        ref_right_img: np.ndarray,
        background_img: np.ndarray,
        plane_corners_in_left: List[Tuple[float, float]],
        auto_method: str = "sift"
    ):
        """
        两阶段设置：阶段1从参考对学习几何，阶段2标定平面映射

        Args:
            ref_left_img: 左参考图
            ref_right_img: 右参考图
            background_img: 新背景图
            plane_corners_in_left: 背景平面在左视角中的四个角点
            auto_method: 特征匹配方法
        """
        # 阶段1：学习相机几何
        H_l2r, meta = self.compute_geometry_from_reference_pair(
            ref_left_img, ref_right_img, method=auto_method
        )
        self.meta["reference_geometry"] = meta

        # 阶段2：标定平面
        ref_size = (ref_left_img.shape[1], ref_left_img.shape[0])
        self.calibrate_plane_homographies(
            background_img, plane_corners_in_left, ref_size
        )

        # 保存最终单应矩阵
        self.H_left = self.plane_to_left
        self.H_right = self.plane_to_right

        self.meta["left"] = {"method": "two_stage_plane", "source": "plane_to_left"}
        self.meta["right"] = {"method": "two_stage_plane", "source": "plane_to_right"}

        print("\n✅ 两阶段单应设置完成")

    def setup_stereo_homographies(
        self,
        background_img: np.ndarray,
        left_reference_img: Optional[np.ndarray] = None,
        right_reference_img: Optional[np.ndarray] = None,
        left_points: Optional[List[Tuple[float, float]]] = None,
        right_points: Optional[List[Tuple[float, float]]] = None,
        background_points_left: Optional[List[Tuple[float, float]]] = None,
        background_points_right: Optional[List[Tuple[float, float]]] = None,
        auto_method: Optional[str] = None
    ):
        """
        为左右相机设置单应矩阵

        Args:
            background_img: 背景图
            left_reference_img: 左相机参考图（用于特征匹配）
            right_reference_img: 右相机参考图（用于特征匹配）
            left_points: 左相机目标图像中的点
            right_points: 右相机目标图像中的点
            background_points_left: 背景图中对应左相机的点
            background_points_right: 背景图中对应右相机的点
            auto_method: 自动特征匹配方法 ("sift", "orb")，若提供则忽略手动点
        """
        # 左相机
        if auto_method and left_reference_img is not None:
            self.H_left, meta_l = self.compute_homography_features(
                background_img, left_reference_img, method=auto_method
            )
            self.meta["left"] = meta_l
        elif background_points_left and left_points:
            self.H_left, meta_l = self.compute_homography_manual(
                background_points_left, left_points
            )
            self.meta["left"] = meta_l
        else:
            raise ValueError("Must provide either auto_method+reference_img or manual points for left camera")

        # 右相机
        if auto_method and right_reference_img is not None:
            self.H_right, meta_r = self.compute_homography_features(
                background_img, right_reference_img, method=auto_method
            )
            self.meta["right"] = meta_r
        elif background_points_right and right_points:
            self.H_right, meta_r = self.compute_homography_manual(
                background_points_right, right_points
            )
            self.meta["right"] = meta_r
        else:
            raise ValueError("Must provide either auto_method+reference_img or manual points for right camera")

    def warp_background(
        self,
        background_img: np.ndarray,
        target_size: Tuple[int, int],
        camera: str = "left",
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REPLICATE
    ) -> np.ndarray:
        """
        对背景图应用单应变换

        Args:
            background_img: 背景图
            target_size: 目标尺寸 (width, height)
            camera: "left" 或 "right"
            interpolation: 插值方法
            border_mode: 边界处理模式

        Returns:
            变换后的背景图
        """
        if camera == "left":
            H = self.H_left
        elif camera == "right":
            H = self.H_right
        else:
            raise ValueError(f"Unknown camera: {camera}")

        if H is None:
            raise RuntimeError(f"Homography for {camera} camera not computed yet")

        warped = cv2.warpPerspective(
            background_img, H, target_size,
            flags=interpolation,
            borderMode=border_mode
        )

        return warped


if __name__ == "__main__":
    # 测试示例
    print("HomographyWarper class loaded. Use in main pipeline.")
