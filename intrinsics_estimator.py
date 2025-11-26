"""
相机内参估计工具
支持从EXIF或HFOV估计相机内参矩阵
"""
from math import tan, radians
from typing import Optional, Tuple, Dict
import numpy as np

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    EXIF_AVAILABLE = True
except ImportError:
    EXIF_AVAILABLE = False
    print("Warning: PIL not available, EXIF reading disabled")


class IntrinsicsEstimator:
    """相机内参估计器"""

    def __init__(self, hfov_deg_fallback: float = 70.0, principal_at_center: bool = True):
        """
        Args:
            hfov_deg_fallback: 默认水平视场角（度）
            principal_at_center: 主点是否位于图像中心
        """
        self.hfov_deg_fallback = hfov_deg_fallback
        self.principal_at_center = principal_at_center

    def _read_exif(self, image_path: str) -> Dict:
        """读取图像EXIF信息"""
        if not EXIF_AVAILABLE:
            return {}

        try:
            img = Image.open(image_path)
            exif_data = img._getexif()
            if exif_data is None:
                return {}

            exif_dict = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                exif_dict[tag_name] = value
            return exif_dict
        except Exception as e:
            print(f"Warning: Failed to read EXIF from {image_path}: {e}")
            return {}

    def _to_float(self, value):
        """将EXIF值转换为浮点数"""
        try:
            # 处理PIL的IFDRational类型
            if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
                return float(value.numerator) / float(value.denominator)
            # 处理元组类型 (numerator, denominator)
            if isinstance(value, tuple) and len(value) == 2:
                return float(value[0]) / float(value[1])
            return float(value)
        except:
            return None

    def focal_px_from_hfov(self, width_px: int, hfov_deg: float) -> float:
        """
        从水平视场角计算焦距（像素单位）

        Args:
            width_px: 图像宽度（像素）
            hfov_deg: 水平视场角（度）

        Returns:
            焦距（像素）
        """
        hfov_rad = radians(hfov_deg)
        return 0.5 * width_px / tan(0.5 * hfov_rad)

    def focal_px_from_35mm(self, width_px: int, f35_mm: float) -> float:
        """
        从35mm等效焦距计算焦距（像素单位）

        Args:
            width_px: 图像宽度（像素）
            f35_mm: 35mm等效焦距（毫米）

        Returns:
            焦距（像素）
        """
        # 35mm胶片宽度为36mm
        return (f35_mm / 36.0) * float(width_px)

    def estimate(
        self,
        image_shape: Tuple[int, int],
        exif_image_path: Optional[str] = None,
        hfov_deg_override: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        估计相机内参

        Args:
            image_shape: 图像尺寸 (H, W)
            exif_image_path: 图像路径（用于读取EXIF）
            hfov_deg_override: 手动指定HFOV（覆盖默认值）

        Returns:
            K: 3x3 内参矩阵
            dist: 1x5 畸变系数（全零）
            meta: 元信息字典 {source, f_px, cx, cy, W, H}
        """
        H, W = int(image_shape[0]), int(image_shape[1])

        # 计算主点位置
        if self.principal_at_center:
            cx = (W - 1) * 0.5
            cy = (H - 1) * 0.5
        else:
            cx = W * 0.5
            cy = H * 0.5

        # 默认使用HFOV估计
        hfov = hfov_deg_override if hfov_deg_override is not None else self.hfov_deg_fallback
        f_px = self.focal_px_from_hfov(W, hfov)
        source = "fallback_hfov"

        # 尝试从EXIF读取
        if exif_image_path:
            exif = self._read_exif(exif_image_path)

            # 优先使用35mm等效焦距
            f35 = self._to_float(exif.get("FocalLengthIn35mmFilm"))
            if f35 and f35 > 0:
                f_px = self.focal_px_from_35mm(W, f35)
                source = "exif_35mm_equiv"
                print(f"Using EXIF 35mm equivalent focal length: {f35:.1f}mm -> {f_px:.1f}px")
            else:
                # 尝试使用FocalLength（但需要传感器宽度，通常不可靠）
                focal_mm = self._to_float(exif.get("FocalLength"))
                if focal_mm:
                    print(f"Found FocalLength {focal_mm:.1f}mm but no sensor size, using HFOV fallback")

        # 构建内参矩阵
        K = np.array([
            [f_px, 0.0, cx],
            [0.0, f_px, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # 零畸变
        dist = np.zeros((1, 5), dtype=np.float64)

        # 元信息
        meta = {
            "source": source,
            "f_px": float(f_px),
            "cx": float(cx),
            "cy": float(cy),
            "W": W,
            "H": H,
            "hfov_deg": hfov
        }

        print(f"Estimated intrinsics from {source}: f={f_px:.1f}px, principal=({cx:.1f}, {cy:.1f})")

        return K, dist, meta


if __name__ == "__main__":
    # 测试示例
    estimator = IntrinsicsEstimator(hfov_deg_fallback=70.0)

    # 示例1: 仅使用分辨率和默认HFOV
    K, dist, meta = estimator.estimate((1080, 1920))
    print("\n=== Test 1: Default HFOV ===")
    print("K=\n", K)
    print("meta=", meta)

    # 示例2: 手动指定HFOV
    K2, dist2, meta2 = estimator.estimate((1080, 1920), hfov_deg_override=75.0)
    print("\n=== Test 2: Manual HFOV=75 ===")
    print("K=\n", K2)
    print("meta=", meta2)

