"""
使用示例：展示如何使用背景合成系统
"""
import cv2
import numpy as np
from background_compositor import BackgroundCompositor, WarpMethod


def example_homography_manual_points():
    """示例1: 使用手动标注点的单应变换"""
    print("\n=== Example 1: Homography with Manual Points ===\n")

    # 创建合成器
    compositor = BackgroundCompositor(method=WarpMethod.HOMOGRAPHY)

    # 加载背景图
    compositor.load_background("path/to/background.jpg")

    # 定义对应点（示例坐标，需根据实际情况调整）
    # 背景图中的四个角点
    background_points_left = [
        (100, 100),   # 左上
        (900, 100),   # 右上
        (900, 700),   # 右下
        (100, 700)    # 左下
    ]

    # 左相机目标图像中的对应点
    left_points = [
        (50, 80),
        (950, 120),
        (920, 680),
        (80, 720)
    ]

    # 右相机对应点（视差偏移）
    background_points_right = background_points_left.copy()
    right_points = [
        (80, 80),
        (980, 120),
        (950, 680),
        (110, 720)
    ]

    # 设置单应变换
    compositor.setup_homography_method(
        background_points_left=background_points_left,
        background_points_right=background_points_right,
        left_points=left_points,
        right_points=right_points
    )

    # 加载前景图像和掩码
    foreground_left = cv2.imread("path/to/foreground_left.jpg")
    foreground_right = cv2.imread("path/to/foreground_right.jpg")

    compositor.load_foreground_masks(
        mask_left_path="path/to/mask_left.png",
        mask_right_path="path/to/mask_right.png"
    )

    # 处理并生成结果
    target_size = (1920, 1080)  # (width, height)
    result_left, result_right = compositor.process_stereo_pair(
        foreground_left,
        foreground_right,
        target_size,
        feather_radius=5,
        color_matching=True
    )

    # 保存结果
    cv2.imwrite("output_left.jpg", result_left)
    cv2.imwrite("output_right.jpg", result_right)
    print("Results saved!")


def example_homography_auto_features():
    """示例2: 使用自动特征匹配的单应变换"""
    print("\n=== Example 2: Homography with Auto Feature Matching ===\n")

    # 创建合成器
    compositor = BackgroundCompositor(method=WarpMethod.HOMOGRAPHY)

    # 加载背景图
    compositor.load_background("path/to/background.jpg")

    # 加载参考图像（原始左右视角的背景）
    left_reference = cv2.imread("path/to/original_left.jpg")
    right_reference = cv2.imread("path/to/original_right.jpg")

    # 使用SIFT自动匹配特征
    compositor.setup_homography_method(
        left_reference_img=left_reference,
        right_reference_img=right_reference,
        auto_method="sift"  # 或 "orb"
    )

    # 加载前景
    foreground_left = cv2.imread("path/to/foreground_left.jpg")
    foreground_right = cv2.imread("path/to/foreground_right.jpg")

    compositor.load_foreground_masks(
        mask_left_path="path/to/mask_left.png",
        mask_right_path="path/to/mask_right.png"
    )

    # 处理
    target_size = (1920, 1080)
    result_left, result_right = compositor.process_stereo_pair(
        foreground_left,
        foreground_right,
        target_size
    )

    # 保存
    cv2.imwrite("output_left_auto.jpg", result_left)
    cv2.imwrite("output_right_auto.jpg", result_right)
    print("Results saved!")


def example_depth_based():
    """示例3: 使用深度图的重投影"""
    print("\n=== Example 3: Depth-based Warping ===\n")

    # 创建合成器
    compositor = BackgroundCompositor(method=WarpMethod.DEPTH)

    # 加载背景图和深度图
    compositor.load_background("path/to/background.jpg")
    depth_map = cv2.imread("path/to/depth_map.png", cv2.IMREAD_ANYDEPTH)

    # 如果深度图是归一化的，需要转换到实际深度（米）
    if depth_map.dtype == np.uint8:
        # 假设8位深度图，映射到0.5-10米
        depth_map = depth_map.astype(np.float32) / 255.0 * 9.5 + 0.5
    elif depth_map.dtype == np.uint16:
        # 16位深度图
        depth_map = depth_map.astype(np.float32) / 1000.0  # 毫米转米

    # 设置深度方法参数
    compositor.setup_depth_method(
        depth_map=depth_map,
        background_image_shape=(1080, 1920),  # (H, W)
        target_image_shape=(1080, 1920),
        hfov_deg=70.0,           # 水平视场角
        baseline=0.065,          # 基线距离（米）
        rotation_y_deg=5.0,      # 对称旋转角度
        background_exif_path="path/to/background.jpg"  # 尝试从EXIF读取
    )

    # 加载前景
    foreground_left = cv2.imread("path/to/foreground_left.jpg")
    foreground_right = cv2.imread("path/to/foreground_right.jpg")

    compositor.load_foreground_masks(
        mask_left_path="path/to/mask_left.png",
        mask_right_path="path/to/mask_right.png"
    )

    # 处理
    target_size = (1920, 1080)
    result_left, result_right = compositor.process_stereo_pair(
        foreground_left,
        foreground_right,
        target_size,
        feather_radius=7
    )

    # 保存
    cv2.imwrite("output_left_depth.jpg", result_left)
    cv2.imwrite("output_right_depth.jpg", result_right)
    print("Results saved!")


def example_without_foreground_mask():
    """示例4: 仅生成扭曲背景（不合成前景）"""
    print("\n=== Example 4: Generate Warped Backgrounds Only ===\n")

    # 创建合成器
    compositor = BackgroundCompositor(method=WarpMethod.HOMOGRAPHY)

    # 加载背景
    compositor.load_background("path/to/background.jpg")

    # 简单设置（使用特征匹配）
    left_reference = cv2.imread("path/to/original_left.jpg")
    right_reference = cv2.imread("path/to/original_right.jpg")

    compositor.setup_homography_method(
        left_reference_img=left_reference,
        right_reference_img=right_reference,
        auto_method="sift"
    )

    # 仅生成扭曲背景
    target_size = (1920, 1080)
    warped_left, warped_right = compositor.generate_warped_backgrounds(target_size)

    # 保存
    cv2.imwrite("warped_bg_left.jpg", warped_left)
    cv2.imwrite("warped_bg_right.jpg", warped_right)
    print("Warped backgrounds saved!")


def example_programmatic_mask():
    """示例5: 程序生成前景掩码"""
    print("\n=== Example 5: Programmatic Foreground Mask ===\n")

    compositor = BackgroundCompositor(method=WarpMethod.HOMOGRAPHY)
    compositor.load_background("path/to/background.jpg")

    # 加载前景图像
    foreground_left = cv2.imread("path/to/foreground_left.jpg")
    foreground_right = cv2.imread("path/to/foreground_right.jpg")

    # 程序生成掩码（例如：基于颜色阈值）
    # 这里假设前景是绿幕，生成反向掩码
    hsv_left = cv2.cvtColor(foreground_left, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    green_mask_left = cv2.inRange(hsv_left, lower_green, upper_green)
    foreground_mask_left = 255 - green_mask_left  # 前景为白色

    hsv_right = cv2.cvtColor(foreground_right, cv2.COLOR_BGR2HSV)
    green_mask_right = cv2.inRange(hsv_right, lower_green, upper_green)
    foreground_mask_right = 255 - green_mask_right

    # 加载掩码
    compositor.load_foreground_masks(
        mask_left=foreground_mask_left,
        mask_right=foreground_mask_right
    )

    # 设置单应变换并处理
    left_reference = cv2.imread("path/to/original_left.jpg")
    right_reference = cv2.imread("path/to/original_right.jpg")

    compositor.setup_homography_method(
        left_reference_img=left_reference,
        right_reference_img=right_reference,
        auto_method="sift"
    )

    target_size = (1920, 1080)
    result_left, result_right = compositor.process_stereo_pair(
        foreground_left,
        foreground_right,
        target_size
    )

    cv2.imwrite("output_left_chromakey.jpg", result_left)
    cv2.imwrite("output_right_chromakey.jpg", result_right)
    print("Results saved!")


if __name__ == "__main__":
    print("Background Warping System - Usage Examples")
    print("=" * 60)
    print("\nPlease uncomment the example you want to run and adjust paths.\n")

    # 取消注释你想运行的示例：

    # example_homography_manual_points()
    # example_homography_auto_features()
    # example_depth_based()
    # example_without_foreground_mask()
    # example_programmatic_mask()

    print("\nExamples completed. Check the output files.")

