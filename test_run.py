"""
测试运行脚本 - 使用生成的测试数据运行所有方法
【更新】添加两阶段自动匹配方法，移除旧的方法A/B
运行前请先执行: python generate_test_data.py
"""
import cv2
import numpy as np
import os
from background_compositor import BackgroundCompositor, WarpMethod


def test_method_two_stage():
    """测试方法：两阶段自动匹配（从参考对学习几何 → 应用到新背景）"""
    print("\n" + "=" * 80)
    print("测试方法：两阶段自动匹配")
    print("=" * 80 + "\n")

    try:
        # 创建合成器
        compositor = BackgroundCompositor(method=WarpMethod.HOMOGRAPHY)

        # 加载新背景（自然风景）
        compositor.load_background("test_data/background.jpg")

        # 加载参考对（办公室场景）
        left_reference = cv2.imread("test_data/reference_left.jpg")
        right_reference = cv2.imread("test_data/reference_right.jpg")

        if left_reference is None or right_reference is None:
            raise RuntimeError("无法加载参考图像")

        # 定义背景平面在左视角中的位置（四个角点）
        # 这些点定义了新背景在左相机视角中应该显示的位置
        plane_corners_in_left = [
            (150, 150),      # 左上
            (1770, 150),     # 右上
            (1770, 930),     # 右下
            (150, 930)       # 左下
        ]

        # 使用两阶段方法设置
        print("阶段1: 从参考对学习相机几何关系...")
        print("阶段2: 将新背景映射到左右视角...")
        compositor.setup_two_stage_homography(
            left_reference,
            right_reference,
            plane_corners_in_left,
            auto_method="sift"
        )

        # 加载前景和掩码
        foreground_left = cv2.imread("test_data/foreground_left.jpg")
        foreground_right = cv2.imread("test_data/foreground_right.jpg")

        compositor.load_foreground_masks(
            mask_left_path="test_data/mask_left.png",
            mask_right_path="test_data/mask_right.png"
        )

        # 处理
        result_left, result_right = compositor.process_stereo_pair(
            foreground_left,
            foreground_right,
            (1920, 1080),
            feather_radius=5
        )

        # 保存结果
        os.makedirs("output", exist_ok=True)
        cv2.imwrite("output/two_stage_left.jpg", result_left)
        cv2.imwrite("output/two_stage_right.jpg", result_right)

        print("\n✅ 两阶段匹配测试成功！")
        print("   结果已保存到:")
        print("   - output/two_stage_left.jpg")
        print("   - output/two_stage_right.jpg")

        # 显示质量指标
        print("\n质量指标:")
        if "reference_geometry" in compositor.warper.meta:
            print(f"   参考对几何: {compositor.warper.meta['reference_geometry']}")
        print(f"   左相机: {compositor.warper.meta.get('left', {})}")
        print(f"   右相机: {compositor.warper.meta.get('right', {})}")

    except Exception as e:
        print(f"\n❌ 两阶段匹配测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def test_method_depth():
    """测试方法：深度引导重投影"""
    print("\n" + "=" * 80)
    print("测试方法：深度引导重投影")
    print("=" * 80 + "\n")

    try:
        # 创建合成器
        compositor = BackgroundCompositor(method=WarpMethod.DEPTH)

        # 加载新背景
        compositor.load_background("test_data/background.jpg")

        # 加载深度图
        depth_map = cv2.imread("test_data/depth_map.png", cv2.IMREAD_ANYDEPTH)
        if depth_map is None:
            raise RuntimeError("无法读取深度图")

        # 转换深度图单位（从毫米到米）
        depth_map = depth_map.astype(np.float32) / 1000.0
        print(f"   深度范围: {depth_map.min():.2f}m - {depth_map.max():.2f}m")

        # 设置深度方法
        compositor.setup_depth_method(
            depth_map=depth_map,
            background_image_shape=(1080, 1920),
            target_image_shape=(1080, 1920),
            hfov_deg=70.0,
            baseline=0.065,
            rotation_y_deg=5.0
        )

        # 加载前景和掩码
        foreground_left = cv2.imread("test_data/foreground_left.jpg")
        foreground_right = cv2.imread("test_data/foreground_right.jpg")

        compositor.load_foreground_masks(
            mask_left_path="test_data/mask_left.png",
            mask_right_path="test_data/mask_right.png"
        )

        # 处理
        result_left, result_right = compositor.process_stereo_pair(
            foreground_left,
            foreground_right,
            (1920, 1080),
            feather_radius=7
        )

        # 保存结果
        os.makedirs("output", exist_ok=True)
        cv2.imwrite("output/depth_left.jpg", result_left)
        cv2.imwrite("output/depth_right.jpg", result_right)

        print("\n✅ 深度重投影测试成功！")
        print("   结果已保存到:")
        print("   - output/depth_left.jpg")
        print("   - output/depth_right.jpg")

    except Exception as e:
        print(f"\n❌ 深度重投影测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def test_warped_backgrounds_only():
    """测试：仅生成扭曲背景（不合成前景，用于调试）"""
    print("\n" + "=" * 80)
    print("测试: 仅生成扭曲背景（两阶段方法，用于调试）")
    print("=" * 80 + "\n")

    try:
        compositor = BackgroundCompositor(method=WarpMethod.HOMOGRAPHY)
        compositor.load_background("test_data/background.jpg")

        # 加载参考对
        left_reference = cv2.imread("test_data/reference_left.jpg")
        right_reference = cv2.imread("test_data/reference_right.jpg")

        # 定义平面位置
        plane_corners_in_left = [
            (150, 150),
            (1770, 150),
            (1770, 930),
            (150, 930)
        ]

        # 两阶段设置
        compositor.setup_two_stage_homography(
            left_reference,
            right_reference,
            plane_corners_in_left,
            auto_method="sift"
        )

        # 仅生成背景
        warped_left, warped_right = compositor.generate_warped_backgrounds((1920, 1080))

        # 保存
        os.makedirs("output", exist_ok=True)
        cv2.imwrite("output/warped_bg_left.jpg", warped_left)
        cv2.imwrite("output/warped_bg_right.jpg", warped_right)

        print("\n✅ 背景扭曲测试成功！")
        print("   结果已保存到:")
        print("   - output/warped_bg_left.jpg")
        print("   - output/warped_bg_right.jpg")
        print("\n提示：检查这些图像，确认背景扭曲效果是否符合预期")

    except Exception as e:
        print(f"\n❌ 背景扭曲测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def check_test_data():
    """检查测试数据是否存在"""
    print("\n检查测试数据...")

    required_files = [
        "test_data/background.jpg",
        "test_data/foreground_left.jpg",
        "test_data/foreground_right.jpg",
        "test_data/mask_left.png",
        "test_data/mask_right.png",
        "test_data/reference_left.jpg",
        "test_data/reference_right.jpg",
        "test_data/depth_map.png"
    ]

    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} [缺失]")
            all_exist = False

    if not all_exist:
        print("\n⚠️  缺少测试数据文件！")
        print("   请先运行: python generate_test_data.py")
        return False

    print("\n✅ 所有测试数据文件就绪！\n")
    return True


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("背景变换系统 - 自动化测试（两阶段匹配版本）")
    print("=" * 80)

    # 检查测试数据
    if not check_test_data():
        return

    print("\n测试说明:")
    print("  - 参考场景: 办公室风格（墙壁/窗户/门）")
    print("  - 新背景: 自然风景（天空/山/树/建筑）")
    print("  - 两者完全不同，使用两阶段匹配方法\n")

    # 运行所有测试
    print("开始运行测试...\n")

    # 测试1: 仅背景扭曲（调试用）
    test_warped_backgrounds_only()

    # 测试2: 两阶段自动匹配
    test_method_two_stage()

    # 测试3: 深度重投影
    test_method_depth()

    # 总结
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    print("\n所有结果已保存到 output/ 目录:")
    print("  - warped_bg_left.jpg / warped_bg_right.jpg  (仅背景)")
    print("  - two_stage_left.jpg / two_stage_right.jpg  (两阶段匹配)")
    print("  - depth_left.jpg / depth_right.jpg          (深度重投影)")
    print("\n请检查输出图像质量，根据需要调整参数。")
    print("\n参数调整建议:")
    print("  - plane_corners_in_left: 调整背景在视角中的位置和大小")
    print("  - hfov_deg: 调整视场角（60-80度）")
    print("  - baseline: 调整左右相机距离（0.05-0.10米）")
    print("  - rotation_y_deg: 调整相机旋转角度（3-7度）")


if __name__ == "__main__":
    main()
