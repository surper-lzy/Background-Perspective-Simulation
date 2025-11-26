"""
端到端背景替换流程 - 集成 ZoeDepth 深度估计
完整流程：
1. 加载输入图像（背景、前景、掩码）
2. 使用 ZoeDepth 自动估计背景深度
3. 使用深度引导方法进行背景扭曲
4. 合成最终结果
"""
import cv2
import numpy as np
import os
from background_compositor import BackgroundCompositor, WarpMethod
from depth_estimator import ZoeDepthEstimator


def run_end_to_end_with_depth_estimation(
    background_path: str,
    foreground_left_path: str,
    foreground_right_path: str,
    mask_left_path: str,
    mask_right_path: str,
    reference_left_path: str = None,
    reference_right_path: str = None,
    output_dir: str = "output",
    use_depth_method: bool = True,
    depth_params: dict = None
):
    """
    端到端背景替换流程

    Args:
        background_path: 新背景图路径
        foreground_left_path: 左相机前景图路径
        foreground_right_path: 右相机前景图路径
        mask_left_path: 左前景掩码路径
        mask_right_path: 右前景掩码路径
        reference_left_path: 左参考图路径（仅两阶段方法需要）
        reference_right_path: 右参考图路径（仅两阶段方法需要）
        output_dir: 输出目录
        use_depth_method: 是否使用深度方法（True）或两阶段方法（False）
        depth_params: 深度方法参数字典
    """
    print("\n" + "=" * 80)
    print("端到端背景替换流程 - ZoeDepth 集成")
    print("=" * 80 + "\n")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # ===== 步骤1: 加载输入图像 =====
    print("[步骤 1/5] 加载输入图像...")
    background_img = cv2.imread(background_path)
    foreground_left = cv2.imread(foreground_left_path)
    foreground_right = cv2.imread(foreground_right_path)

    if background_img is None:
        raise FileNotFoundError(f"背景图像未找到: {background_path}")
    if foreground_left is None or foreground_right is None:
        raise FileNotFoundError("前景图像未找到")

    H, W = foreground_left.shape[:2]
    print(f"   ✓ 背景图: {background_img.shape}")
    print(f"   ✓ 前景图: {foreground_left.shape}")

    if use_depth_method:
        # ===== 深度引导方法 =====
        print("\n使用方法: 深度引导重投影 + ZoeDepth 自动估计\n")

        # ===== 步骤2: ZoeDepth 深度估计 =====
        print("[步骤 2/5] 使用 ZoeDepth 估计背景深度...")
        estimator = ZoeDepthEstimator(model_type="ZoeD_NK", use_local=True)

        # 估计深度
        depth_meters = estimator.estimate_depth(background_img, output_size=(W, H))

        # 保存深度图
        depth_output_path = os.path.join(output_dir, "estimated_depth.png")
        depth_uint16 = (depth_meters * 1000.0).astype(np.uint16)
        cv2.imwrite(depth_output_path, depth_uint16)
        print(f"   ✓ 深度图已保存: {depth_output_path}")

        # 保存深度可视化
        depth_vis = estimator._visualize_depth(depth_meters)
        depth_vis_path = os.path.join(output_dir, "estimated_depth_visualization.jpg")
        cv2.imwrite(depth_vis_path, depth_vis)
        print(f"   ✓ 深度可视化已保存: {depth_vis_path}")

        # ===== 步骤3: 设置深度重投影参数 =====
        print("\n[步骤 3/5] 设置深度重投影参数...")

        # 使用默认参数或用户提供的参数
        if depth_params is None:
            depth_params = {
                'hfov_deg': 70.0,        # 水平视场角
                'baseline': 0.065,        # 基线距离（米）
                'rotation_y_deg': 5.0     # Y轴旋转角度
            }

        compositor = BackgroundCompositor(method=WarpMethod.DEPTH)
        compositor.load_background(background_path)

        compositor.setup_depth_method(
            depth_map=depth_meters,
            background_image_shape=background_img.shape[:2],
            target_image_shape=(H, W),
            **depth_params
        )
        print(f"   ✓ 深度参数: HFOV={depth_params['hfov_deg']}°, "
              f"Baseline={depth_params['baseline']}m, "
              f"Rotation={depth_params['rotation_y_deg']}°")

    else:
        # ===== 两阶段单应方法 =====
        print("\n使用方法: 两阶段单应变换（不使用深度估计）\n")

        if reference_left_path is None or reference_right_path is None:
            raise ValueError("两阶段方法需要提供参考图像（reference_left 和 reference_right）")

        print("[步骤 2/5] 跳过深度估计（使用几何方法）")
        print("[步骤 3/5] 设置两阶段单应变换...")

        # 加载参考图
        ref_left = cv2.imread(reference_left_path)
        ref_right = cv2.imread(reference_right_path)

        if ref_left is None or ref_right is None:
            raise FileNotFoundError("参考图像未找到")

        # 定义背景平面在左视角中的位置
        plane_corners = [
            (int(W * 0.1), int(H * 0.1)),      # 左上
            (int(W * 0.9), int(H * 0.1)),      # 右上
            (int(W * 0.9), int(H * 0.9)),      # 右下
            (int(W * 0.1), int(H * 0.9))       # 左下
        ]

        compositor = BackgroundCompositor(method=WarpMethod.HOMOGRAPHY)
        compositor.load_background(background_path)

        compositor.setup_two_stage_homography(
            ref_left,
            ref_right,
            plane_corners,
            auto_method="sift"
        )
        print(f"   ✓ 两阶段单应变换设置完成")

    # ===== 步骤4: 加载掩码并处理 =====
    print("\n[步骤 4/5] 加载前景掩码并合成...")
    compositor.load_foreground_masks(
        mask_left_path=mask_left_path,
        mask_right_path=mask_right_path
    )

    # 处理立体对
    result_left, result_right = compositor.process_stereo_pair(
        foreground_left,
        foreground_right,
        (W, H),
        feather_radius=7,
        color_matching=False
    )

    # ===== 步骤5: 保存结果 =====
    print("\n[步骤 5/5] 保存最终结果...")

    method_name = "depth" if use_depth_method else "two_stage"
    output_left_path = os.path.join(output_dir, f"result_{method_name}_left.jpg")
    output_right_path = os.path.join(output_dir, f"result_{method_name}_right.jpg")

    cv2.imwrite(output_left_path, result_left)
    cv2.imwrite(output_right_path, result_right)

    print(f"   ✓ 左视角结果: {output_left_path}")
    print(f"   ✓ 右视角结果: {output_right_path}")

    # 也保存仅背景扭曲的结果（用于对比）
    warped_left, warped_right = compositor.generate_warped_backgrounds((W, H))
    cv2.imwrite(os.path.join(output_dir, f"warped_bg_{method_name}_left.jpg"), warped_left)
    cv2.imwrite(os.path.join(output_dir, f"warped_bg_{method_name}_right.jpg"), warped_right)

    print("\n" + "=" * 80)
    print("✅ 端到端流程完成！")
    print("=" * 80)
    print(f"\n请查看输出目录: {output_dir}/")
    print(f"  - result_{method_name}_left.jpg (最终左视角)")
    print(f"  - result_{method_name}_right.jpg (最终右视角)")
    if use_depth_method:
        print(f"  - estimated_depth.png (估计的深度图)")
        print(f"  - estimated_depth_visualization.jpg (深度可视化)")
    print(f"  - warped_bg_{method_name}_*.jpg (仅背景扭曲)")


def run_with_test_data():
    """使用测试数据运行完整流程"""
    print("\n" + "=" * 80)
    print("使用测试数据运行端到端流程")
    print("=" * 80)

    # 检查测试数据是否存在
    if not os.path.exists("test_data/background.jpg"):
        print("\n⚠️  测试数据不存在，正在生成...")
        import generate_test_data
        generate_test_data.create_test_data()

    # ===== 方法1: 深度引导 + ZoeDepth =====
    print("\n" + "▶" * 40)
    print("方法1: 深度引导重投影 + ZoeDepth 自动估计")
    print("▶" * 40)

    run_end_to_end_with_depth_estimation(
        background_path="test_data/037.jpg",
        foreground_left_path="test_data/foreground_left.jpg",
        foreground_right_path="test_data/foreground_right.jpg",
        mask_left_path="test_data/mask_left.png",
        mask_right_path="test_data/mask_right.png",
        output_dir="output",
        use_depth_method=True,
        depth_params={
            'hfov_deg': 70.0,
            'baseline': 0.065,
            'rotation_y_deg': 5.0
        }
    )

    # ===== 方法2: 两阶段单应 =====
    print("\n\n" + "▶" * 40)
    print("方法2: 两阶段单应变换（对比）")
    print("▶" * 40)

    run_end_to_end_with_depth_estimation(
        background_path="test_data/037.jpg",
        foreground_left_path="test_data/foreground_left.jpg",
        foreground_right_path="test_data/foreground_right.jpg",
        mask_left_path="test_data/mask_left.png",
        mask_right_path="test_data/mask_right.png",
        reference_left_path="test_data/reference_left.jpg",
        reference_right_path="test_data/reference_right.jpg",
        output_dir="output",
        use_depth_method=False
    )

    print("\n" + "=" * 80)
    print("🎉 所有方法运行完成！请查看 output/ 目录对比结果")
    print("=" * 80)


if __name__ == "__main__":
    # 运行测试数据的完整流程
    run_with_test_data()

