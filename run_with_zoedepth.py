"""
端到端背景替换流程 - 集成 ZoeDepth 深度估计 + 鱼眼投影
完整流程：
1. 保持前景和掩码为原始鱼眼格式（避免画质损失）
2. 对针孔背景图C进行深度估计
3. 基于深度生成虚拟的针孔立体背景（左/右）
4. 将针孔背景重投影为鱼眼风格（匹配参考图AB）
5. 在鱼眼域合成最终结果
"""
import cv2
import numpy as np
import os
from background_compositor import BackgroundCompositor, WarpMethod
from depth_estimator import ZoeDepthEstimator


# ==========================================
# 核心工具：针孔 -> 鱼眼 投影
# ==========================================

def convert_pinhole_to_fisheye(pinhole_img, fisheye_K, fisheye_xi, fisheye_D, pinhole_K, output_size):
    """
    将针孔图像重投影为鱼眼图像（逆向映射）

    原理：
    1. 遍历目标鱼眼图像的每个像素
    2. 反投影到3D单位球面（使用Mei全向相机模型）
    3. 投影到针孔平面
    4. 采样颜色值

    Args:
        pinhole_img: 输入的针孔图像 (H, W, 3)
        fisheye_K: 目标鱼眼相机内参矩阵 (3x3 list)
        fisheye_xi: 目标鱼眼相机的xi参数 (float)
        fisheye_D: 目标鱼眼相机的畸变系数 (4元素list, 通常为[0,0,0,0])
        pinhole_K: 源针孔相机内参矩阵 (3x3 list)
        output_size: 输出鱼眼图像尺寸 (height, width)

    Returns:
        鱼眼投影后的图像 (H, W, 3)
    """
    h, w = output_size

    # 1. 生成目标鱼眼图像的像素坐标网格
    grid_y, grid_x = np.indices((h, w), dtype=np.float32)
    points_fisheye = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1).reshape(-1, 1, 2)

    # 2. 将鱼眼像素反投影到3D单位球面
    try:
        points_3d = cv2.omnidir.undistortPoints(
            points_fisheye,
            np.array(fisheye_K, dtype=np.float64),
            np.array(fisheye_D, dtype=np.float64),
            np.array([fisheye_xi], dtype=np.float64),
            np.eye(3, dtype=np.float64)  # R
        )
    except AttributeError:
        raise ImportError(
            "需要 opencv-contrib-python 包含 cv2.omnidir 模块\n"
            "请运行: pip uninstall opencv-python && pip install opencv-contrib-python"
        )

    # 3. 将3D球面点投影到虚拟针孔相机平面
    X = points_3d[:, 0, 0]
    Y = points_3d[:, 0, 1]

    # 归一化坐标，假设Z=1的平面
    # 在undistortPoints中已经得到了归一化坐标

    # 应用针孔相机内参
    fx, fy = pinhole_K[0][0], pinhole_K[1][1]
    cx, cy = pinhole_K[0][2], pinhole_K[1][2]

    u_pin = X * fx + cx
    v_pin = Y * fy + cy

    # 4. 生成重映射表
    map_x = u_pin.reshape(h, w).astype(np.float32)
    map_y = v_pin.reshape(h, w).astype(np.float32)

    # 5. 执行重映射（采样）
    distorted_img = cv2.remap(
        pinhole_img,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return distorted_img


# ==========================================
# 核心工具：全景图 -> 鱼眼 提取（新增方法3）
# ==========================================

def extract_fisheye_from_equirectangular(equirect_img, fisheye_K, fisheye_xi, fisheye_D, output_size, R=None):
    """
    从全景图(Equirectangular)中提取鱼眼视角图像
    原理：球面几何重投影 (Spherical Re-projection)

    Args:
        equirect_img: 输入的全景图 (H, W, 3) - Equirectangular格式
        fisheye_K: 目标鱼眼相机内参矩阵 (3x3 list)
        fisheye_xi: 目标鱼眼相机的xi参数 (float)
        fisheye_D: 目标鱼眼相机的畸变系数 (4元素list, 通常为[0,0,0,0])
        output_size: 输出鱼眼图像尺寸 (height, width)
        R: 旋转矩阵 (3x3)，控制相机的朝向，默认None表示朝向全景图中心

    Returns:
        生成的鱼眼图像 (H, W, 3)
    """
    h_out, w_out = output_size
    h_pano, w_pano = equirect_img.shape[:2]

    if R is None:
        R = np.eye(3, dtype=np.float32)

    # 1. 生成目标鱼眼图像的像素坐标网格
    grid_y, grid_x = np.indices((h_out, w_out), dtype=np.float32)
    points_fisheye = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1).reshape(-1, 1, 2)

    # 2. 将鱼眼像素反投影到相机坐标系的单位球面上 (x, y, z)
    try:
        # undistortPoints 返回的是归一化坐标 (x, y)，我们需要将其转换为 3D 方向
        points_undist = cv2.omnidir.undistortPoints(
            points_fisheye,
            np.array(fisheye_K, dtype=np.float64),
            np.array(fisheye_D, dtype=np.float64),
            np.array([fisheye_xi], dtype=np.float64),
            np.eye(3, dtype=np.float64)
        )
        # points_undist 形状可能是 (N, 1, 2) 或 (N, 2)，统一处理
        points_undist = points_undist.reshape(-1, 2)

        # 将 2D 归一化坐标 (x, y) 转换为 3D 单位球面坐标 (x, y, z)
        # 假设这些点在 Z=1 平面上，然后归一化到单位球面
        x_norm = points_undist[:, 0]
        y_norm = points_undist[:, 1]
        z_norm = np.ones_like(x_norm)

        # 归一化到单位球面
        norm = np.sqrt(x_norm*x_norm + y_norm*y_norm + z_norm*z_norm)
        points_3d_cam = np.stack([x_norm/norm, y_norm/norm, z_norm/norm], axis=1)

    except AttributeError:
        raise ImportError(
            "需要 opencv-contrib-python 包含 cv2.omnidir 模块\n"
            "请运行: pip uninstall opencv-python && pip install opencv-contrib-python"
        )


    # 3. 应用旋转矩阵 (相机坐标系 -> 世界/全景图坐标系)
    points_3d_world = points_3d_cam @ R.T

    x = points_3d_world[:, 0]
    y = points_3d_world[:, 1]
    z = points_3d_world[:, 2]

    # 4. 笛卡尔坐标 -> 球面坐标 (经度/纬度)
    # 经度 phi: atan2(x, z) -> [-pi, pi]
    # 纬度 theta: arcsin(-y) -> [-pi/2, pi/2]
    norm = np.sqrt(x*x + y*y + z*z)
    phi = np.arctan2(x, z)
    theta = np.arcsin(-y / (norm + 1e-6))

    # 5. 球面坐标 -> 全景图UV坐标
    u = (phi + np.pi) / (2 * np.pi) * w_pano
    v = (np.pi / 2.0 - theta) / np.pi * h_pano

    # 6. 重映射（采样）
    map_x = u.reshape(h_out, w_out).astype(np.float32)
    map_y = v.reshape(h_out, w_out).astype(np.float32)

    # 边界处理：全景图左右是循环的，使用BORDER_WRAP
    fisheye_view = cv2.remap(
        equirect_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP  # 全景图左右边缘循环连接
    )

    return fisheye_view


# ==========================================
# 辅助函数：掩码合成（直接粘贴版本）
# ==========================================

def composite_with_mask(foreground, background, mask, threshold=None, use_binary=True):
    """
    使用掩码进行合成，直接粘贴前景到背景上

    Args:
        foreground: 前景图像 (H, W, 3) - 分割后的图像
        background: 背景图像 (H, W, 3)
        mask: 掩码 (H, W) - 255为前景，0为背景
        threshold: 二值化阈值，None则自动计算（推荐）
        use_binary: 是否二值化掩码，True可避免半透明边缘

    Returns:
        合成后的图像 (H, W, 3)
    """
    # 1. 确保尺寸匹配
    if mask.shape[:2] != foreground.shape[:2]:
        print(f"   [INFO] 调整掩码尺寸: {mask.shape[:2]} -> {foreground.shape[:2]}")
        mask = cv2.resize(mask, (foreground.shape[1], foreground.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    if background.shape[:2] != foreground.shape[:2]:
        print(f"   [INFO] 调整背景尺寸: {background.shape[:2]} -> {foreground.shape[:2]}")
        background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))

    # 2. 智能阈值选择
    if threshold is None:
        # 自动选择阈值：使用 Otsu 方法或者简单的中值
        # 如果掩码是二值的（只有0和255），使用127
        # 否则使用 Otsu 自动计算
        unique_values = np.unique(mask)
        if len(unique_values) <= 3 and 255 in unique_values and 0 in unique_values:
            # 掩码基本是二值的
            threshold = 127
            print(f"   [INFO] 检测到二值掩码，使用阈值 {threshold}")
        else:
            # 使用 Otsu 方法自动计算最佳阈值
            threshold, _ = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            print(f"   [INFO] 使用 Otsu 自动阈值: {threshold:.0f}")

    # 3. 二值化掩码
    if use_binary:
        _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    else:
        binary_mask = mask

    # 4. 创建输出图像，先复制背景
    result = background.copy()

    # 5. 直接在掩码区域粘贴前景，使用布尔索引
    # 将掩码转换为布尔数组（只有大于127的才被认为是前景）
    mask_bool = binary_mask > 127

    # 在掩码为True的地方，直接使用前景像素（包括黑色像素）
    # 这确保了前景中的黑色物体不会被误判为背景
    result[mask_bool] = foreground[mask_bool]

    return result



def run_fisheye_background_replacement(
    background_path: str,
    foreground_left_path: str,
    foreground_right_path: str,
    mask_left_path: str,
    mask_right_path: str,
    params_left: dict,
    params_right: dict,
    output_dir: str = "output",
    depth_params: dict = None,
    reference_left_path: str = None,
    reference_right_path: str = None,
    use_depth_method: bool = True
):
    """
    鱼眼立体背景替换 - 主流程

    工作流程：
    1. 加载原始鱼眼前景和掩码（保持不动）
    2. 对针孔背景图进行深度估计
    3. 生成虚拟的针孔立体背景（左/右）
    4. 将针孔背景投影为鱼眼风格
    5. 在鱼眼域合成最终结果

    Args:
        background_path: 新背景图路径（普通针孔相机拍摄）
        foreground_left_path: 左相机前景图路径（鱼眼）
        foreground_right_path: 右相机前景图路径（鱼眼）
        mask_left_path: 左前景掩码路径（鱼眼）
        mask_right_path: 右前景掩码路径（鱼眼）
        params_left: 左相机鱼眼参数 {'K': [[...]], 'xi': float}
        params_right: 右相机鱼眼参数 {'K': [[...]], 'xi': float}
        output_dir: 输出目录
        depth_params: 深度方法参数 {'hfov_deg', 'baseline', 'rotation_y_deg'}
        reference_left_path: 左参考图路径（两阶段方法需要）
        reference_right_path: 右参考图路径（两阶段方法需要）
        use_depth_method: True使用深度方法，False使用两阶段方法
    """
    print("\n" + "=" * 80)
    print("鱼眼立体背景替换 - 新工作流")
    print("策略: 保持前景为鱼眼 + 对背景加畸变")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # ===== 步骤1: 加载原始鱼眼图像（前景 & 掩码）=====
    print("\n[步骤 1/6] 加载原始鱼眼前景和掩码...")
    fg_left = cv2.imread(foreground_left_path)
    fg_right = cv2.imread(foreground_right_path)
    mask_left = cv2.imread(mask_left_path, cv2.IMREAD_GRAYSCALE)
    mask_right = cv2.imread(mask_right_path, cv2.IMREAD_GRAYSCALE)
    bg_img = cv2.imread(background_path)

    if any(x is None for x in [fg_left, fg_right, mask_left, mask_right, bg_img]):
        raise FileNotFoundError("部分输入图像未找到，请检查路径")

    H_fish, W_fish = fg_left.shape[:2]
    print(f"   [OK] 鱼眼前景尺寸: {W_fish}x{H_fish}")
    print(f"   [OK] 针孔背景尺寸: {bg_img.shape[1]}x{bg_img.shape[0]}")

    # ===== 步骤2: 深度估计（在针孔背景图上进行）=====
    print("\n[步骤 2/6] ZoeDepth 深度估计（针孔背景）...")
    estimator = ZoeDepthEstimator(model_type="ZoeD_NK", use_local=True)

    # 对背景图进行深度估计
    depth_meters = estimator.estimate_depth(bg_img)

    # 保存深度图
    depth_vis = estimator._visualize_depth(depth_meters)
    cv2.imwrite(os.path.join(output_dir, "depth_visualization.jpg"), depth_vis)
    depth_uint16 = (depth_meters * 1000.0).astype(np.uint16)
    cv2.imwrite(os.path.join(output_dir, "depth_map.png"), depth_uint16)
    print(f"   [OK] 深度图已保存")

    # ===== 步骤3: 生成虚拟针孔立体背景（Stereo Warping）=====
    print("\n[步骤 3/6] 生成虚拟针孔立体背景...")

    # 定义虚拟针孔相机（中间态）
    virtual_h, virtual_w = bg_img.shape[:2]

    # 设置虚拟相机的焦距（根据背景图调整）
    virtual_f = None
    if depth_params and depth_params.get('virtual_focal_length'):
        virtual_f = depth_params['virtual_focal_length']
    if virtual_f is None:
        virtual_f = virtual_w / 2.0

    virtual_K = [
        [virtual_f, 0, virtual_w / 2.0],
        [0, virtual_f, virtual_h / 2.0],
        [0, 0, 1]
    ]

    # 设置深度参数
    if depth_params is None:
        depth_params = {
            'hfov_deg': 90.0,
            'baseline': 0.65,
            'rotation_y_deg': 0.0
        }
    else:
        depth_params = depth_params.copy()

    depth_params_for_warper = {k: v for k, v in depth_params.items() if k != 'virtual_focal_length'}

    # 使用 BackgroundCompositor 生成立体背景
    if use_depth_method:
        compositor = BackgroundCompositor(method=WarpMethod.DEPTH)
        compositor.load_background(background_path)
        compositor.setup_depth_method(
            depth_map=depth_meters,
            background_image_shape=bg_img.shape[:2],
            target_image_shape=(virtual_h, virtual_w),
            **depth_params_for_warper
        )
        print(f"   [OK] 深度方法参数: HFOV={depth_params_for_warper.get('hfov_deg', '未知')}°, "
              f"Baseline={depth_params_for_warper.get('baseline', '未知')}m")
    else:
        # 两阶段方法
        if reference_left_path is None or reference_right_path is None:
            raise ValueError("两阶段方法需要提供参考图像")

        print("   使用两阶段单应方法...")
        ref_left = cv2.imread(reference_left_path)
        ref_right = cv2.imread(reference_right_path)

        compositor = BackgroundCompositor(method=WarpMethod.HOMOGRAPHY)
        compositor.load_background(background_path)

        plane_corners = [
            (int(virtual_w * 0.1), int(virtual_h * 0.1)),
            (int(virtual_w * 0.9), int(virtual_h * 0.1)),
            (int(virtual_w * 0.9), int(virtual_h * 0.9)),
            (int(virtual_w * 0.1), int(virtual_h * 0.9))
        ]

        compositor.setup_two_stage_homography(
            ref_left, ref_right, plane_corners, auto_method="sift"
        )

    # 获取扭曲后的针孔背景（左/右）
    pinhole_bg_left, pinhole_bg_right = compositor.generate_warped_backgrounds((virtual_w, virtual_h))

    cv2.imwrite(os.path.join(output_dir, "debug_pinhole_bg_left.jpg"), pinhole_bg_left)
    cv2.imwrite(os.path.join(output_dir, "debug_pinhole_bg_right.jpg"), pinhole_bg_right)
    print(f"   [OK] 针孔立体背景生成完成")

    # ===== 步骤4: 将针孔背景投影为鱼眼风格（Distortion）=====
    print("\n[步骤 4/6] 将针孔背景投影为鱼眼风格...")

    # 准备鱼眼畸变参数（假设D为0）
    D_zero = [0, 0, 0, 0]

    # 投影左背景
    fisheye_bg_left = convert_pinhole_to_fisheye(
        pinhole_bg_left,
        params_left['K'],
        params_left['xi'],
        D_zero,
        virtual_K,
        (H_fish, W_fish)
    )

    # 投影右背景
    fisheye_bg_right = convert_pinhole_to_fisheye(
        pinhole_bg_right,
        params_right['K'],
        params_right['xi'],
        D_zero,
        virtual_K,
        (H_fish, W_fish)
    )

    cv2.imwrite(os.path.join(output_dir, "debug_fisheye_bg_left.jpg"), fisheye_bg_left)
    cv2.imwrite(os.path.join(output_dir, "debug_fisheye_bg_right.jpg"), fisheye_bg_right)
    print(f"   [OK] 鱼眼背景投影完成")

    # ===== 步骤5: 最终合成（在鱼眼域）=====
    print("\n[步骤 5/6] 鱼眼域合成...")

    # 保存掩码调试信息
    print(f"   [DEBUG] 左掩码: min={mask_left.min()}, max={mask_left.max()}, "
          f"mean={mask_left.mean():.1f}")
    print(f"   [DEBUG] 右掩码: min={mask_right.min()}, max={mask_right.max()}, "
          f"mean={mask_right.mean():.1f}")

    # threshold=None 使用自动阈值（Otsu方法），确保前景完整保留
    final_left = composite_with_mask(fg_left, fisheye_bg_left, mask_left, threshold=None)
    final_right = composite_with_mask(fg_right, fisheye_bg_right, mask_right, threshold=None)

    print(f"   [OK] 合成完成")

    # ===== 步骤6: 保存结果 =====
    print("\n[步骤 6/6] 保存最终结果...")

    method_name = "depth" if use_depth_method else "homography"
    cv2.imwrite(os.path.join(output_dir, f"final_fisheye_left_{method_name}.jpg"), final_left)
    cv2.imwrite(os.path.join(output_dir, f"final_fisheye_right_{method_name}.jpg"), final_right)

    print("\n" + "=" * 80)
    print("[DONE] 鱼眼背景替换完成！")
    print("=" * 80)
    print(f"\n输出目录: {output_dir}/")
    print(f"  - final_fisheye_left_{method_name}.jpg   (最终左视角)")
    print(f"  - final_fisheye_right_{method_name}.jpg  (最终右视角)")
    print(f"  - depth_visualization.jpg       (深度可视化)")
    print(f"  - debug_pinhole_bg_*.jpg        (中间针孔背景)")
    print(f"  - debug_fisheye_bg_*.jpg        (中间鱼眼背景)")


def run_panorama_background_replacement(
    panorama_path: str,
    foreground_left_path: str,
    foreground_right_path: str,
    mask_left_path: str,
    mask_right_path: str,
    params_left: dict,
    params_right: dict,
    output_dir: str = "output"
):
    """
    方法3: 全景图背景替换流程

    使用 360° 全景图作为背景源，直接提取鱼眼视角
    适用场景：天空、远景、风景等无穷远背景

    Args:
        panorama_path: 全景图路径 (Equirectangular格式)
        foreground_left_path: 左相机前景图路径（鱼眼）
        foreground_right_path: 右相机前景图路径（鱼眼）
        mask_left_path: 左前景掩码路径（鱼眼）
        mask_right_path: 右前景掩码路径（鱼眼）
        params_left: 左相机鱼眼参数 {'K': [[...]], 'xi': float}
        params_right: 右相机鱼眼参数 {'K': [[...]], 'xi': float}
        output_dir: 输出目录
    """
    print("\n" + "=" * 80)
    print("方法3: 全景图提取 (Spherical Extraction)")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载图像
    print("\n[步骤 1/3] 加载图像...")
    pano_img = cv2.imread(panorama_path)
    fg_left = cv2.imread(foreground_left_path)
    fg_right = cv2.imread(foreground_right_path)
    mask_left = cv2.imread(mask_left_path, cv2.IMREAD_GRAYSCALE)
    mask_right = cv2.imread(mask_right_path, cv2.IMREAD_GRAYSCALE)

    if pano_img is None:
        raise FileNotFoundError(f"全景图未找到: {panorama_path}")
    if any(x is None for x in [fg_left, fg_right, mask_left, mask_right]):
        raise FileNotFoundError("部分输入图像未找到，请检查路径")

    H_fish, W_fish = fg_left.shape[:2]
    print(f"   [OK] 全景图尺寸: {pano_img.shape[1]}x{pano_img.shape[0]}")
    print(f"   [OK] 鱼眼前景尺寸: {W_fish}x{H_fish}")

    # 2. 提取背景 (分别提取左右眼，使用各自的相机参数)
    print("\n[步骤 2/3] 从全景图中提取鱼眼背景...")
    D_zero = [0, 0, 0, 0]

    # 左眼提取
    print("   提取左眼背景...")
    bg_fisheye_left = extract_fisheye_from_equirectangular(
        pano_img, params_left['K'], params_left['xi'], D_zero, (H_fish, W_fish)
    )

    # 右眼提取
    print("   提取右眼背景...")
    bg_fisheye_right = extract_fisheye_from_equirectangular(
        pano_img, params_right['K'], params_right['xi'], D_zero, (H_fish, W_fish)
    )

    # 保存调试图像
    cv2.imwrite(os.path.join(output_dir, "debug_pano_bg_left.jpg"), bg_fisheye_left)
    cv2.imwrite(os.path.join(output_dir, "debug_pano_bg_right.jpg"), bg_fisheye_right)
    print("   [OK] 鱼眼背景提取完成")

    # 保存掩码调试信息
    print(f"   [DEBUG] 左掩码: min={mask_left.min()}, max={mask_left.max()}, "
          f"mean={mask_left.mean():.1f}")
    print(f"   [DEBUG] 右掩码: min={mask_right.min()}, max={mask_right.max()}, "
          f"mean={mask_right.mean():.1f}")

    # 3. 合成
    print("\n[步骤 3/3] 合成并保存...")
    # threshold=None 使用自动阈值（Otsu方法），根据掩码灰度分布自动选择最佳阈值
    # 这样可以确保：
    # 1. 前景中的黑色物体（如黑色衣服、黑色背包）不会被误判为背景
    # 2. 只有掩码中标记为前景的区域才会从前景图中复制像素
    final_left = composite_with_mask(fg_left, bg_fisheye_left, mask_left, threshold=None)
    final_right = composite_with_mask(fg_right, bg_fisheye_right, mask_right, threshold=None)

    # 保存二值化后的掩码用于检查
    # 注意：这里的阈值是手动设置的用于调试，实际合成使用的是自动阈值
    _, binary_left = cv2.threshold(mask_left, 127, 255, cv2.THRESH_BINARY)
    _, binary_right = cv2.threshold(mask_right, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(output_dir, "debug_mask_binary_left.jpg"), binary_left)
    cv2.imwrite(os.path.join(output_dir, "debug_mask_binary_right.jpg"), binary_right)


    # 4. 保存
    cv2.imwrite(os.path.join(output_dir, "final_fisheye_left_pano.jpg"), final_left)
    cv2.imwrite(os.path.join(output_dir, "final_fisheye_right_pano.jpg"), final_right)

    print("\n" + "=" * 80)
    print("[DONE] 全景图方法完成！")
    print("=" * 80)
    print(f"\n输出目录: {output_dir}/")
    print(f"  - final_fisheye_left_pano.jpg   (最终左视角)")
    print(f"  - final_fisheye_right_pano.jpg  (最终右视角)")
    print(f"  - debug_pano_bg_*.jpg           (中间鱼眼背景)")


def run_with_test_data():
    """使用测试数据运行完整流程（鱼眼前景 + 针孔背景加畸变）"""

    print("\n" + "=" * 80)
    print("测试数据运行 - 鱼眼背景替换新工作流")
    print("=" * 80)

    # ==========================================
    # 1. 定义相机参数（真实鱼眼标定数据）
    # ==========================================
    params_left = {
        "K": [[1027.71, 0.00, 320.00],
              [0.00, 1027.71, 320.00],
              [0.00, 0.00, 1.00]],
        "xi": 0.976611
    }

    params_right = {
        "K": [[1099.93, 0.00, 320.00],
              [0.00, 1099.93, 320.00],
              [0.00, 0.00, 1.00]],
        "xi": 0.905052
    }

    # ==========================================
    # 2. 定义输入文件路径
    # ==========================================
    input_paths = {
        'bg': 'test_data/037.jpg',                    # 新背景（针孔）
        'pano': 'test_data/pano.png',                 # 全景图（可选，Equirectangular格式）
        'fg_left': 'test_data/segment/2_00402.png',   # 左前景（鱼眼）
        'fg_right': 'test_data/segment/4_00402.png',  # 右前景（鱼眼）
        'mask_left': 'test_data/mask/2_00402.png',    # 左掩码（鱼眼）
        'mask_right': 'test_data/mask/4_00402.png',   # 右掩码（鱼眼）
        # 可选：两阶段方法需要的参考图
        'ref_left': 'test_data/images/2_00402.jpg',
        'ref_right': 'test_data/images/4_00402.jpg',
    }

    # 检查文件是否存在
    for key, path in input_paths.items():
        if not os.path.exists(path):
            print(f"[WARNING] 文件不存在: {key} = {path}")

    # ==========================================
    # 3. 运行方法1: 深度引导方法
    # ==========================================
    print("\n" + "=" * 40)
    print("方法1: 深度引导 + 鱼眼投影")
    print("=" * 40)

    run_fisheye_background_replacement(
        background_path=input_paths['bg'],
        foreground_left_path=input_paths['fg_left'],
        foreground_right_path=input_paths['fg_right'],
        mask_left_path=input_paths['mask_left'],
        mask_right_path=input_paths['mask_right'],
        params_left=params_left,
        params_right=params_right,
        output_dir="output",
        use_depth_method=True,
        depth_params={
            'hfov_deg': 90.0,           # 虚拟针孔相机视场角
            'baseline': 0.065,          # 左右眼基线距离（米）
            'rotation_y_deg': 5.0,      # Y轴旋转
            'virtual_focal_length': None  # None则自动计算
        }
    )

    # ==========================================
    # 4. 运行方法2: 两阶段单应方法（可选）
    # ==========================================
    if os.path.exists(input_paths['ref_left']) and os.path.exists(input_paths['ref_right']):
        print("\n" + "=" * 40)
        print("方法2: 两阶段单应 + 鱼眼投影")
        print("=" * 40)

        run_fisheye_background_replacement(
            background_path=input_paths['bg'],
            foreground_left_path=input_paths['fg_left'],
            foreground_right_path=input_paths['fg_right'],
            mask_left_path=input_paths['mask_left'],
            mask_right_path=input_paths['mask_right'],
            params_left=params_left,
            params_right=params_right,
            reference_left_path=input_paths['ref_left'],
            reference_right_path=input_paths['ref_right'],
            output_dir="output",
            use_depth_method=False
        )
    else:
        print("\n[INFO] 跳过两阶段方法（未找到参考图像）")

    # ==========================================
    # 5. 运行方法3: 全景图提取方法（可选）
    # ==========================================
    if os.path.exists(input_paths.get('pano', '')):
        print("\n" + "=" * 40)
        print("方法3: 全景图提取 + 鱼眼投影")
        print("=" * 40)

        run_panorama_background_replacement(
            panorama_path=input_paths['pano'],
            foreground_left_path=input_paths['fg_left'],
            foreground_right_path=input_paths['fg_right'],
            mask_left_path=input_paths['mask_left'],
            mask_right_path=input_paths['mask_right'],
            params_left=params_left,
            params_right=params_right,
            output_dir="output"
        )
    else:
        print("\n[INFO] 跳过全景图方法（未找到全景图文件 test_data/pano.png）")
        print("      如需使用全景图方法，请准备 Equirectangular 格式的全景图")

    print("\n" + "=" * 80)
    print("[SUCCESS] 所有处理完成！")
    print("=" * 80)
    print("\n请查看 output/ 目录:")
    print("  - final_fisheye_left_*.jpg   (最终鱼眼左视角)")
    print("  - final_fisheye_right_*.jpg  (最终鱼眼右视角)")
    print("  - depth_visualization.jpg    (背景深度图)")
    print("  - debug_*.jpg                (中间调试图)")


if __name__ == "__main__":
    # 运行测试数据的完整流程
    run_with_test_data()
