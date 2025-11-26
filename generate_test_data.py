# python
"""
快速生成测试数据
运行此脚本将在 test_data/ 目录下生成所有必需的测试文件
【更新】参考图和新背景使用完全不同的场景，支持两阶段匹配
"""
import cv2
import numpy as np
import os


def create_reference_scene(W, H):
    """生成参考场景（原始相机拍摄的背景）- 办公室风格"""
    ref = np.ones((H, W, 3), dtype=np.uint8) * 180  # 浅灰底

    # 模拟墙面纹理：砖墙效果
    brick_h, brick_w = 60, 120
    for i in range(0, H, brick_h):
        offset = (i // brick_h) % 2 * (brick_w // 2)
        for j in range(-offset, W, brick_w):
            x1, y1 = max(0, j), i
            x2, y2 = min(W, j + brick_w - 4), min(H, i + brick_h - 4)
            color = (160 + np.random.randint(-15, 15),
                    165 + np.random.randint(-15, 15),
                    170 + np.random.randint(-15, 15))
            cv2.rectangle(ref, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(ref, (x1, y1), (x2, y2), (120, 120, 120), 1)

    # 添加窗户
    window_w, window_h = 300, 400
    window_x, window_y = W - 450, 150
    cv2.rectangle(ref, (window_x, window_y),
                 (window_x + window_w, window_y + window_h), (200, 230, 250), -1)
    # 窗框
    cv2.rectangle(ref, (window_x, window_y),
                 (window_x + window_w, window_y + window_h), (80, 80, 80), 8)
    cv2.line(ref, (window_x + window_w//2, window_y),
            (window_x + window_w//2, window_y + window_h), (80, 80, 80), 6)
    cv2.line(ref, (window_x, window_y + window_h//2),
            (window_x + window_w, window_y + window_h//2), (80, 80, 80), 6)

    # 添加门框
    door_w, door_h = 180, 480
    door_x, door_y = 200, H - door_h - 50
    cv2.rectangle(ref, (door_x, door_y),
                 (door_x + door_w, door_y + door_h), (140, 120, 100), -1)
    cv2.rectangle(ref, (door_x, door_y),
                 (door_x + door_w, door_y + door_h), (60, 50, 40), 6)
    cv2.circle(ref, (door_x + door_w - 30, door_y + door_h//2), 12, (220, 200, 50), -1)

    # 标记文字
    cv2.putText(ref, "ORIGINAL SCENE", (W//2 - 280, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (60, 60, 60), 4)

    return ref


def create_new_background(W, H):
    """生成要替换的新背景 - 自然风景风格"""
    bg = np.zeros((H, W, 3), dtype=np.uint8)

    # 天空渐变（上蓝下白）
    for i in range(H // 2):
        t = i / (H // 2)
        bg[i, :, 0] = int(200 - 80 * t)   # B
        bg[i, :, 1] = int(180 - 50 * t)   # G
        bg[i, :, 2] = int(100 + 20 * t)   # R

    # 地面渐变（上浅下深）
    for i in range(H // 2, H):
        t = (i - H // 2) / (H // 2)
        bg[i, :, 0] = int(80 - 20 * t)    # B
        bg[i, :, 1] = int(140 - 30 * t)   # G
        bg[i, :, 2] = int(90 - 20 * t)    # R

    # 添加"山脉"轮廓
    mountain_pts = []
    for x in range(0, W, 50):
        y = H // 2 - 100 + int(50 * np.sin(x / 150.0)) - int(30 * np.cos(x / 80.0))
        mountain_pts.append([x, y])
    mountain_pts.append([W, H // 2 + 50])
    mountain_pts.append([0, H // 2 + 50])
    mountain_pts = np.array(mountain_pts, np.int32)
    cv2.fillPoly(bg, [mountain_pts], (60, 100, 80))

    # 添加"太阳"
    sun_center = (W - 250, 180)
    cv2.circle(bg, sun_center, 80, (100, 200, 255), -1)
    cv2.circle(bg, sun_center, 85, (150, 220, 255), 3)

    # 添加几何装饰物（帮助深度匹配）
    # 左侧树
    tree_x, tree_y = 300, H // 2 + 100
    cv2.rectangle(bg, (tree_x - 15, tree_y), (tree_x + 15, tree_y + 200), (40, 60, 35), -1)
    cv2.circle(bg, (tree_x, tree_y - 30), 80, (50, 120, 60), -1)

    # 右侧建筑物
    building_x = W * 2 // 3
    building_y = H // 2 + 50
    cv2.rectangle(bg, (building_x, building_y),
                 (building_x + 200, building_y + 300), (100, 100, 140), -1)
    # 窗户
    for wy in range(building_y + 40, building_y + 280, 60):
        for wx in range(building_x + 30, building_x + 180, 50):
            cv2.rectangle(bg, (wx, wy), (wx + 30, wy + 40), (200, 220, 100), -1)

    # 标记文字
    cv2.putText(bg, "NEW BACKGROUND", (W // 2 - 320, H - 50),
               cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 6)

    # 添加四个角点标记（用于平面标定）
    corners = [(100, 100), (W - 100, 100), (W - 100, H - 100), (100, H - 100)]
    for i, (x, y) in enumerate(corners):
        cv2.circle(bg, (x, y), 30, (255, 0, 0), -1)
        cv2.circle(bg, (x, y), 35, (255, 255, 255), 3)
        cv2.putText(bg, f"{i+1}", (x - 12, y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return bg


def create_depth_for_new_background(W, H):
    """生成与新背景对应的深度图"""
    # 基础深度：从上到下渐变（天空远，地面近）
    depth = np.zeros((H, W), dtype=np.float32)

    # 天空部分：10-15米
    for i in range(H // 2):
        t = i / (H // 2)
        depth[i, :] = 15.0 - 5.0 * t

    # 地面部分：5-8米
    for i in range(H // 2, H):
        t = (i - H // 2) / (H // 2)
        depth[i, :] = 8.0 - 3.0 * t

    # 山脉：远景12米
    mountain_mask = np.zeros((H, W), dtype=np.uint8)
    mountain_pts = []
    for x in range(0, W, 50):
        y = H // 2 - 100 + int(50 * np.sin(x / 150.0)) - int(30 * np.cos(x / 80.0))
        mountain_pts.append([x, y])
    mountain_pts.append([W, H // 2 + 50])
    mountain_pts.append([0, H // 2 + 50])
    mountain_pts = np.array(mountain_pts, np.int32)
    cv2.fillPoly(mountain_mask, [mountain_pts], 255)
    depth = np.where(mountain_mask > 0, 12.0, depth)

    # 太阳：最远20米
    sun_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(sun_mask, (W - 250, 180), 80, 255, -1)
    depth = np.where(sun_mask > 0, 20.0, depth)

    # 左侧树：近景3米
    tree_x, tree_y = 300, H // 2 + 100
    tree_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(tree_mask, (tree_x - 15, tree_y), (tree_x + 15, tree_y + 200), 255, -1)
    cv2.circle(tree_mask, (tree_x, tree_y - 30), 80, 255, -1)
    depth = np.where(tree_mask > 0, 3.0, depth)

    # 右侧建筑物：中景6米
    building_mask = np.zeros((H, W), dtype=np.uint8)
    building_x = W * 2 // 3
    building_y = H // 2 + 50
    cv2.rectangle(building_mask, (building_x, building_y),
                 (building_x + 200, building_y + 300), 255, -1)
    depth = np.where(building_mask > 0, 6.0, depth)

    # 平滑处理
    depth = cv2.GaussianBlur(depth, (31, 31), 8)
    depth = np.clip(depth, 2.5, 20.0)

    return depth


def add_camera_effects(img):
    """添加相机拍摄效果：轻微模糊和噪声"""
    out = cv2.GaussianBlur(img, (3, 3), 0.5)
    noise = np.random.normal(0, 5, img.shape).astype(np.float32)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def create_test_data():
    """生成完整的测试数据集"""
    output_dir = "test_data"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("生成测试数据（两阶段匹配版本）...")
    print("=" * 60)

    H, W = 1080, 1920

    # 1. 生成参考场景（原始背景 - 办公室风格）
    print("\n[1/8] 生成参考场景...")
    ref_scene = create_reference_scene(W, H)

    # 左视角：轻微透视变换
    M_left = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [W, 0], [W, H], [0, H]]),
        np.float32([[50, 30], [W - 50, 50], [W - 80, H - 30], [80, H - 50]])
    )
    ref_left = cv2.warpPerspective(ref_scene, M_left, (W, H))
    ref_left = add_camera_effects(ref_left)
    cv2.putText(ref_left, "REF LEFT", (50, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 100, 100), 5)
    ref_left_path = os.path.join(output_dir, "reference_left.jpg")
    cv2.imwrite(ref_left_path, ref_left)
    print(f"   ✓ 左参考图已保存: {ref_left_path}")

    # 右视角：另一个透视变换
    M_right = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [W, 0], [W, H], [0, H]]),
        np.float32([[80, 50], [W - 80, 30], [W - 50, H - 50], [50, H - 30]])
    )
    ref_right = cv2.warpPerspective(ref_scene, M_right, (W, H))
    ref_right = add_camera_effects(ref_right)
    cv2.putText(ref_right, "REF RIGHT", (W - 450, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 255), 5)
    ref_right_path = os.path.join(output_dir, "reference_right.jpg")
    cv2.imwrite(ref_right_path, ref_right)
    print(f"   ✓ 右参考图已保存: {ref_right_path}")

    # 2. 生成新背景（自然风景风格 - 完全不同）
    print("[2/8] 生成新背景图...")
    background = create_new_background(W, H)
    bg_path = os.path.join(output_dir, "background.jpg")
    cv2.imwrite(bg_path, background)
    print(f"   ✓ 新背景图已保存: {bg_path}")

    # 3. 生成左相机前景图
    print("[3/8] 生成左相机前景图...")
    foreground_left = np.ones((H, W, 3), dtype=np.uint8) * 80
    center_x, center_y = W // 2, H // 2
    cv2.circle(foreground_left, (center_x, center_y - 100), 80, (180, 150, 120), -1)
    cv2.rectangle(foreground_left, (center_x - 100, center_y),
                 (center_x + 100, center_y + 250), (100, 100, 200), -1)
    cv2.putText(foreground_left, "LEFT", (center_x - 60, center_y + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    fg_left_path = os.path.join(output_dir, "foreground_left.jpg")
    cv2.imwrite(fg_left_path, foreground_left)
    print(f"   ✓ 左前景图已保存: {fg_left_path}")

    # 4. 生成右相机前景图
    print("[4/8] 生成右相机前景图...")
    foreground_right = np.roll(foreground_left, 50, axis=1)
    cv2.rectangle(foreground_right, (center_x - 60 + 50, center_y + 20),
                 (center_x + 90 + 50, center_y + 80), (100, 100, 200), -1)
    cv2.putText(foreground_right, "RIGHT", (center_x - 70 + 50, center_y + 60),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    fg_right_path = os.path.join(output_dir, "foreground_right.jpg")
    cv2.imwrite(fg_right_path, foreground_right)
    print(f"   ✓ 右前景图已保存: {fg_right_path}")

    # 5. 生成左前景掩码
    print("[5/8] 生成左前景掩码...")
    mask_left = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask_left, (center_x, center_y - 100), 80, 255, -1)
    cv2.rectangle(mask_left, (center_x - 100, center_y),
                 (center_x + 100, center_y + 250), 255, -1)
    mask_left = cv2.GaussianBlur(mask_left, (15, 15), 5)
    mask_left_path = os.path.join(output_dir, "mask_left.png")
    cv2.imwrite(mask_left_path, mask_left)
    print(f"   ✓ 左掩码已保存: {mask_left_path}")

    # 6. 生成右前景掩码
    print("[6/8] 生成右前景掩码...")
    mask_right = np.roll(mask_left, 50, axis=1)
    mask_right_path = os.path.join(output_dir, "mask_right.png")
    cv2.imwrite(mask_right_path, mask_right)
    print(f"   ✓ 右掩码已保存: {mask_right_path}")

    # 7. 生成深度图（与新背景对应）
    print("[7/8] 生成深度图（与新背景对应）...")
    depth_map = create_depth_for_new_background(W, H)

    depth_uint16 = (depth_map * 1000.0).astype(np.uint16)
    depth_path = os.path.join(output_dir, "depth_map.png")
    cv2.imwrite(depth_path, depth_uint16)
    print(f"   ✓ 深度图已保存: {depth_path}")

    # 深度可视化
    dmin, dmax = float(depth_map.min()), float(depth_map.max())
    depth_vis = ((depth_map - dmin) / max(1e-6, (dmax - dmin)) * 255.0).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    depth_vis_path = os.path.join(output_dir, "depth_map_visualization.jpg")
    cv2.imwrite(depth_vis_path, depth_vis)
    print(f"   ✓ 深度可视化已保存: {depth_vis_path}")

    # 深度叠加在背景上
    overlay = cv2.addWeighted(background, 0.6, depth_vis, 0.4, 0)
    overlay_path = os.path.join(output_dir, "depth_on_background.jpg")
    cv2.imwrite(overlay_path, overlay)
    print(f"   ✓ 深度叠加可视化已保存: {overlay_path}")

    # 8. 生成配置文件
    print("[8/8] 生成配置信息...")
    config_path = os.path.join(output_dir, "test_config.txt")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("测试数据配置信息（两阶段匹配版本）\n")
        f.write("=" * 60 + "\n\n")

        f.write("=== 场景说明 ===\n")
        f.write("参考场景（reference_*.jpg）: 办公室风格（墙壁/窗户/门）\n")
        f.write("新背景（background.jpg）: 自然风景风格（天空/山/树/建筑）\n")
        f.write("两者内容完全不同，需要两阶段匹配\n\n")

        f.write(f"图像尺寸: {W} x {H}\n\n")

        f.write("=== 方法A：手动标注点 ===\n")
        f.write("不适用（参考图与背景内容不同）\n\n")

        f.write("=== 方法B：两阶段自动匹配 ===\n")
        f.write("阶段1：从 reference_left.jpg ↔ reference_right.jpg 学习相机几何\n")
        f.write("阶段2：将 background.jpg 作为平面映射到左右视角\n")
        f.write("需要提供：背景平面在左视角中的四个角点\n")
        f.write("建议角点（对应背景图的四角标记）:\n")
        f.write(f"  左上: (150, 150)\n")
        f.write(f"  右上: ({W - 150}, 150)\n")
        f.write(f"  右下: ({W - 150}, {H - 150})\n")
        f.write(f"  左下: (150, {H - 150})\n\n")

        f.write("=== 方法C：深度重投影 ===\n")
        f.write(f"深度范围: {dmin:.2f}m - {dmax:.2f}m\n")
        f.write("深度分布:\n")
        f.write("  - 天空: 10-15m\n")
        f.write("  - 山脉: 12m\n")
        f.write("  - 太阳: 20m (最远)\n")
        f.write("  - 地面: 5-8m\n")
        f.write("  - 左侧树: 3m (最近)\n")
        f.write("  - 右侧建筑: 6m\n\n")

        f.write("=== 建议参数 ===\n")
        f.write("hfov_deg: 70.0\n")
        f.write("baseline: 0.065 (米)\n")
        f.write("rotation_y_deg: 5.0 (度)\n")

    print(f"   ✓ 配置信息已保存: {config_path}")

    print("\n" + "=" * 60)
    print("✅ 测试数据生成完成！")
    print("=" * 60)
    print(f"\n所有文件已保存到: {os.path.abspath(output_dir)}\n")
    print("文件说明:")
    print("  1. reference_left.jpg   - 左参考图（办公室场景）")
    print("  2. reference_right.jpg  - 右参考图（办公室场景）")
    print("  3. background.jpg       - 新背景（自然风景，完全不同）")
    print("  4-6. foreground_*.jpg, mask_*.png - 前景和掩码")
    print("  7-9. depth_*.png/jpg    - 深度图（与新背景对应）")
    print("  10. test_config.txt     - 配置信息\n")
    print("现在可以运行 test_run.py 进行测试！")


if __name__ == "__main__":
    create_test_data()
