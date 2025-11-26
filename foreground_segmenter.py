"""
前景分割脚本
功能：根据掩码图从原图中分割出前景目标
支持单张图像处理和批量处理
"""
import cv2
import numpy as np
import os
from pathlib import Path


class ForegroundSegmenter:
    """前景分割器"""

    def __init__(self):
        """初始化分割器"""
        pass

    def segment_foreground(self, image_bgr, mask):
        """
        从图像中分割前景

        参数:
            image_bgr: 原始BGR图像 (numpy array)
            mask: 掩码图像，可以是单通道或三通道 (numpy array)
                  白色(255)表示前景，黑色(0)表示背景

        返回:
            foreground_bgr: 分割出的前景图像，背景为透明(BGRA格式)
            foreground_rgb: 分割出的前景图像，背景为白色(BGR格式，便于预览)
        """
        # 确保图像和掩码尺寸一致
        if image_bgr.shape[:2] != mask.shape[:2]:
            print(f"[警告] 图像尺寸 {image_bgr.shape[:2]} 与掩码尺寸 {mask.shape[:2]} 不匹配，正在调整掩码...")
            mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]))

        # 转换掩码为单通道灰度图
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask.copy()

        # 自适应二值化：根据掩码的实际像素值范围选择阈值
        max_val = np.max(mask_gray)
        min_val = np.min(mask_gray)

        # 如果掩码的最大值远小于255，使用自适应阈值
        if max_val < 200:
            # 使用中间值作为阈值
            threshold = (max_val + min_val) / 2
            print(f"[调试] 掩码像素范围: {min_val}-{max_val}, 使用自适应阈值: {threshold:.1f}")
        else:
            threshold = 127

        _, mask_binary = cv2.threshold(mask_gray, threshold, 255, cv2.THRESH_BINARY)

        # --- 新增：填充轮廓以确保掩码是实心的 ---
        # 寻找轮廓
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建一个黑色的背景，并在上面绘制并填充轮廓
        filled_mask = np.zeros_like(mask_binary)
        cv2.drawContours(filled_mask, contours, -1, (255), thickness=cv2.FILLED)
        # -----------------------------------------

        # 方法1: 创建带Alpha通道的前景（透明背景）
        foreground_bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
        foreground_bgra[:, :, 3] = filled_mask  # 设置Alpha通道

        # 方法2: 创建白色背景的前景（便于预览）
        foreground_white = image_bgr.copy()
        background_pixels = filled_mask == 0
        foreground_white[background_pixels] = [255, 255, 255]  # 背景设为白色

        # 方法3: 只保留前景区域（其他区域为黑色）
        foreground_black = cv2.bitwise_and(image_bgr, image_bgr, mask=filled_mask)

        return foreground_bgra, foreground_white, foreground_black

    def segment_from_files(self, image_path, mask_path, output_dir=None,
                           save_formats=['png', 'white', 'black']):
        """
        从文件路径读取图像和掩码，进行分割并保存
        """
        # ────────────── 1. 读图 ──────────────
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path))

        if image is None:
            raise FileNotFoundError(f"❌ 无法读取图像: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"❌ 无法读取掩码: {mask_path}")

        # ────────────── 2. 执行分割 ──────────────
        fg_trans, fg_white, fg_black = self.segment_foreground(image, mask)

        # ────────────── 3. 检查结果 ──────────────
        # 检查Alpha通道，如果全透明则发出警告
        if len(fg_trans.shape) == 3 and fg_trans.shape[2] == 4:
            alpha_channel = fg_trans[:, :, 3]
            if np.count_nonzero(alpha_channel) == 0:
                print("⚠️  警告：分割后的前景是完全透明的！请检查掩码文件内容。")
                print(f"   - 掩码路径: {mask_path}")
                print(f"   - 掩码原始唯一值: {np.unique(mask)}")

        # ────────────── 4. 保存文件 ──────────────
        saved_paths = {}
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        os.makedirs(output_dir, exist_ok=True)

        image_name = Path(image_path).stem
        if 'png' in save_formats:
            p = os.path.join(output_dir, f"{image_name}_foreground_transparent.png")
            cv2.imwrite(p, fg_trans)
            saved_paths['transparent'] = p
            print(f"[保存] 透明背景 : {p}")
        if 'white' in save_formats:
            p = os.path.join(output_dir, f"{image_name}_foreground_white.jpg")
            cv2.imwrite(p, fg_white)
            saved_paths['white'] = p
            print(f"[保存] 白色背景 : {p}")
        if 'black' in save_formats:
            p = os.path.join(output_dir, f"{image_name}_foreground_black.jpg")
            cv2.imwrite(p, fg_black)
            saved_paths['black'] = p
            print(f"[保存] 黑色背景 : {p}")

        return saved_paths

    def batch_segment(self, image_mask_pairs, output_dir='output/segmented'):
        """
        批量处理多张图像

        参数:
            image_mask_pairs: 图像和掩码路径对的列表
                             格式: [(image_path1, mask_path1), (image_path2, mask_path2), ...]
            output_dir: 输出目录

        返回:
            results: 处理结果列表
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []

        for idx, (img_path, mask_path) in enumerate(image_mask_pairs):
            print(f"\n[处理 {idx+1}/{len(image_mask_pairs)}] {img_path}")
            try:
                saved = self.segment_from_files(img_path, mask_path, output_dir)
                results.append({'success': True, 'paths': saved})
            except Exception as e:
                print(f"[错误] 处理失败: {e}")
                results.append({'success': False, 'error': str(e)})

        return results


def process_single_image():
    """单张图像处理示例"""
    print("=" * 60)
    print("前景分割脚本 - 单图像模式")
    print("=" * 60)

    # 配置输入输出路径
    image_path = "test_data/2_00402.jpg"  # 原始图像
    mask_path = "test_data/mask2_00402.png"  # 掩码图像
    output_dir = "output/segmented"

    # 创建分割器
    segmenter = ForegroundSegmenter()

    # 执行分割
    try:
        saved_paths = segmenter.segment_from_files(
            image_path=image_path,
            mask_path=mask_path,
            output_dir=output_dir,
            save_formats=['png', 'white', 'black']
        )

        print("\n" + "=" * 60)
        print("分割完成！已保存以下文件:")
        for format_type, path in saved_paths.items():
            print(f"  - {format_type}: {path}")
        print("=" * 60)

    except Exception as e:
        print(f"\n[错误] 分割失败: {e}")


def process_batch():
    """批量处理示例"""
    print("=" * 60)
    print("前景分割脚本 - 批量处理模式")
    print("=" * 60)

    # 定义图像和掩码对
    pairs = [
        ("test_data/2_00402.jpg", "test_data/mask2_00402.png"),
        ("test_data/4_00402.jpg", "test_data/mask4_00402.png"),
    ]

    # 创建分割器
    segmenter = ForegroundSegmenter()

    # 批量处理
    results = segmenter.batch_segment(pairs, output_dir="output/segmented_batch")

    # 统计结果
    success_count = sum(1 for r in results if r['success'])
    print(f"\n批量处理完成: {success_count}/{len(results)} 成功")


def process_mask_folder():
    """处理mask文件夹中的所有图像"""
    print("=" * 60)
    print("前景分割脚本 - 文件夹批量处理模式")
    print("=" * 60)

    # 扫描mask文件夹
    mask_base_dir = "mask"
    output_dir = "output/segmented_mask_folder"

    segmenter = ForegroundSegmenter()
    pairs = []

    # 遍历mask文件夹中的子文件夹
    for camera_folder in ["2_right", "4_left"]:
        camera_path = os.path.join(mask_base_dir, camera_folder)
        if not os.path.exists(camera_path):
            continue

        # 遍历该相机文件夹中的掩码文件
        mask_files = [f for f in os.listdir(camera_path)
                     if f.endswith(('.jpg', '.png')) and 'origin' not in f]

        for mask_file in mask_files:
            mask_path = os.path.join(camera_path, mask_file)

            # 尝试在origin子文件夹中找到对应的原始图像
            origin_path = os.path.join(camera_path, "origin", mask_file)

            if os.path.exists(origin_path):
                pairs.append((origin_path, mask_path))
                print(f"[发现] {origin_path} + {mask_path}")
            else:
                print(f"[跳过] 未找到原始图像: {origin_path}")

    if pairs:
        print(f"\n共找到 {len(pairs)} 对图像-掩码，开始处理...\n")
        results = segmenter.batch_segment(pairs, output_dir=output_dir)
        success_count = sum(1 for r in results if r['success'])
        print(f"\n批量处理完成: {success_count}/{len(results)} 成功")
    else:
        print("\n未找到任何图像-掩码对")


if __name__ == "__main__":
    # 提供三种使用方式，可以根据需要选择

    # 方式1: 处理单张图像
    print("\n选择处理模式:")
    print("1. 单张图像处理")
    print("2. 批量处理（test_data）")
    print("3. 批量处理（mask文件夹）")

    mode = input("\n请输入模式编号 (1/2/3, 默认为1): ").strip()

    if mode == "2":
        process_batch()
    elif mode == "3":
        process_mask_folder()
    else:
        process_single_image()

