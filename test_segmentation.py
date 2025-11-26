#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前景用原图、背景涂黑 —— 单张输出
Usage:
    python fg_orig_bg_black.py \
        --img_dir  data/images \
        --mask_dir data/masks \
        --out_root output/fg_orig_bg_black
"""
import argparse
import cv2
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Foreground=original, Background=black")
    parser.add_argument("--img_dir",  default='test_data/images', type=Path)
    parser.add_argument("--mask_dir", default='test_data/mask', type=Path)
    parser.add_argument("--out_root", default="output/fg_orig_bg_black", type=Path)
    parser.add_argument("--ext", default=".png")
    return parser.parse_args()

def check_dirs(*dirs):
    for d in dirs:
        if not d.is_dir():
            print(f"[ERROR] Directory not exists: {d}")
            sys.exit(1)

def get_pairs(img_dir, mask_dir, ext):
    img_dict  = {p.stem: p for p in img_dir.iterdir()
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}
    mask_dict = {p.stem: p for p in mask_dir.iterdir()
                 if p.suffix.lower() == ext.lower()}
    common = sorted(set(img_dict) & set(mask_dict))
    if not common:
        print("[ERROR] No matched pairs found")
        sys.exit(1)
    print(f"[INFO] Found {len(common)} pairs")
    return [(img_dict[k], mask_dict[k]) for k in common]

def fg_orig_bg_black(image, mask):
    """
    image: (H,W,3)  BGR
    mask:  (H,W)    0=背景, 1~4=前景
    return: (H,W,3) 背景全黑，前景原图
    """
    fg_mask = (mask != 0).astype(np.uint8)          # 0/1
    fg_mask_3c = np.stack([fg_mask]*3, axis=-1)     # (H,W,3)
    result = image * fg_mask_3c                     # 广播乘法
    return result

def main():
    args = parse_args()
    check_dirs(args.img_dir, args.mask_dir)
    pairs = get_pairs(args.img_dir, args.mask_dir, args.ext)
    args.out_root.mkdir(parents=True, exist_ok=True)

    for img_path, mask_path in tqdm(pairs, desc="Processing"):
        image = cv2.imread(str(img_path))
        mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            print(f"[WARN] Read error, skip: {img_path} | {mask_path}")
            continue
        if image.shape[:2] != mask.shape:
            print(f"[WARN] Shape mismatch, skip: {img_path}")
            continue
        out_img = fg_orig_bg_black(image, mask)
        out_path = args.out_root / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), out_img)

    print("[INFO] All done, results saved in:", args.out_root.resolve())

if __name__ == "__main__":
    main()