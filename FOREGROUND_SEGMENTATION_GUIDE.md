# 前景分割脚本使用指南

## 功能说明

`foreground_segmenter.py` 脚本用于根据掩码图从原始图像中分割出前景目标，支持以下功能：

- 单张图像处理
- 批量处理
- 自动扫描文件夹
- 多种输出格式（透明背景PNG、白色背景JPG、黑色背景JPG）

## 快速使用

### 运行脚本

```bash
python foreground_segmenter.py
```

运行后会提示选择处理模式：
- **模式1**: 单张图像处理（处理test_data中的示例）
- **模式2**: 批量处理test_data文件夹
- **模式3**: 批量处理mask文件夹（推荐）

### 模式3说明（推荐使用）

此模式会自动扫描 `mask/` 文件夹的结构：

```
mask/
├── 2_right/
│   ├── 00030.jpg          # 掩码图像
│   ├── 00116.jpg
│   ├── 00704.jpg
│   └── origin/
│       ├── 00030.jpg      # 原始图像
│       ├── 00116.jpg
│       └── 00704.jpg
└── 4_left/
    ├── 00030.jpg
    ├── 00116.jpg
    ├── 00704.jpg
    └── origin/
        ├── 00030.jpg
        ├── 00116.jpg
        └── 00704.jpg
```

脚本会自动匹配每个掩码文件与其对应的原始图像，并输出到 `output/segmented_mask_folder/` 目录。

## 代码示例

### 在其他脚本中使用

```python
from foreground_segmenter import ForegroundSegmenter

# 创建分割器
segmenter = ForegroundSegmenter()

# 方法1: 从文件分割
segmenter.segment_from_files(
    image_path="path/to/image.jpg",
    mask_path="path/to/mask.png",
    output_dir="output/results",
    save_formats=['png', 'white']  # 只保存透明和白底版本
)

# 方法2: 从内存中的数组分割
import cv2
image = cv2.imread("image.jpg")
mask = cv2.imread("mask.png")
fg_transparent, fg_white, fg_black = segmenter.segment_foreground(image, mask)

# 方法3: 批量处理
pairs = [
    ("img1.jpg", "mask1.png"),
    ("img2.jpg", "mask2.png"),
]
results = segmenter.batch_segment(pairs, output_dir="output/batch")
```

## 输出格式

脚本会生成三种格式的前景图像：

1. **透明背景PNG** (`*_foreground_transparent.png`)
   - BGRA格式，Alpha通道表示透明度
   - 适用于需要叠加到其他背景的场景
   - 文件较大但保留完整透明信息

2. **白色背景JPG** (`*_foreground_white.jpg`)
   - 前景保留，背景填充白色
   - 适用于预览和文档展示
   - 文件较小

3. **黑色背景JPG** (`*_foreground_black.jpg`)
   - 前景保留，背景填充黑色
   - 适用于检查分割效果
   - 文件较小

## 注意事项

1. **掩码格式要求**：
   - 白色（255）表示前景区域
   - 黑色（0）表示背景区域
   - 支持灰度图和彩色图

2. **尺寸匹配**：
   - 如果掩码尺寸与原图不匹配，脚本会自动调整掩码尺寸
   - 建议使用相同尺寸以获得最佳效果

3. **文件命名**：
   - 输出文件名基于原始图像文件名
   - 自动添加后缀区分不同格式

## 常见问题

### Q: 如何只输出透明背景的PNG？

修改 `save_formats` 参数：
```python
segmenter.segment_from_files(
    image_path="image.jpg",
    mask_path="mask.png",
    save_formats=['png']  # 只保存PNG格式
)
```

### Q: 如何自定义输出路径？

指定 `output_dir` 参数：
```python
segmenter.segment_from_files(
    image_path="image.jpg",
    mask_path="mask.png",
    output_dir="custom/output/path"
)
```

### Q: 批量处理时如何知道哪些成功了？

检查返回的结果列表：
```python
results = segmenter.batch_segment(pairs)
for idx, result in enumerate(results):
    if result['success']:
        print(f"第{idx+1}个成功，文件保存在: {result['paths']}")
    else:
        print(f"第{idx+1}个失败，错误: {result['error']}")
```

## 性能优化

- 对于大批量图像，考虑分批处理
- PNG格式文件较大，如果磁盘空间有限可以只保存JPG格式
- 处理高分辨率图像时可能需要较多内存

## 集成到现有流程

该脚本可以与现有的背景替换流程集成：

```
原始图像 + 掩码 
    ↓
foreground_segmenter.py (分割前景)
    ↓
分割后的前景图像
    ↓
background_compositor.py (合成到新背景)
    ↓
最终结果
```

