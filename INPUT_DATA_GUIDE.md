# 输入数据要求说明

运行背景变换代码需要准备以下输入数据。根据使用的方法不同，所需数据也不同。

---

## 📋 必需的输入数据

### 🎨 1. 背景图（必需）
- **文件名**: `background.jpg` (或任意图像格式)
- **说明**: 你想要替换到场景中的新背景图
- **格式**: JPG, PNG, BMP等常见图像格式
- **尺寸**: 任意，建议至少1920×1080
- **示例**: 风景照片、虚拟场景、纯色背景等

### 👤 2. 前景图像（必需）
需要左右两张：
- **文件名**: `foreground_left.jpg`, `foreground_right.jpg`
- **说明**: 包含前景主体（人物、物体）的原始图像
- **格式**: JPG, PNG等
- **尺寸**: 需要相同（例如都是1920×1080）
- **要求**: 左右相机拍摄的对称视角图像

### 🎭 3. 前景掩码（必需，用于合成）
需要左右两张：
- **文件名**: `mask_left.png`, `mask_right.png`
- **说明**: 标记前景区域的二值掩码
- **格式**: PNG或灰度图
- **像素值**: 
  - **255 (白色)** = 前景（保留区域）
  - **0 (黑色)** = 背景（替换区域）
- **尺寸**: 必须与对应的前景图像完全一致
- **获取方法**:
  - 使用Photoshop、GIMP手动抠图
  - 使用removebg.com等在线工具
  - 使用深度学习分割模型（如Segment Anything）
  - 绿幕/蓝幕拍摄后色度键抠像

---

## 🔧 方法A：单应变换 + 手动标注点

### 额外需要：无
手动在代码中定义对应点坐标即可。

#### 📍 如何确定对应点：
1. 在**背景图**中选择4个以上特征点（建议选择角点或明显特征）
2. 在**目标视角图**中找到对应的位置
3. 记录坐标 `(x, y)`，像素坐标从左上角(0,0)开始

#### 示例标注：
```python
# 背景图中的点
background_points_left = [
    (100, 100),   # 背景图的左上角某处
    (900, 100),   # 背景图的右上角某处
    (900, 700),   # 背景图的右下角某处
    (100, 700)    # 背景图的左下角某处
]

# 左相机视角中希望这些点出现的位置
left_points = [
    (50, 80),     # 对应第一个点
    (950, 120),   # 对应第二个点
    (920, 680),   # 对应第三个点
    (80, 720)     # 对应第四个点
]
```

**工具推荐**: 使用图像查看器（如IrfanView）或在线工具查看鼠标坐标

---

## 🔧 方法B：单应变换 + 自动特征匹配

### 额外需要：参考图像
- **文件名**: `reference_left.jpg`, `reference_right.jpg`
- **说明**: 原始场景中左右相机拍摄的背景参考图（不含前景或仅含背景）
- **格式**: JPG, PNG等
- **尺寸**: 与目标输出尺寸一致
- **用途**: 系统自动提取特征点并与新背景图匹配

**适用场景**: 
- 有原始背景图像可用
- 背景纹理丰富（便于特征匹配）
- 不想手动标注对应点

---

## 🔧 方法C：深度引导重投影

### 额外需要：深度图
- **文件名**: `depth_map.png` 或 `depth_map.exr`
- **说明**: 背景图对应的深度信息
- **格式**: 
  - 16位PNG（推荐）
  - 32位EXR（高精度）
  - 8位PNG（精度较低）
- **尺寸**: 必须与背景图完全一致
- **深度值**: 实际物理距离（米），例如：
  - 近处物体: 0.5-2米
  - 中景: 2-10米
  - 远景: 10-100米
- **获取方法**:
  - **MiDaS**: 单目深度估计模型（免费）
    ```bash
    git clone https://github.com/isl-org/MiDaS
    python run.py --model_type dpt_large --input background.jpg
    ```
  - **DepthAnything**: 最新的深度估计模型
  - **LeReS**: 高精度深度估计
  - **ZoeDepth**: 度量深度估计
  - **真实深度相机**: 使用RealSense、Kinect等设备采集

#### 深度图格式转换：
```python
import cv2
import numpy as np

# 读取MiDaS输出（通常是归一化的）
depth = cv2.imread("depth_midas.png", cv2.IMREAD_ANYDEPTH)

# 转换为实际物理距离（米）
# 假设场景深度范围为0.5米到20米
depth_meters = 0.5 + (depth.astype(float) / depth.max()) * 19.5

# 保存为16位PNG
depth_uint16 = (depth_meters * 1000).astype(np.uint16)  # 转为毫米
cv2.imwrite("depth_metric.png", depth_uint16)
```

### 额外参数（需要估算或测量）：
- **baseline**: 左右相机基线距离（米），典型值：0.06-0.10米
- **rotation_y_deg**: 左右相机对称旋转角度（度），典型值：3-7度
- **hfov_deg**: 水平视场角（度），典型值：60-80度

---

## 📂 完整输入文件清单

### 最小配置（方法A：手动点）
```
input/
├── background.jpg           # 新背景图
├── foreground_left.jpg      # 左相机前景
├── foreground_right.jpg     # 右相机前景
├── mask_left.png           # 左前景掩码
└── mask_right.png          # 右前景掩码
```

### 自动匹配配置（方法B）
```
input/
├── background.jpg           # 新背景图
├── reference_left.jpg       # 原始左视角背景
├── reference_right.jpg      # 原始右视角背景
├── foreground_left.jpg      # 左相机前景
├── foreground_right.jpg     # 右相机前景
├── mask_left.png           # 左前景掩码
└── mask_right.png          # 右前景掩码
```

### 深度重投影配置（方法C）
```
input/
├── background.jpg           # 新背景图
├── depth_map.png           # 背景深度图
├── foreground_left.jpg      # 左相机前景
├── foreground_right.jpg     # 右相机前景
├── mask_left.png           # 左前景掩码
└── mask_right.png          # 右前景掩码
```

---

## 🛠️ 快速测试：生成模拟数据

如果暂时没有真实数据，可以用以下脚本生成测试数据：

```python
import cv2
import numpy as np

# 1. 生成测试背景图
background = np.random.randint(100, 200, (1080, 1920, 3), dtype=np.uint8)
cv2.rectangle(background, (400, 300), (1500, 800), (50, 150, 250), -1)
cv2.imwrite("test_background.jpg", background)

# 2. 生成测试前景图
foreground_left = np.random.randint(50, 150, (1080, 1920, 3), dtype=np.uint8)
cv2.circle(foreground_left, (960, 540), 200, (255, 100, 100), -1)
cv2.imwrite("test_foreground_left.jpg", foreground_left)

foreground_right = foreground_left.copy()
foreground_right = np.roll(foreground_right, 50, axis=1)  # 模拟视差
cv2.imwrite("test_foreground_right.jpg", foreground_right)

# 3. 生成测试掩码
mask = np.zeros((1080, 1920), dtype=np.uint8)
cv2.circle(mask, (960, 540), 200, 255, -1)
cv2.imwrite("test_mask_left.png", mask)

mask_right = np.roll(mask, 50, axis=1)
cv2.imwrite("test_mask_right.png", mask_right)

# 4. 生成测试深度图（可选）
depth = np.ones((1080, 1920), dtype=np.float32) * 5.0  # 5米
depth[300:800, 400:1500] = 2.0  # 近景2米
depth_uint16 = (depth * 1000).astype(np.uint16)
cv2.imwrite("test_depth.png", depth_uint16)

print("测试数据已生成！")
```

---

## ⚠️ 常见问题

### Q1: 没有前景掩码怎么办？
**A**: 有几种方法：
1. 使用在线抠图工具：removebg.com, photoroom.com
2. 使用Python库：
   ```bash
   pip install rembg
   python -c "from rembg import remove; from PIL import Image; Image.open('fg.jpg').save('mask.png')"
   ```
3. 如果是绿幕拍摄，使用色度键：
   ```python
   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   mask = cv2.inRange(hsv, (40,40,40), (80,255,255))
   mask = 255 - mask  # 反转
   ```

### Q2: 没有深度图怎么办？
**A**: 
1. 使用单应变换方法（方法A或B）替代
2. 使用MiDaS生成深度图（见上文）
3. 如果背景是平面，深度方法优势不大

### Q3: 左右图像尺寸不一致？
**A**: 先统一尺寸：
```python
img_right = cv2.resize(img_right, (img_left.shape[1], img_left.shape[0]))
```

### Q4: 如何验证输入数据正确？
**A**: 运行验证脚本：
```python
import cv2

# 检查文件是否存在且可读
files = ["background.jpg", "foreground_left.jpg", "mask_left.png"]
for f in files:
    img = cv2.imread(f)
    if img is None:
        print(f"❌ {f} 读取失败")
    else:
        print(f"✅ {f} OK - 尺寸: {img.shape}")

# 检查掩码格式
mask = cv2.imread("mask_left.png", 0)
print(f"掩码取值范围: {mask.min()} - {mask.max()}")
print(f"前景像素数: {np.sum(mask > 128)}")
```

---

## 📖 推荐工作流程

1. **准备阶段**：
   - 收集或拍摄左右视角前景图
   - 选择或创建新背景图
   - 生成前景掩码（抠图）

2. **选择方法**：
   - 背景简单/平面 → 方法A（手动点）
   - 有原始背景图 → 方法B（自动匹配）
   - 背景复杂/3D → 方法C（深度图）

3. **测试运行**：
   - 先用 `example_without_foreground_mask()` 测试背景扭曲
   - 检查结果质量
   - 调整参数（视场角、基线、旋转角）

4. **完整合成**：
   - 添加前景掩码
   - 调整羽化半径
   - 可选颜色匹配

详细代码示例见 `example_usage.py`！

