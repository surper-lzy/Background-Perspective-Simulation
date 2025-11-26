# 背景变换代码使用指南

## ⭐ 新功能：ZoeDepth 深度估计集成

现在支持使用 **ZoeDepth** 自动估计背景深度，无需手动提供深度图！

### 🎯 端到端运行（推荐）

一键运行完整流程，包含自动深度估计：

```bash
python run_with_zoedepth.py
```

这将自动完成：
1. ✅ ZoeDepth 深度估计
2. ✅ 深度引导背景扭曲
3. ✅ 前景背景合成
4. ✅ 生成左右视角结果

**详细说明**：见 `ZOEDEPTH_INTEGRATION_GUIDE.md`

---

## 🚀 快速开始（传统方法）

### 第一步：生成测试数据
```bash
python generate_test_data.py
```
这将在 `test_data/` 目录下生成所有必需的测试文件（背景图、前景图、掩码、深度图等）。

### 第二步：运行测试
```bash
python test_run.py
```
这将自动运行所有三种方法并生成结果到 `output/` 目录。

### 第三步：查看结果
检查 `output/` 目录中的图像：
- `warped_bg_*.jpg` - 仅背景扭曲效果
- `two_stage_*.jpg` - 两阶段自动匹配结果
- `depth_*.jpg` - 深度重投影结果

---

## 项目结构

```
C:\Users\lzy\Desktop\Test\
├── depth_estimator.py            # ⭐ ZoeDepth 深度估计模块（新）
├── run_with_zoedepth.py          # ⭐ 端到端运行脚本（新）
├── intrinsics_estimator.py      # 相机内参估计工具
├── homography_warper.py          # 单应变换背景扭曲
├── depth_warper.py               # 深度引导重投影
├── background_compositor.py      # 背景合成主类
├── example_usage.py              # 使用示例
├── requirements.txt              # 依赖包列表
└── background_warping.md         # 原始方案文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 基本使用流程

#### 方法A：单应变换 + 手动标注点

```python
from background_compositor import BackgroundCompositor, WarpMethod

# 创建合成器
compositor = BackgroundCompositor(method=WarpMethod.HOMOGRAPHY)

# 加载背景
compositor.load_background("background.jpg")

# 定义对应点
background_points_left = [(100, 100), (900, 100), (900, 700), (100, 700)]
left_points = [(50, 80), (950, 120), (920, 680), (80, 720)]

# 设置单应变换
compositor.setup_homography_method(
    background_points_left=background_points_left,
    background_points_right=background_points_right,
    left_points=left_points,
    right_points=right_points
)

# 加载前景和掩码
compositor.load_foreground_masks(
    mask_left_path="mask_left.png",
    mask_right_path="mask_right.png"
)

# 处理并保存
result_left, result_right = compositor.process_stereo_pair(
    foreground_left, foreground_right, (1920, 1080)
)
```

#### 方法B：单应变换 + 自动特征匹配

```python
compositor = BackgroundCompositor(method=WarpMethod.HOMOGRAPHY)
compositor.load_background("background.jpg")

# 使用SIFT自动匹配
compositor.setup_homography_method(
    left_reference_img=left_reference,
    right_reference_img=right_reference,
    auto_method="sift"  # 或 "orb"
)

# 其余步骤同上
```

#### 方法C：深度引导重投影

```python
compositor = BackgroundCompositor(method=WarpMethod.DEPTH)
compositor.load_background("background.jpg")

# 加载深度图
depth_map = cv2.imread("depth.png", cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0

# 设置深度方法
compositor.setup_depth_method(
    depth_map=depth_map,
    background_image_shape=(1080, 1920),
    target_image_shape=(1080, 1920),
    hfov_deg=70.0,
    baseline=0.065,
    rotation_y_deg=5.0
)

# 其余步骤同上
```

## 模块说明

### IntrinsicsEstimator
- 从EXIF或HFOV估计相机内参
- 支持35mm等效焦距自动读取
- 提供回退机制确保稳定性

### HomographyWarper
- 支持手动点和自动特征匹配
- 使用RANSAC提高鲁棒性
- 输出重投影误差等元信息

### DepthWarper
- 3D点云反投影
- 前向渲染 + z-buffer遮挡处理
- 双边滤波平滑深度
- 自动空洞填补

### BackgroundCompositor
- 统一的接口整合所有功能
- 支持前景背景合成
- 边缘羽化和颜色匹配

## 注意事项

1. **坐标点标注**：确保手动点按照(x, y)顺序，像素坐标从左上角(0,0)开始
2. **深度图单位**：确保深度值为实际物理单位（米），不是归一化值
3. **掩码格式**：前景掩码应为灰度图，255=前景，0=背景
4. **图像尺寸**：所有输入图像尺寸应一致或手动调整

## 调试建议

1. 先运行`example_without_foreground_mask()`检查背景扭曲效果
2. 使用`print(compositor.warper.meta)`查看单应矩阵质量指标
3. 深度方法会输出覆盖率百分比，低于80%需检查参数
4. 调整`feather_radius`参数优化前景背景过渡

详细示例请参考 `example_usage.py`
