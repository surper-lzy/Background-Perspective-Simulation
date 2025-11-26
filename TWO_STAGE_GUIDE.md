# 两阶段背景替换系统 - 完整说明

## 🎯 核心改进

已成功实现**两阶段自动匹配流程**，解决了"参考场景与新背景内容不同"的问题：

### 阶段1：学习相机几何关系
- 输入：`reference_left.jpg` ↔ `reference_right.jpg`（办公室场景）
- 方法：SIFT特征匹配 + RANSAC
- 输出：左→右的相对单应矩阵 `H_ref_left_to_right`

### 阶段2：应用到新背景
- 输入：`background.jpg`（自然风景）+ 用户定义的平面四角点
- 计算：
  - `plane_to_left = getPerspectiveTransform(背景四角 → 左视角位置)`
  - `plane_to_right = H_ref_left_to_right @ plane_to_left`
- 输出：新背景映射到左右视角的单应矩阵

### 关键优势
✅ **内容解耦**：参考场景与新背景可以完全不同  
✅ **几何一致**：左右视角关系来自真实相机对  
✅ **灵活控制**：通过`plane_corners_in_left`调整背景位置和大小  
✅ **自动化**：无需手动标注参考图与背景的对应点  

---

## 📂 文件结构

```
C:\Users\lzy\Desktop\Test\
├── 核心模块
│   ├── intrinsics_estimator.py       # 内参估计
│   ├── homography_warper.py          # 单应变换（含两阶段方法）✨新增
│   ├── depth_warper.py               # 深度重投影
│   └── background_compositor.py      # 主合成器（含两阶段接口）✨新增
│
├── 测试脚本
│   ├── generate_test_data.py         # 生成测试数据 ✨完全重写
│   └── test_run.py                   # 运行测试 ✨完全重写
│
├── 示例与文档
│   ├── example_usage.py              # 使用示例
│   ├── requirements.txt              # 依赖包
│   ├── README.md                     # 使用指南
│   ├── INPUT_DATA_GUIDE.md           # 输入数据说明
│   └── background_warping.md         # 原始方案文档
│
└── 生成的数据（运行后）
    ├── test_data/                    # 测试数据
    │   ├── reference_left.jpg        # 参考场景-左（办公室）
    │   ├── reference_right.jpg       # 参考场景-右（办公室）
    │   ├── background.jpg            # 新背景（自然风景）✨完全不同
    │   ├── depth_map.png             # 深度图（与新背景对应）✨已对齐
    │   ├── foreground_*.jpg          # 前景图
    │   └── mask_*.png                # 前景掩码
    │
    └── output/                       # 输出结果
        ├── warped_bg_left.jpg        # 仅背景扭曲-左
        ├── warped_bg_right.jpg       # 仅背景扭曲-右
        ├── two_stage_left.jpg        # 两阶段方法-左 ✨主要结果
        ├── two_stage_right.jpg       # 两阶段方法-右 ✨主要结果
        ├── depth_left.jpg            # 深度方法-左
        └── depth_right.jpg           # 深度方法-右
```

---

## 🚀 快速开始

### 1. 生成测试数据
```bash
python generate_test_data.py
```

**生成内容：**
- 参考场景：办公室风格（灰色墙面、窗户、门）
- 新背景：自然风景（蓝天、山脉、树木、建筑物）
- 深度图：与新背景完美对应（天空远、树近）
- 前景和掩码：模拟人物

### 2. 运行测试
```bash
python test_run.py
```

**测试项目：**
1. ✅ 仅背景扭曲（调试用）
2. ✅ 两阶段自动匹配（主要方法）
3. ✅ 深度引导重投影

### 3. 查看结果
检查 `output/` 目录中的图像，重点查看：
- `two_stage_left.jpg` / `two_stage_right.jpg` - 自动匹配结果
- `depth_left.jpg` / `depth_right.jpg` - 深度重投影结果

---

## 🔧 实际使用示例

### 场景：替换会议室背景为海滩风景

```python
import cv2
from background_compositor import BackgroundCompositor, WarpMethod

# 1. 创建合成器
compositor = BackgroundCompositor(method=WarpMethod.HOMOGRAPHY)

# 2. 加载新背景（海滩风景）
compositor.load_background("beach_background.jpg")

# 3. 加载左右相机拍摄的参考场景（原始会议室）
ref_left = cv2.imread("meeting_room_left.jpg")
ref_right = cv2.imread("meeting_room_right.jpg")

# 4. 定义背景在左视角中的显示位置（四个角点）
# 建议：覆盖整个可见区域，留出边缘
plane_corners = [
    (100, 100),      # 左上
    (1820, 100),     # 右上
    (1820, 980),     # 右下
    (100, 980)       # 左下
]

# 5. 两阶段自动匹配
compositor.setup_two_stage_homography(
    ref_left, ref_right, plane_corners, auto_method="sift"
)

# 6. 加载前景（会议中的人）和掩码
compositor.load_foreground_masks(
    mask_left_path="person_mask_left.png",
    mask_right_path="person_mask_right.png"
)

# 7. 处理并保存
foreground_left = cv2.imread("person_left.jpg")
foreground_right = cv2.imread("person_right.jpg")

result_left, result_right = compositor.process_stereo_pair(
    foreground_left, foreground_right, 
    (1920, 1080),
    feather_radius=5,
    color_matching=True  # 启用颜色匹配
)

cv2.imwrite("output_left.jpg", result_left)
cv2.imwrite("output_right.jpg", result_right)
```

---

## 📊 两种方法对比

| 特性 | 两阶段单应变换 | 深度重投影 |
|------|---------------|-----------|
| **适用场景** | 平面背景、远景 | 复杂3D场景 |
| **输入要求** | 参考对 + 平面角点 | 参考对 + 深度图 |
| **计算复杂度** | 低 | 中等 |
| **遮挡处理** | 无 | 有（z-buffer） |
| **视差一致性** | 近似 | 精确 |
| **鲁棒性** | 高（仅依赖4点） | 依赖深度质量 |
| **推荐场景** | 虚拟背景墙、海报 | 室外风景、复杂场景 |

---

## ⚙️ 参数调整指南

### plane_corners_in_left（背景位置）
```python
# 全屏背景
[(50, 50), (1870, 50), (1870, 1030), (50, 1030)]

# 中央区域
[(400, 200), (1520, 200), (1520, 880), (400, 880)]

# 左侧区域（如虚拟窗户）
[(100, 100), (800, 100), (800, 980), (100, 980)]
```

### 深度参数（方法C）
```python
hfov_deg = 70.0          # 视场角，60-80度
baseline = 0.065         # 基线距离（米），0.05-0.10
rotation_y_deg = 5.0     # 旋转角度（度），3-7
```

---

## 🛠️ 常见问题

### Q1: 两阶段匹配失败，提示特征不足？
**A**: 
- 确保参考对（左右图）有足够纹理和重叠区域
- 尝试增加 `max_features=10000`
- 检查图像质量，避免过度模糊或曝光不足

### Q2: 背景扭曲效果不理想？
**A**: 
- 调整 `plane_corners_in_left` 的位置和大小
- 检查 `warped_bg_*.jpg` 确认背景变换是否正确
- 确保平面角点顺序为：左上→右上→右下→左下

### Q3: 深度方法覆盖率低？
**A**: 
- 检查深度图单位是否正确（米）
- 调整 `baseline` 和 `rotation_y_deg`
- 确保深度图与背景图完全对齐

### Q4: 前景边缘有"贴片感"？
**A**: 
- 增加 `feather_radius=10`
- 启用 `color_matching=True`
- 提高前景掩码质量（使用更好的抠图算法）

---

## 📈 性能优化建议

1. **降低分辨率**：先用640×360测试，再用全分辨率
2. **缓存单应矩阵**：相同参考对可复用几何关系
3. **批处理**：多帧视频可共享参考对几何
4. **GPU加速**：使用 `cv2.cuda` 模块（需编译支持）

---

## 🎓 技术细节

### 两阶段方法的数学原理

**阶段1：恢复相机几何**
```
H_ref = findHomography(ref_left, ref_right)
```
表示：右视角 = H_ref × 左视角

**阶段2：应用到新背景**
```
H_plane_to_left = getPerspectiveTransform(bg_corners, left_corners)
H_plane_to_right = H_ref @ H_plane_to_left
```
表示：
- 背景 → 左视角：直接映射
- 背景 → 右视角：先映射到左，再通过 H_ref 到右

### 关键假设
1. 参考对与新背景的相机内参相同
2. 新背景可视为单个平面
3. 相机仅有平移和旋转（无缩放）

---

## 📝 总结

✅ **已实现两阶段自动匹配**：参考场景与新背景完全不同  
✅ **深度图已对齐**：深度图与新背景几何一致  
✅ **完整测试流程**：一键生成数据并运行所有方法  
✅ **灵活可控**：通过角点调整背景位置  

现在你可以：
1. 运行 `python generate_test_data.py` 生成测试数据
2. 运行 `python test_run.py` 查看所有方法的效果
3. 根据需要调整参数或替换真实数据

祝使用愉快！🚀

