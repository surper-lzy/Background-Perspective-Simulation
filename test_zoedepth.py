"""
快速测试 ZoeDepth 是否正常工作
运行此脚本验证模型加载和深度估计功能
"""
import os
import sys

print("=" * 60)
print("ZoeDepth 集成测试")
print("=" * 60)

# 测试 1: 检查 ZoeDepth 目录
print("\n[测试 1/4] 检查 ZoeDepth 目录...")
zoedepth_path = os.path.join(os.path.dirname(__file__), "ZoeDepth")
if os.path.exists(zoedepth_path):
    print(f"   ✅ ZoeDepth 目录存在: {zoedepth_path}")
    hubconf_path = os.path.join(zoedepth_path, "hubconf.py")
    if os.path.exists(hubconf_path):
        print(f"   ✅ hubconf.py 文件存在")
    else:
        print(f"   ❌ hubconf.py 文件不存在")
        sys.exit(1)
else:
    print(f"   ❌ ZoeDepth 目录不存在: {zoedepth_path}")
    print("\n请确保已将 ZoeDepth 目录放置在项目根目录下")
    sys.exit(1)

# 测试 2: 检查依赖
print("\n[测试 2/4] 检查 Python 依赖...")
try:
    import torch
    print(f"   ✅ torch 版本: {torch.__version__}")
    print(f"   ✅ CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA 版本: {torch.version.cuda}")
except ImportError:
    print("   ❌ torch 未安装，请运行: pip install torch torchvision")
    sys.exit(1)

try:
    import cv2
    print(f"   ✅ opencv-python 版本: {cv2.__version__}")
except ImportError:
    print("   ❌ opencv-python 未安装，请运行: pip install opencv-python")
    sys.exit(1)

try:
    import timm
    print(f"   ✅ timm 已安装")
except ImportError:
    print("   ❌ timm 未安装，请运行: pip install timm")
    sys.exit(1)

# 测试 3: 加载 ZoeDepth 模型
print("\n[测试 3/4] 加载 ZoeDepth 模型...")
try:
    from depth_estimator import ZoeDepthEstimator
    estimator = ZoeDepthEstimator(model_type="ZoeD_NK", use_local=True)

    if estimator.model is not None:
        print("   ✅ ZoeDepth 模型加载成功")
    else:
        print("   ❌ ZoeDepth 模型加载失败")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ 模型加载异常: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 4: 深度估计（如果有测试图像）
print("\n[测试 4/4] 测试深度估计功能...")
test_image_path = "test_data/036.jpg"

if not os.path.exists(test_image_path):
    print(f"   ⚠️  测试图像不存在: {test_image_path}")
    print("   提示：运行 'python generate_test_data.py' 生成测试数据")
    print("\n跳过深度估计测试（前3项测试已通过）")
else:
    try:
        import cv2
        import numpy as np

        # 读取测试图像
        image = cv2.imread(test_image_path)
        print(f"   ✅ 测试图像加载成功: {image.shape}")

        # 估计深度
        print("   ⏳ 正在估计深度（可能需要几秒）...")
        depth = estimator.estimate_depth(image)

        print(f"   ✅ 深度估计成功")
        print(f"      - 深度图尺寸: {depth.shape}")
        print(f"      - 深度范围: {depth.min():.2f}m - {depth.max():.2f}m")

        # 保存深度图
        output_dir = "test_data"
        depth_output = os.path.join(output_dir, "zoedepth_test_output.png")
        depth_uint16 = (depth * 1000.0).astype(np.uint16)
        cv2.imwrite(depth_output, depth_uint16)
        print(f"   ✅ 深度图已保存: {depth_output}")

        # 保存可视化
        depth_vis = estimator._visualize_depth(depth)
        vis_output = os.path.join(output_dir, "zoedepth_test_visualization.jpg")
        cv2.imwrite(vis_output, depth_vis)
        print(f"   ✅ 深度可视化已保存: {vis_output}")

    except Exception as e:
        print(f"   ❌ 深度估计失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# 全部测试通过
print("\n" + "=" * 60)
print("🎉 所有测试通过！ZoeDepth 集成成功")
print("=" * 60)
print("\n下一步：")
print("  1. 运行 'python run_with_zoedepth.py' 执行完整流程")
print("  2. 查看 'ZOEDEPTH_INTEGRATION_GUIDE.md' 了解详细用法")

