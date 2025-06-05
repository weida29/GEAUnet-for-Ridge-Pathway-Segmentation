mport matplotlib.pyplot as plt
import numpy as np

# 模型名称
models = [
    "UNet", "BiUNet", "LightMUNet", "ULite", "DeepLabv3",
    "MALUNet", "EGEUNet", "GEAUNet(ours)"
]

# 总的mIoU和参数量
total_miou = [75.0, 76.8, 73.7, 81.9, 73.5, 78.8, 78.9, 82.4]
params = [5.7, 19.429636, 0.380839, 0.878468, 11.021108,
          0.858202, 0.514648, 0.987264]

# 设置点的大小为一致的值
point_size = 100  # 可以调整这个值来改变点的大小

# 定义颜色（使用相同的颜色，对于一致性）
colors = np.linspace(0, 1, len(models))

# 创建散点图
plt.figure(figsize=(12, 8))
scatter = plt.scatter(params, total_miou, c=colors, s=point_size, cmap='viridis', alpha=0.7, edgecolors="w", linewidth=0.5)

# 添加标签
for i, model in enumerate(models):
    plt.annotate(model, (params[i], total_miou[i]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=10)  # 增加文字大小

# 设置标题和轴标签
plt.title("Model Comparison: Total mIoU vs Parameters", fontsize=16)
plt.xlabel("Parameters (M)", fontsize=12)
plt.ylabel("Total mIoU (%)", fontsize=12)

# 添加颜色条
plt.colorbar(scatter, label='Model Index')

# 显示网格
plt.grid(True, linestyle='--', alpha=0.6)

# 显示图形
plt.show()