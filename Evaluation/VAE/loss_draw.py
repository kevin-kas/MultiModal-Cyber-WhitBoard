import numpy as np
import matplotlib.pyplot as plt

# 从文档数据中提取Epoch和Loss
epochs = []
losses = []
with open('VAE_train_log.txt', 'r') as f:
    for line in f:
        if 'Epoch:' in line:
            epoch = int(line.split(',')[0].split(': ')[1])
            loss = float(line.split(',')[1].split(': ')[1])
            epochs.append(epoch)
            losses.append(loss)

# 绘制损失曲线（仅保留蓝色曲线，移除散点）
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, color='#1f77b4', linewidth=2, alpha=0.8)  # 蓝色曲线

# 图表美化（与原设置一致）
plt.title('VAE Training Loss Curve',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Epoch', fontsize=12, labelpad=10)
plt.ylabel('Loss', fontsize=12, labelpad=10)
plt.xticks(np.arange(0, 301, 20))  # 每20个Epoch显示刻度
plt.grid(True, linestyle='--', alpha=0.7)  # 显示网格线
plt.xlim(0, 300)  # 设置x轴范围
plt.ylim(min(losses)*0.95, max(losses)*1.05)  # 自适应y轴范围

# 添加文本注释（显示最终损失值）
final_loss = losses[-1]
plt.text(250, final_loss*1.02,
         f'Final Loss: {final_loss:.4f}',
         fontsize=12, color='red', fontweight='bold')

plt.tight_layout()
plt.show()