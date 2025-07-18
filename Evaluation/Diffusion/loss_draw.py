import numpy as np
import matplotlib.pyplot as plt

# 提取Epoch和Loss数据
epochs = [int(i) for i in range(1, 101)]
losses = []
with open('Diffusion_train_log.txt', 'r') as f:
    for line in f:
        if 'Epoch:' in line:
            loss = float(line.split('loss=')[-1].strip())
            losses.append(loss)

# 绘制损失曲线（蓝色曲线）
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, color='#1f77b4', linewidth=2, alpha=0.9)  # 蓝色代码#1f77b4
plt.grid(True, linestyle='--', alpha=0.7)  # 虚线网格

# 图表美化（与原设置一致）
plt.title('Diffusion Model Training Loss Curve',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Epoch', fontsize=12, labelpad=10)
plt.ylabel('Loss', fontsize=12, labelpad=10)

plt.xlim(min(epochs)-1, max(epochs)+1)
plt.ylim(min(losses)*0.95, max(losses)*1.05)

# **调整注释位置：移动到右侧空白区域**
final_loss = losses[-1]
plt.text(max(epochs),  # x坐标设为最大Epoch值
         final_loss*1.02,  # y坐标保持原有偏移
         f'Final Loss: {final_loss:.4f}',
         fontsize=12,
         color='#1f77b4',
         fontweight='bold',
         ha='right',  # 文本右对齐
         va='bottom')  # 垂直底部对齐

plt.tight_layout()
plt.show()