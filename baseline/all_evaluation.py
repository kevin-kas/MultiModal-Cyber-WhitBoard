import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8')

acc_list = [0.69, 0.75, 0.85,0.91 ,0.99]
recall_list = [0.69, 0.7, 0.85, 0.91,0.97]
f1_list = [0.48, 0.72, 0.84, 0.95,0.99]
models = ['GNB', 'PCA+ETC', 'PCA+MLP','PCA+SVM', 'CNN']

x = np.arange(len(models))
width = 0.25

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制多组柱状图
offsets = [-width, 0, width]
metrics = [acc_list, recall_list, f1_list]
labels = ['Accuracy', 'Recall', 'F1-Score']
colors = ['#4C72B0', '#55A868', '#C6B46C']

for i, (offset, metric, label, color) in enumerate(zip(offsets, metrics, labels, colors)):
    rects = ax.bar(x + offset, metric, width, label=label, color=color, alpha=0.8, edgecolor='black', linewidth=0.7)
    ax.bar_label(rects, padding=3, fmt='%.2f')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Performance Metrics Comparison of Different Models', fontsize=15, pad=15)
ax.set_xticks(x, models, fontsize=10)
ax.set_ylim(0, 1.1)

ax.grid(axis='y', linestyle='--', alpha=0.7)

ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

ax.set_facecolor('#f8f9fa')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()