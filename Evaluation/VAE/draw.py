import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
fid_gen_real = [163.5690023018815,150.82359385292958,120.31659805863664,128.74636038842291,131.8199217680865,
                 130.9562419334468,102.60953183537428,116.45306521201218,92.10021357845605,129.61792859324575]
fid_real_real = [28.206246510788674,26.49741858198425,36.03752621390305,33.29976242766463,39.07646536065127,
                 32.29706532681135,32.96777021711142,39.92356706555438,33.51464268816713,32.73234156371501]

index = np.arange(len(labels))
bar_width = 0.35
opacity = 0.9

fig, ax = plt.subplots(figsize=(10, 6))

color_gen = '#4285F4'  # 蓝色
color_real = '#D0D0D0'  # 浅灰色（更易区分黑色标签）

bars_gen = ax.bar(index, fid_gen_real, bar_width,
                  alpha=opacity,
                  color=color_gen,
                  edgecolor='white',
                  label='Generated-Real',
                  zorder=3)

bars_real = ax.bar(index + bar_width, fid_real_real, bar_width,
                   alpha=opacity,
                   color=color_real,
                   edgecolor='white',
                   label='Real-Real',
                   zorder=3)

# 添加黑色数据标签（适配浅色柱体）
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom',
                fontsize=10,
                color='black',  # 黑色标签
                fontweight='bold')

add_labels(bars_gen)
add_labels(bars_real)

# **关键修改：Y轴从0开始**
ax.set_ylim(bottom=0)  # 设置Y轴下限为0
# 自动计算上限并预留10%空间
ymax = max(fid_gen_real + fid_real_real) * 1.1
ax.set_ylim(0, ymax)

ax.set_title('FID Score Comparison Between Generated and Real Samples',
             fontsize=16,
             fontweight='bold',
             pad=20)

ax.set_xlabel('Digit Class (0-9)',
              fontsize=12,
              labelpad=10)

ax.set_ylabel('FID Score',
              fontsize=12,
              labelpad=10)

ax.set_xticks(index + bar_width/2)
ax.set_xticklabels(labels, fontsize=10)

ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
ax.set_axisbelow(True)

ax.legend(loc='upper right',
          frameon=True,
          facecolor='white',
          edgecolor='lightgray',
          fontsize=10,
          ncol=1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(top=0.92)

plt.show()