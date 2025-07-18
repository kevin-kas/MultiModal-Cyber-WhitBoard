FID_between_generate_and_real=[64.50195658055614,56.65962734712522,77.19346595286805,77.55579889099393,78.46428601835493
,70.60129160206917,66.36257642728084,64.75679091657719,72.74357792908864,72.50566084012017]
FID_between_real_and_real=[ 28.206246510788674,26.49741858198425,36.03752621390305,33.29976242766463,39.07646536065127
,32.29706532681135,32.96777021711142,39.92356706555438,33.51464268816713,32.73234156371501]
label=[0,1,2,3,4,5,6,7,8,9]

import numpy as np
import matplotlib.pyplot as plt
# 设置柱状图的位置和宽度
index = np.arange(len(label))
bar_width = 0.35

# 绘制柱状图
fig, ax = plt.subplots()
bars_A = ax.bar(index, FID_between_generate_and_real, bar_width, label='generate-real', color='skyblue')
bars_B = ax.bar(index + bar_width, FID_between_real_and_real, bar_width, label='real-real', color='lightcoral')

# 添加图表元素
ax.set_title('Comparison Between Real and Generate Figure')
ax.set_xlabel('Numbers')
ax.set_ylabel('FID score')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(label)
ax.legend()

# 显示图表
plt.show()