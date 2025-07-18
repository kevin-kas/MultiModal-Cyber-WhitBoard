import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import seaborn as sns
plt.style.use('seaborn-v0_8')
data_dir = 'Train_data'
class_names = []
data_distribution = []

for i in os.listdir(data_dir):
    class_path = os.path.join(data_dir, i)
    if os.path.isdir(class_path):
        class_names.append(i)
        data_distribution.append(len(os.listdir(class_path)))

if not data_distribution:
    print("No valid data directories found or data is empty")
    exit()
plt.figure(figsize=(12, 6))
colors = sns.color_palette("viridis", len(class_names))
x = np.arange(len(class_names))
bars = plt.bar(x, data_distribution, width=0.6, color=colors, alpha=0.8, edgecolor='black', linewidth=0.7)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height}', ha='center', va='bottom', fontsize=10)
plt.title('Training Data Class Distribution', fontsize=15, pad=15)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.xticks(x, class_names, rotation=45, ha='right', fontsize=10)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.gca().set_facecolor('#f8f9fa')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(class_names))]
plt.legend(handles, class_names, loc='upper right', bbox_to_anchor=(1.2, 1))
plt.tight_layout()
plt.show()