import os
import cv2
import argparse
import torch
import json
import pickle as pkl
from tqdm import tqdm
import time

from utils import load_config, load_checkpoint, compute_edit_distance
from models.infer_model import Inference
from dataset import Words
from counting_utils import gen_counting_label

import numpy as np
import pandas as pd
parser = argparse.ArgumentParser(description='model testing')
parser.add_argument('--dataset', default='CROHME', type=str, help='数据集名称')
parser.add_argument('--image_path', default='datasets/CROHME/14_test_images.pkl', type=str, help='测试image路径')
parser.add_argument('--label_path', default='datasets/CROHME/14_test_labels.txt', type=str, help='测试label路径')
parser.add_argument('--word_path', default='datasets/CROHME/words_dict.txt', type=str, help='测试dict路径')

parser.add_argument('--draw_map', default=False)
args = parser.parse_args()

if not args.dataset:
    print('请提供数据集名称')
    exit(-1)

if args.dataset == 'CROHME':
    config_file = 'config.yaml'

"""加载config文件"""
params = load_config(config_file)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = device
words = Words(args.word_path)
params['word_num'] = len(words)

if 'use_label_mask' not in params:
    params['use_label_mask'] = False
print(params['decoder']['net'])
model = Inference(params, draw_map=args.draw_map)
model = model.to(device)

load_checkpoint(model, None, params['checkpoint'])
model.eval()

with open(args.image_path, 'rb') as f:
    images = pkl.load(f)

with open(args.label_path) as f:
    lines = f.readlines()

line_right = 0
e1, e2, e3 = 0, 0, 0
bad_case = {}
model_time = 0
mae_sum, mse_sum = 0, 0
mae_total = []
mse_total = []
with torch.no_grad():
    for line in tqdm(lines):
        name, *labels = line.split()
        name = name.split('.')[0] if name.endswith('jpg') else name
        input_labels = labels
        labels = ' '.join(labels)
        img = images[name]
        img = torch.Tensor(255-img) / 255
        img = img.unsqueeze(0).unsqueeze(0)
        img = img.to(device)
        a = time.time()
        
        input_labels = words.encode(input_labels)
        input_labels = torch.LongTensor(input_labels)
        input_labels = input_labels.unsqueeze(0).to(device)

        probs, _, mae, mse = model(img, input_labels, os.path.join(params['decoder']['net'], name))
        mae_total.append(mae)
        mse_total.append(mse)
        mae_sum += mae
        mse_sum += mse
        model_time += (time.time() - a)

        prediction = words.decode(probs)
        if prediction == labels:
            line_right += 1
        else:
            bad_case[name] = {
                'label': labels,
                'predi': prediction
            }
            print(name, prediction, labels)

        distance = compute_edit_distance(prediction, labels)
        if distance <= 1:
            e1 += 1
        if distance <= 2:
            e2 += 1
        if distance <= 3:
            e3 += 1

print(f'model time: {model_time}')
print(f'ExpRate: {line_right / len(lines)}')
print(f'mae: {mae_sum / len(lines)}')
print(f'mse: {mse_sum / len(lines)}')
print(f'e1: {e1 / len(lines)}')
print(f'e2: {e2 / len(lines)}')
print(f'e3: {e3 / len(lines)}')

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(mae_total, bins=100)
plt.xlabel('The MAE of prediction for each class of symbol')
plt.ylabel('count')
plt.title('mae distribution')
plt.subplot(1, 2, 2)
plt.hist(mse_total, bins=100)
plt.xlabel('The MSE of prediction for each class of symbol')
plt.ylabel('count')
plt.title('mse distribution')
plt.savefig('mse_and_mae_distribution.png')
plt.show()

class_correct = {word: 0 for word in words.words_dict.keys()}  # Changed to words_dict.keys()
class_total = {word: 0 for word in words.words_dict.keys()}    # Changed here too

class_tp = {word: 0 for word in words.words_dict.keys()}
class_fp = {word: 0 for word in words.words_dict.keys()}
class_fn = {word: 0 for word in words.words_dict.keys()}

for line in lines:
    name, *true_labels = line.split()
    name = name.split('.')[0] if name.endswith('jpg') else name
    true_labels = ' '.join(true_labels)
    pred_labels = bad_case.get(name, {}).get('predi', '')
    
    true_word_list = true_labels.split()
    pred_word_list = pred_labels.split()
    for true_word, pred_word in zip(true_word_list, pred_word_list):
        if true_word in words.words_dict:
            class_total[true_word] += 1
            if true_word == pred_word:
                class_correct[true_word] += 1
                class_tp[true_word] += 1
            else:
                class_fn[true_word] += 1
                if pred_word in class_fp:
                    class_fp[pred_word] += 1

class_precision = {}
class_recall = {}
class_f1 = {}


special_tokens = {'eos', 'sos'}  # Add any other special tokens used
for word in class_total:
    # Handle special tokens that might not be in words_dict
    if word not in words.words_dict:
        continue
        
    tp = class_tp.get(word, 0)
    fp = class_fp.get(word, 0)
    fn = class_fn.get(word, 0)
    
    # 处理分母为零的情况
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    class_precision[word] = precision
    class_recall[word] = recall
    class_f1[word] = f1

# 计算每个类别的准确率 (with fallback for zero counts)
class_accuracies = {
    word: class_correct[word] / count if count > 0 else 0.0 
    for word, count in class_total.items()
}

# 修改原sorted_accuracies为包含更多指标 (this is where the error occurs)
sorted_metrics = sorted(
    [(word, class_accuracies[word], class_precision[word], class_recall[word], class_f1[word]) 
     for word in class_total],
    key=lambda x: x[1], 
    reverse=True
)

# 在DataFrame中新增列
acc_df = pd.DataFrame(sorted_metrics, columns=['Class', 'Accuracy', 'Precision', 'Recall', 'F1'])
# 保存到CSV文件
acc_df.to_csv('accuracy_metrics.csv', index=False)

print(acc_df)

# 新增整体指标计算（微平均）
total_tp = sum(class_tp.values())
total_fp = sum(class_fp.values())
total_fn = sum(class_fn.values())

micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

print(f'\nMicro Average:')
print(f'Precision: {micro_precision:.4f}')
print(f'Recall: {micro_recall:.4f}')
print(f'F1 Score: {micro_f1:.4f}')

# 宏平均计算
macro_precision = sum(class_precision.values()) / len(class_precision)
macro_recall = sum(class_recall.values()) / len(class_recall)
macro_f1 = sum(class_f1.values()) / len(class_f1)

print(f'\nMacro Average:')
print(f'Precision: {macro_precision:.4f}')
print(f'Recall: {macro_recall:.4f}')
print(f'F1 Score: {macro_f1:.4f}')


# 计算每个类别的准确率
class_accuracies = {}
for word in class_total:
    if class_total[word] > 0:
        class_accuracies[word] = class_correct[word] / class_total[word]

# 按准确率降序排序
# 修改这个重复的块 (remove extra columns from this DataFrame)
# 修正第一个DataFrame的打印（第226行附近）
print("详细分类指标:")
print(acc_df)  # 这个df已经是5列的数据

# 修正第二个重复的DataFrame打印（第240行附近）
sorted_accuracies = sorted(class_accuracies.items(), key=lambda item: item[1], reverse=True)
sorted_accuracies = sorted_accuracies[:5] + sorted_accuracies[-10:-5]  # 取前5和后5个非零类别
acc_df = pd.DataFrame(sorted_accuracies, columns=['Class', 'Accuracy'])  # Only 2 columns here
print(acc_df)   

# 创建太阳图时只使用准确率数据
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='polar')
categories = len(sorted_metrics)  # 使用完整的sorted_metrics数据
angles = [n / float(categories) * 2 * np.pi for n in range(categories)]

# === 新增颜色和宽度定义 ===
colors = plt.cm.viridis(np.linspace(0, 1, categories))  # 颜色映射
width = 2 * np.pi / categories * 0.9  # 条形宽度
# ==========================

bars = ax.bar(
    angles,
    [acc[1] for acc in sorted_metrics],  # 只取accuracy值
    width=width,
    color=colors,
    edgecolor='white',
    linewidth=0.5,
    align='center'
)

# 添加标签和装饰
ax.set_theta_offset(np.pi/2)  # 旋转起始角度
ax.set_theta_direction(-1)    # 顺时针方向
ax.set_rlabel_position(315)    # 半径标签位置
plt.title("Symbol Recognition Accuracy Sunburst", pad=20)

# 添加图例
legend_labels = [f"{word.replace(chr(92), '')}: {acc:.2f}"  # 使用chr(92)代替反斜杠
                for word, acc in sorted_accuracies]
plt.legend(bars, legend_labels, 
          loc='upper right',
          bbox_to_anchor=(1.15, 1.15),
          fontsize=8)

plt.savefig('sunburst_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

# 删除原来的绘图循环
for word, accuracy in sorted_accuracies:
    plt.figure(figsize=(6, 4))
    plt.bar([word], [accuracy], color='blue')
    plt.ylim(0, 1)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy for Class {word}')
    
    # Sanitize filename by replacing backslashes with underscores
    safe_word = word.replace('\\', '_')
    plt.savefig(f'accuracy_.png')
    
    plt.close()

with open(f'{params["decoder"]["net"]}_bad_case.json','w') as f:
    json.dump(bad_case,f,ensure_ascii=False)
