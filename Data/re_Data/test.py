import torch
import torch.nn as nn
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

# 定义符号到标签的映射
object_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
               10: '+', 11: '-', 12: '×', 13: '÷', 14: '(', 15: ')', 16: '=',
               17: 'log', 18: 'sqrt', 19: 'sin', 20: 'cos', 21: 'π'}

model=torch.load("model25.pth",map_location=torch.device('cpu'),weights_only=False)
# 数据预处理
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

# 加载测试数据集（请根据实际情况修改数据集路径）
try:
    test_dataset = ImageFolder(root="test_data", transform=transform)
    print(f"测试集加载成功，共{len(test_dataset)}个样本")

    # 为了加速演示，可以只使用部分样本
    if len(test_dataset) > 1000:
        indices = np.random.choice(len(test_dataset), 1000, replace=False)
        test_dataset = Subset(test_dataset, indices)
        print(f"使用{len(test_dataset)}个样本进行测试")

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
except Exception as e:
    print(f"数据集加载失败: {e}")
    exit()

# 收集预测结果和真实标签
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        img_tran = torch.reshape(images, (1, 1, 32, 32))
        pred = model(img_tran)
        _, predicted = torch.max(pred, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(15, 15))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(object_dict.values()))
disp.plot(include_values=True, cmap='Blues', xticks_rotation='vertical', values_format='d')
plt.title('混淆矩阵')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 计算分类准确率
accuracy = np.trace(cm) / np.sum(cm)
print(f"整体准确率: {accuracy:.4f}")

# 计算每个类别的准确率
class_accuracy = cm.diagonal() / cm.sum(axis=1)
for i, (label, acc) in enumerate(zip(object_dict.values(), class_accuracy)):
    print(f"{label}的识别准确率: {acc:.4f}")