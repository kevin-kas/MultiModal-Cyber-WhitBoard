import numpy as np
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from PIL import Image, ImageEnhance, ImageFilter
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA

np.random.seed(42)


def load_images_and_labels(folder_path, augment=False):
    images = []
    labels = []
    class_names = [d for d in os.listdir(folder_path)
                   if os.path.isdir(os.path.join(folder_path, d))]

    for class_name in class_names:
        class_folder = os.path.join(folder_path, class_name)
        img_names = [f for f in os.listdir(class_folder)
                     if not f.startswith('.') and '.ipynb' not in f]

        for img_name in img_names:
            img_path = os.path.join(class_folder, img_name)
            try:
                img = Image.open(img_path).convert('L')  # 转为灰度图
                img = img.resize((45, 45))

                if augment:
                    images.append(np.array(img).flatten())
                    labels.append(class_name)

                    for angle in [-15, 15]:
                        rotated_img = img.rotate(angle, expand=False)
                        images.append(np.array(rotated_img).flatten())
                        labels.append(class_name)

                    enhancer = ImageEnhance.Brightness(img)
                    for factor in [0.8, 1.2]:
                        brightened_img = enhancer.enhance(factor)
                        images.append(np.array(brightened_img).flatten())
                        labels.append(class_name)
                else:
                    images.append(np.array(img).flatten())
                    labels.append(class_name)

            except Exception as e:
                print(f"无法加载图像 {img_path}: {e}")

    return np.array(images), np.array(labels)


print("正在加载训练数据...")
train_images, train_labels = load_images_and_labels('../train_data', augment=False)
print(f"训练数据加载完成，共{len(train_images)}个样本")

print("正在加载测试数据...")
test_images, test_labels = load_images_and_labels('../test_data', augment=False)
print(f"测试数据加载完成，共{len(test_images)}个样本")

if train_labels.ndim > 1:
    train_labels = train_labels.ravel()
if test_labels.ndim > 1:
    test_labels = test_labels.ravel()

le = LabelEncoder()
train_labels_encoded = le.fit_transform(train_labels)
test_labels_encoded = le.transform(test_labels)

scaler = StandardScaler()
train_images_scaled = scaler.fit_transform(train_images)
test_images_scaled = scaler.transform(test_images)

pca = PCA(n_components=0.95)
train_images_pca = pca.fit_transform(train_images_scaled)
test_images_pca = pca.transform(test_images_scaled)

print(f"特征维度从 {train_images.shape[1]} 降到 {train_images_pca.shape[1]}")


base_gnb = GaussianNB()

param_grid = {'var_smoothing': np.logspace(0, -9, num = 100)}

grid_search = GridSearchCV(
    estimator = base_gnb, param_grid = param_grid, cv = 5
)

grid_search.fit(train_images_pca, train_labels_encoded)
best_gnb = grid_search.best_estimator_

print(f"最佳参数: {grid_search.best_params_}")

train_accuracy = best_gnb.score(train_images_pca, train_labels_encoded)
test_accuracy = best_gnb.score(test_images_pca, test_labels_encoded)

print(f"训练集准确率: {train_accuracy:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")

test_predictions = best_gnb.predict(test_images_pca)
print("\n分类报告:")
print(classification_report(
    test_labels_encoded,
    test_predictions,
    target_names=le.classes_
))

cm = confusion_matrix(test_labels_encoded, test_predictions)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)

plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.title('混淆矩阵')
plt.tight_layout()
plt.savefig('confusion_matrix_gnb.png')
plt.show()

# 保存模型和相关组件
joblib.dump(best_gnb, 'gnb_model.pkl')
joblib.dump(scaler,'scaler.pkl')
joblib.dump(pca, 'pca.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("模型和预处理组件已保存")

def predict_single_image(image_path):
    loaded_model = joblib.load('gnb_model.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    loaded_pca = joblib.load('pca.pkl')
    loaded_le = joblib.load('label_encoder.pkl')

    img = Image.open(image_path).convert('L')
    img = img.resize((45, 45))
    img_array = np.array(img).flatten().reshape(1, -1)

    img_scaled = loaded_scaler.transform(img_array)
    img_pca = loaded_pca.transform(img_scaled)

    prediction_encoded = loaded_model.predict(img_pca)
    prediction_class = loaded_le.inverse_transform(prediction_encoded)

    return prediction_class[0]
try:
    sample_prediction = predict_single_image('./test_data/1/exp79.jpg')
    print(f"示例预测结果: {sample_prediction}")
except Exception as e:
    print(f"预测示例图像时出错: {e}")
    print("请确保测试图像路径正确")