import os
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from torch import nn
from scipy.linalg import sqrtm
from PIL import Image


def preprocess_images(images):
    """
    对输入图像进行预处理，使其符合Inception V3模型的输入要求
    """
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    processed_images = []
    for image in images:
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
        processed_image = preprocess(image.transpose(1, 2, 0)).unsqueeze(0)
        processed_images.append(processed_image)
    return torch.cat(processed_images, dim=0)


def calculate_activation_statistics(images, model, batch_size=32, dims=2048):
    """
    此函数用于计算输入图像的激活统计信息，即特征均值和协方差矩阵。
    """
    model.eval()
    act = np.empty((len(images), dims))

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        with torch.no_grad():
            pred = model(batch)
        if len(pred.shape) > 2:
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.reshape(pred.size(0), -1)
        act[i:i + batch_size] = pred.cpu().numpy()

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    此函数用于计算两个高斯分布之间的Frechet距离。
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_fid(real_images, generated_images, model, dims=2048):
    """
    此函数用于计算真实图像和生成图像之间的FID得分，使用Inception V3模型。
    """
    real_images = preprocess_images(real_images)
    generated_images = preprocess_images(generated_images)
    mu_real, sigma_real = calculate_activation_statistics(real_images, model, dims=dims)
    mu_generated, sigma_generated = calculate_activation_statistics(generated_images, model, dims=dims)
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_generated, sigma_generated)
    return fid


def load_images_from_folder(folder, target_size=(28, 28), num_samples=1000):
    images = []
    count = 0
    for filename in os.listdir(folder):
        if count >= num_samples:
            break
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('L')  # 转换为单通道灰度图
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0  # 归一化到 [0, 1]
            img_array = np.expand_dims(img_array, axis=0)  # 添加通道维度
            images.append(img_array)
            count += 1
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    return np.array(images)


inception_model = models.inception_v3(pretrained=True)
inception_model.fc = nn.Identity()
inception_model.AuxLogits = None

# 示例使用，导入自己的图片
if __name__ == "__main__":
    for i in range(10):
        real_images_folder = f"Train_data/{i}"
        generated_images_folder = f"Train_data/{i}"
        real_images = load_images_from_folder(real_images_folder, num_samples=1000)
        generated_images = load_images_from_folder(generated_images_folder,num_samples=100)
        fid_score = calculate_fid(real_images, generated_images, inception_model)
        print(f"FID score{i}: {fid_score}")
