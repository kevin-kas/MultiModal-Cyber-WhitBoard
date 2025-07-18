import torch
from PIL import Image
import torchvision.utils as vutils
import torchvision.transforms as transforms
from .VAE import CVAE
import time
model=CVAE(num_classes=10)

state_dict = torch.load('VAE/cvae_model.pth')
model.load_state_dict(state_dict)
model.eval()

device='cpu'
def generate_digits(model, digit_string):
    model.eval()
    generated_images = []
    for digit in digit_string:
        digit = int(digit)
        label = torch.tensor([digit], dtype=torch.long).to(device)
        z = torch.randn(1, 20).to(device)
        with torch.no_grad():
            generated_image = model.decoder(z, label).view(1, 1, 28, 28)
            generated_images.append(generated_image)
    generated_images = torch.cat(generated_images, dim=0)
    return generated_images
def generate_fig(digit_string):
    import matplotlib.pyplot as plt
    import numpy as np
    def init_plt():
        """初始化matplotlib.pyplot设置"""
        plt.figure(figsize=(len(digit_string) * 0.7, 1))  # 根据数字长度动态调整图像宽度
        plt.rcParams['figure.dpi'] = 300  # 设置图像分辨率
        plt.rcParams['axes.facecolor'] = 'white'  # 设置背景颜色
        plt.rcParams['savefig.facecolor'] = 'white'  # 设置保存图像的背景颜色
    init_plt()
    if '.' in digit_string:
        digit_string1=digit_string.split('.')[0]
        digit_string2=digit_string.split('.')[1]
        generated_images1 = generate_digits(model, digit_string1)
        print(digit_string2)
        generated_images2 = generate_digits(model,digit_string2)
        print(digit_string1)

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        image_point=Image.open('VAE/point.png')
        image_point=transform(image_point).unsqueeze(0).to(device)
        new_image=torch.cat([generated_images1,image_point,generated_images2],dim=0)
        grid = vutils.make_grid(new_image.cpu(), nrow=len(digit_string), normalize=True,padding=0)

        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.axis('off')
        plt.savefig(f'VAE/output/output{time.time()}.png')
    else:
        generated_images = generate_digits(model, digit_string)
        grid = vutils.make_grid(generated_images.cpu(), nrow=len(digit_string), normalize=True,padding=0)
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.axis('off')
        plt.savefig(f'VAE/output/output{time.time()}.png')