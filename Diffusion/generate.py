import sys
import matplotlib.pyplot as plt
from .labeled_Train import *
import time
def dif_generate(num):
    model = torch.load('Diffusion/Unet.pth', weights_only=False, map_location='cpu')
    image_list = []
    gaussian_diffusion = GaussianDiffusion(timesteps=500)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for char in str(num):
        if char.isdigit():
            digit = int(char)
            target_labels = torch.full((1,), digit, dtype=torch.long, device=device)
            generated_images = gaussian_diffusion.sample(model, 28, batch_size=1, channels=1, y_label=target_labels)
            image_list.append(generated_images[499])
        elif char == '.':
            try:
                image = Image.open('Diffusion/point.png').convert('L').resize((28, 28))
                image_array = (np.array(image) / 255.0 * 2 - 1).reshape(1, 1, 28, 28)
                image_list.append(image_array[0, 0])
            except FileNotFoundError:
                print("Error: point.png not found!")
                sys.exit(1)
        else:
            print(f"Invalid symbol: {char}")
            sys.exit(1)

    if len(image_list) == 0:
        print("No images generated!")
        sys.exit(0)

    fig, axes = plt.subplots(1, len(image_list), figsize=(len(image_list) * 1.2, 2))
    if len(image_list) == 1:
        axes = [axes]

    for i, img_data in enumerate(image_list):
        if isinstance(img_data, torch.Tensor):
            img = img_data.squeeze().cpu().numpy()
        else:
            img = img_data.squeeze()
        img = (img + 1) / 2
        if img.shape != (28, 28):
            img = img.reshape(28, 28)
        threshold = 0.8
        binary_img = np.where(img > threshold, 1, 0)

        axes[i].imshow(binary_img, cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'Diffusion/output/output{time.time()}.png')