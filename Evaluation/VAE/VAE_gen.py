import torch
from PIL import Image
import torchvision.utils as vutils
import torchvision.transforms as transforms
from VAE import CVAE
model=CVAE(num_classes=10)

state_dict = torch.load('cvae_model.pth')
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

def generate_fig(digit_string,num1):
    generated_images = generate_digits(model, digit_string)
    import matplotlib.pyplot as plt
    import numpy as np
    grid = vutils.make_grid(generated_images.cpu(), nrow=len(digit_string), normalize=True,padding=0)
    plt.figure(figsize=(0.28, 0.28))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis('off')
    plt.savefig(f'output2/{digit_string}/task{num1}.png')
    plt.close()

for i in range(10):
    digit_string = str(i)
    for num in range(100):
        generate_fig(digit_string,num)
    print(i)