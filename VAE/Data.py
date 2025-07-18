import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import os

transforms=torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor()
    ]
)

class Data(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.listdir(self.path)

        for i in self.image_path:
            if not i.endswith('.png'):
                self.image_path.remove(i)
        self.class_to_idx = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                             '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                             }

    def __getitem__(self, idx):
        image_name=self.image_path[idx]
        image_item_path=os.path.join(self.root_dir,self.label_dir,image_name)
        img=transforms(Image.open(image_item_path).convert('L'))
        return img,torch.tensor(self.class_to_idx[image_name.strip('.png').split('_')[-1]])

    def __len__(self):
        return len(self.image_path)