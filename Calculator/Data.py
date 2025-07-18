import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import os

transforms=torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor()
    ]
)

class Data(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.image_path=os.listdir(self.path)

        for i in self.image_path:
            if not i.endswith('.jpg'):
                self.image_path.remove(i)
        self.class_to_idx ={'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                            '+': 10, '-': 11, 'times': 12, 'div': 13, '(': 14, ')': 15, '=': 16,
                            'log': 17, 'sqrt': 18, 'sin': 19, 'cos': 20,
                            'pi': 21}
    def __getitem__(self, idx):
        image_name=self.image_path[idx]
        image_item_path=os.path.join(self.root_dir,self.label_dir,image_name)
        img=transforms(Image.open(image_item_path).convert('L'))
        label=self.label_dir
        return img,torch.tensor(self.class_to_idx[label])
    def __len__(self):
        return len(self.image_path)
class Data2(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir=root_dir
        self.transform=transform
        self.image_file=[os.path.join(root_dir,f) for f in os.listdir(root_dir)
                         if os.path.isfile(os.path.join(root_dir,f))]
    def __len__(self):
        return len(self.image_file)
    def __getitem__(self, idx):
        img_path=self.image_file[idx]
        image=Image.open(img_path).convert('RGB')
        if self.transform:
            image=self.transform(image)
        return image,0