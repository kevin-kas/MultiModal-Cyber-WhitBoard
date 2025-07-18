import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import os

transforms1=torchvision.transforms.Compose(
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
        img=transforms1(Image.open(image_item_path).convert('L'))
        return img,torch.tensor(self.class_to_idx[image_name.strip('.png').split('_')[-1]])

    def __len__(self):
        return len(self.image_path)

import torch
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self,num_classes,latent_dim=20):
        super(CVAE,self).__init__()
        self.conv1=nn.Conv2d(1,32,kernel_size=4,stride=2,padding=1)
        self.conv2=nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1)
        self.conv3=nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)
        self.conv4=nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)

        self.embed_c=nn.Embedding(num_classes,128)
        self.fc1=nn.Linear(256*2*2+128,256)

        self.fc_mu=nn.Linear(256,latent_dim)
        self.fc_logvar=nn.Linear(256,latent_dim)

        self.fc2=nn.Linear(latent_dim+128,1024)
        self.deconv1=nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1)
        self.deconv2=nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1)
        self.deconv3=nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1)
        self.deconv4=nn.ConvTranspose2d(32,1,kernel_size=4,stride=2,padding=1)

    def encoder(self,x,c):
        c=c.long()
        c_embed=self.embed_c(c)
        x=torch.relu(self.conv1(x))
        x=torch.relu(self.conv2(x))
        x=torch.relu(self.conv3(x))
        x=torch.relu(self.conv4(x))
        x=x.view(x.size(0),-1)
        x=torch.cat([x,c_embed],dim=1)
        x=torch.relu(self.fc1(x))
        return self.fc_mu(x),self.fc_logvar(x)

    def decoder(self,z,c):
        c=c.long()
        c_embed=self.embed_c(c)
        z=torch.cat([z,c_embed],dim=1)
        z=torch.relu(self.fc2(z))
        z=z.view(z.size(0),256,2,2)
        z=torch.relu(self.deconv1(z))
        z=torch.relu(self.deconv2(z))
        z=torch.relu(self.deconv3(z))
        z=torch.sigmoid(self.deconv4(z))
        return z

    def reparameterize(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        return mu+eps*std

    def forward(self,x,c):
        mu,logvar=self.encoder(x,c)
        z=self.reparameterize(mu,logvar)
        return self.decoder(z,c),mu,logvar

import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def lossfunction(recon_x,x,mu,logvar):
    BCE=nn.functional.binary_cross_entropy(recon_x.view(-1,28*28),x.view(-1,28*28),reduction='sum')
    KLD=-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    return BCE+KLD

def train_cvae(model,train_loader,optimizer,num,epochs=100):
    model.train()
    all_train_loss=[]
    for epoch in range(epochs):
        train_loss=0
        for batch_idx,(data,labels) in enumerate(train_loader):
            data,labels=data.to(device),labels.to(device)
            optimizer.zero_grad()
            recon_batch,mu,logvar=model(data,labels)
            loss=lossfunction(recon_batch,data,mu,logvar)
            loss.backward()
            train_loss+=loss.item()
            optimizer.step()
        if (epoch+1)%20==0:
            print(f'Epoch: {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}')
        all_train_loss.append(train_loss / len(train_loader.dataset))
    torch.save(model.state_dict(), f'models/cvae_model{num}.pth')
    return model.state_dict(),sum(all_train_loss)/len(all_train_loss)

def generate_digit(model,digit_string):
    model.eval()
    generated_images=[]
    for digit in digit_string:
        digit=int(digit)
        label=torch.tensor([digit],dtype=torch.long).to(device)
        z=torch.randn(1,20).to(device)
        with torch.no_grad():
            generated_image=model.decoder(z,label).view(1,1,28,28)
            generated_images.append(generated_image)
    generated_images=torch.cat(generated_images,dim=0)
    return generated_images

train_datasets=[Data(root_dir='Number',label_dir=str(i)) for i in range(10)]
all_train=torch.utils.data.ConcatDataset(train_datasets)

param_grid={'lr':[1e-3,1e-4,1e-5],'batch_size':[64,128]}
num1=1
best_loss=float('inf')
best_model=None
best_lr=0
best_batch_size=0
for lr in param_grid['lr']:
    for batch_size in param_grid['batch_size']:
        dataLoader = DataLoader(all_train, batch_size=batch_size, shuffle=True, drop_last=True)
        model=CVAE(num_classes=10).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        print(f'lr={lr}, batch_size={batch_size}')
        model1,avg_loss=train_cvae(model,dataLoader,optimizer,num1,epochs=300)
        num1+=1
        if avg_loss<best_loss:
            best_loss=avg_loss
            best_model=model1
            best_lr=lr
            best_batch_size=batch_size
torch.save(best_model, 'cvae_best_model.pth')
print(f'Best model parameters: lr={best_lr}, batch_size={best_batch_size}, loss={best_loss}')
