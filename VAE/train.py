from .Data import Data
from .VAE import CVAE
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

device=torch.device('cpu')

def lossfunction(recon_x,x,mu,logvar):
    BCE=nn.functional.binary_cross_entropy(recon_x.view(-1,28*28),x.view(-1,28*28),reduction='sum')
    KLD=-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    return BCE+KLD

def train_cvae(model,train_loader,optimizer,epochs=100):
    model.train()
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
        print(f'Epoch: {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}')
    torch.save(model.state_dict(), 'cvae_model.pth')

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

train_datasets=[Data(root_dir='Train_data',label_dir=str(i)) for i in range(10)]
all_train=torch.utils.data.ConcatDataset(train_datasets)
dataLoader=DataLoader(all_train,batch_size=64,shuffle=True,drop_last=True)

model=CVAE(num_classes=10).to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)

train_cvae(model,dataLoader,optimizer,epochs=300)
