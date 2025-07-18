import os
import torch
from Recognize_Model import Model_re
from torch.utils.data import DataLoader
import torch.nn as nn

train_root_list=['test_data','train_data']
label_root_list=symbol=[
        '0','1','2','3','4','5','6','7','8','9',
        '+','-','times','div','(',')','=',
        'log','sqrt','sin','cos',
        'pi'
    ]
from Data import Data
train_list=[Data(train_root_list[1],i) for i in label_root_list]
test_list=[Data(train_root_list[0],i) for i in label_root_list]

all_train_data=None
for i in train_list:
    if all_train_data is None:
        all_train_data=i
    else:
        all_train_data+=i
all_test_data=None
for i in test_list:
    if all_test_data is None:
        all_test_data=i
    else:
        all_test_data+=i
train_data_loader=DataLoader(all_train_data,batch_size=64,shuffle=True,drop_last=True)
test_data_loader=DataLoader(all_test_data,batch_size=64,shuffle=True,drop_last=True)
train_size=len(all_train_data)
test_size=len(all_test_data)
if os.path.exists('models')==False:
    os.makedirs('models')
model=Model_re()
if torch.cuda.is_available():
    model=model.cuda()

loss_fn=nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn=loss_fn.cuda()

learning_rate=0.001
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

total_train_step=0
total_test_step=0
epoch=25
acc_list=[]
for i in range(epoch):
    print(f"-------第{i+1}轮训练开始")
    model.train()

    for data in train_data_loader:
        img, target = data
        if torch.cuda.is_available():
            img = img.cuda()
            target = target.cuda()
        output = model(img)
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step+=1
    model.eval()
    total_test_loss=0
    total_accuracy=0

    with torch.no_grad():
        for data in test_data_loader:
            img,target=data
            if torch.cuda.is_available():
                img=img.cuda()
                target=target.cuda()
            output=model(img)
            loss=loss_fn(output,target)
            total_test_loss+=loss
            accuracy=(output.argmax(1)==target).sum()
            total_accuracy+=accuracy
    print(f"The total loss on the test set{total_test_loss}")
    print(f"The total accuracy on the test set{total_accuracy / test_size}")
    torch.save(model.state_dict(), f"models/model{i + 1}.pth")
    print("model have saved")
    #import the early stopping
    acc_list.append(total_accuracy/test_size)