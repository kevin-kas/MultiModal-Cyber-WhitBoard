object_dict={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                            10: '+', 11: '-', 12:'times', 13: 'div', 14: '(', 15: ')', 16: '=',
                            17:'log' ,18: 'sqrt', 19: 'sin', 20: 'cos',
                            21: 'pi'}
import torch
import torch.nn as nn
from PIL import Image
import torchvision
from Cut import Extractor

execator=Extractor('output_fig')
execator.get('samples/img4.png')
execator.exec()

model=torch.load("models/model25.pth",map_location=torch.device('cpu'),weights_only=False)
transform=torchvision.transforms.Compose(
	[
		torchvision.transforms.Resize((32,32)),
		torchvision.transforms.ToTensor()
	]
)
import os
list1=os.listdir('output_fig')
with torch.no_grad():
	for i in list1:
		print(i)
		path=os.path.join('output_fig',i)
		img=transform(Image.open(path).convert('L'))
		img_tran=torch.reshape(img,(1,1,32,32))
		pred=model(img_tran)
		print(object_dict[int(pred.argmax(1))])