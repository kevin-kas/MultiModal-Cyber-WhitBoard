import warnings
warnings.filterwarnings("ignore")
from .Fix import Lexer,hss
import torch
from PIL import Image
import torchvision
from .Cut import Extractor

def build(input_filedir,model_path):
    lexer=Lexer()

    object_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                   10: '+', 11: '-', 12: 'times', 13: 'div', 14: '(', 15: ')', 16: '=',
                   17: 'log', 18: 'sqrt', 19: 'sin', 20: 'cos',
                   21: 'pi'}

    pred_res_list=[]
    target_file="output_fig"
    exactor=Extractor(target_file)
    sample_file=input_filedir
    exactor.get(sample_file)
    extracted=exactor.exec()

    from .Recognize_Model import Model_re
    model=Model_re()
    model.load_state_dict(torch.load(model_path))
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor()
        ]
    )
    import os
    list1=os.listdir(target_file)
    with torch.no_grad():
        for i in list1:
            path=os.path.join(target_file,i)
            img=transform(Image.open(path).convert('L'))
            img_tran=torch.reshape(img,(1,1,32,32))
            pred=model(img_tran)
            pred_res_list.append(object_dict[int(pred.argmax(1))])

    pos_label_list=[]
    for i in range(len(extracted)):
        pos_label_list.append((extracted[i][0],pred_res_list[i]))

    tokens=lexer.lex(pos_label_list)
    tokens=sorted(tokens,key=lambda x:(x.x,x.y))
    string1=hss(tokens)
    return string1