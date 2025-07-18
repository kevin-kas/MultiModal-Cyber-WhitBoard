from PIL import Image
import cv2
import numpy as np
import os
class Extractor:
    def __init__(self,output_path):
        self.img=None
        self.output=output_path
        self.target_size=45

    def get(self,filename):
        self.img=Image.open(filename)
        return self.img

    def exec(self):
        pim=self.img
        img=cv2.cvtColor(np.array(pim),cv2.COLOR_RGB2BGR)
        res=img.copy()
        img=cv2.bitwise_not(img)
        _,thresh=cv2.threshold(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),127,255,0)
        contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        extracted=[]

        if os.path.exists(f'{self.output}')==False:
            os.makedirs(f'{self.output}')
        for i in range(len(contours)):
            x,y,w,h=cv2.boundingRect(contours[i])
            mask=np.zeros(img.shape,np.uint8)
            cv2.drawContours(mask,contours,i,(255,255,255),thickness=cv2.FILLED)
            out=cv2.bitwise_not(cv2.subtract(mask,res))[y:y+h,x:x+w]
            if out.size == 0:
                continue
            background = np.full((self.target_size, self.target_size, 3), (255, 255, 255), dtype=np.uint8)
            original_h, original_w = out.shape[:2]
            ratio = min(
                self.target_size / original_h,
                self.target_size / original_w
            )
            new_w = int(original_w * ratio)
            new_h = int(original_h * ratio)
            resized = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_AREA)
            x_offset = (self.target_size - new_w) // 2
            y_offset = (self.target_size - new_h) // 2

            # Put the resized image in the center of the background
            try:
                background[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            except ValueError:
                # Handle possible boundary errors
                new_h = min(new_h, self.target_size - y_offset)
                new_w = min(new_w, self.target_size - x_offset)
                background[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized[:new_h, :new_w]

            output_path = os.path.join(self.output, f"extracted_{i}.png")
            if w<10 and h<10:
                continue
            cv2.imwrite(output_path, background)
            extracted.append(((x, y, w, h), output_path))
        return extracted