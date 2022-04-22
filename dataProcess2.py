import argparse
import cv2
from os import listdir

from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch
to_tensor=transforms.ToTensor()
to_img=transforms.ToPILImage()
device = torch.device("cuda:0")
class Net(nn.Module):
    def __init__(self,kernel=2):
        super(Net,self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel,stride=kernel)
    def forward(self,input):
        out = self.pool(input)
        return out
def train():
    lr_count=1
    hr_count=1
    lr_srcnn_count=1
    model = Net(args.kernel).to(device)
    for img_path in listdir(args.images_dir):
        hr=cv2.imread(args.images_dir+"/"+img_path)
        lr = cv2.GaussianBlur(hr,(3,3),1)

        lr = lr[:, :, (2, 1, 0)]
        lr=to_tensor(lr)
        lr = lr.to(device)
        with torch.no_grad():
            out = model(lr)
        lr = to_img(out)
        lr.save("data2/X{}/train/ns/{}".format(args.scale,img_path))

        lr = Image.open(args.images_dir+"/"+img_path)
        lr_srcnn = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=Image.BICUBIC)
        lr_srcnn.save("data2/X{}/train/lr_srcnn/{}".format(args.scale, img_path))
    print(hr_count)
    print(lr_count)
    print(lr_srcnn_count)

def eval():
    model = Net(args.kernel).to(device)
    for img_path in listdir(args.images_dir):
        hr = cv2.imread(args.images_dir + "/"+img_path)
        lr = cv2.GaussianBlur(hr, (3, 3), 1)
        lr = lr[:,:,(2,1,0)]
        lr = to_tensor(lr)
        lr = lr.to(device)
        with torch.no_grad():
            out = model(lr)
        lr = to_img(out)
        lr.save("data2/X{}/TestManga109/ns/".format(args.scale)+img_path)

        lr_srcnn = lr.resize((lr.width*args.scale,lr.height*args.scale),resample=Image.BICUBIC)
        lr_srcnn.save("data2/X{}/TestManga109/lr_srcnn/".format(args.scale)+img_path)

if __name__== "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--images_dir",type=str,required=True)
    parser.add_argument("--scale",type=int,default=3)
    parser.add_argument("--patch_size",type=int,default=192)
    parser.add_argument("--stride",type=int,default=192)
    parser.add_argument("--kernel",type=int,default=2)
    parser.add_argument("--eval",action="store_true")
    args=parser.parse_args()

    if not args.eval:
        train()
    else:
        eval()
