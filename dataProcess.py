import argparse
from os import listdir

from torchvision import transforms
from PIL import Image

to_tensor=transforms.ToTensor()
to_img=transforms.ToPILImage()

def train():
    lr_count=1
    hr_count=1
    lr_srcnn_count=1
    for img_path in listdir(args.images_dir):
        hr=Image.open(args.images_dir+"/"+img_path)
        img_width=(hr.width // args.scale) * args.scale
        img_height=(hr.height // args.scale) * args.scale

        hr=hr.resize((img_width , img_height), resample=Image.BICUBIC)
        lr=hr.resize((img_width // args.scale , img_height // args.scale ), resample=Image.BICUBIC)
        lr_srcnn=lr.resize((img_width, img_height),resample=Image.BICUBIC)

        hr=to_tensor(hr)
        lr=to_tensor(lr)
        lr_srcnn=to_tensor(lr_srcnn)

        for i in range(0,hr.shape[1]-args.patch_size+1,args.stride):
            for j in range(0,hr.shape[2]-args.patch_size+1,args.stride):
                hr_sub=hr[:,i:i+args.patch_size,j:j+args.patch_size]
                hr_img=to_img(hr_sub)
                hr_img.save("data/X{}/train/gt/{}.png".format(args.scale,hr_count))
                hr_count=hr_count+1

                #裁剪srcnn所需的图片
                # lr_srcnn_sub=lr_srcnn[:,i:i+args.patch_size,j:j+args.patch_size]
                # lr_srcnn_img=to_img(lr_srcnn_sub)
                # lr_srcnn_img.save("data/X{}/train/lr_srcnn/{}.png".format(args.scale,lr_srcnn_count))
                # lr_srcnn_count = lr_srcnn_count + 1
        for i in range(0, lr.shape[1] - (args.patch_size // args.scale - 1), args.stride // args.scale):
            for j in range(0, lr.shape[2] - (args.patch_size // args.scale - 1), args.stride // args.scale):
                lr_sub = lr[:, i:i + args.patch_size // args.scale, j:j + args.patch_size // args.scale]
                lr_img = to_img(lr_sub)
                lr_img.save("data/X{}/train/ns/{}.png".format(args.scale, lr_count))
                lr_count=lr_count+1
    print(hr_count)
    print(lr_count)
    # print(lr_srcnn_count)

def eval():
    for img_path in listdir(args.images_dir):
        hr = Image.open(args.images_dir + "/"+img_path)
        img_width = (hr.width // args.scale) * args.scale
        img_height = (hr.height // args.scale) * args.scale

        hr = hr.resize((img_width, img_height), resample=Image.BICUBIC)
        lr = hr.resize((img_width // args.scale, img_height // args.scale), resample=Image.BICUBIC)
        lr_srcnn = lr.resize((img_width, img_height), resample=Image.BICUBIC)

        hr.save("data/X{}/TestManga109/gt/".format(args.scale)+img_path)
        lr.save("data/X{}/TestManga109/ns/".format(args.scale)+img_path)
        lr_srcnn.save("data/X{}/TestManga109/lr_srcnn/".format(args.scale)+img_path)


if __name__== "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--images_dir",type=str,required=True)
    parser.add_argument("--scale",type=int,default=3)
    parser.add_argument("--patch_size",type=int,default=192)
    parser.add_argument("--stride",type=int,default=192)
    parser.add_argument("--eval",action="store_true")
    args=parser.parse_args()

    if not args.eval:
        train()
    else:
        eval()
