import numpy as np
import math, os
from skimage.metrics import structural_similarity as compare_ssim

def calc_ssim(output,target):
    output=output.reshape(output.shape[2],output.shape[3],-1)
    output=output.cpu()
    output = output.detach().numpy()
    target=target.reshape(target.shape[2],target.shape[3],-1)
    target = target.cpu()
    target = target.detach().numpy()
    ssim=compare_ssim(output,target,multichannel=True)
    return ssim

def calc_psnr(sr, hr, scale, rgb_range, div2k=False):
    diff = (sr - hr).data.div(rgb_range)
    if div2k:
        shave = scale + 6
    else:
        shave = 0#scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 0.256789*256
            convert[0, 1, 0, 0] = 0.504129*256
            convert[0, 2, 0, 0] = 0.097906*256
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    '''
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    '''
    if shave == 0:
        valid = diff
    else:
        valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def list_filter(file_list, tail):
    r = []
    for f in file_list:
        s = os.path.splitext(f)
        if s[1] == tail:
            r.append(f)
    return r

def pre_process(x):
    if (len(x.shape)==3):
        if x.shape[2] == 3:
            x = x.astype(np.float32)
            x = 16. + (64.738*x[:,:,2]+129.057*x[:,:,1]+25.064*x[:,:,0])/256.
    if(len(x.shape)==2):
        x = x.reshape(x.shape+(1,))
    x = x.astype(np.float32)/255.0
    y = x.transpose(2,0,1)
    return y

def pos_process(x):
    if len(x.shape) == 4:
        x = x[0]
    
    t = x.cpu().numpy().transpose(1,2,0)
    t = np.round(t*255)
    t[t>255]=255
    t[t<0]=0
    return t

def get_model_name(task):
    s = 'models/'
    s = s + task + '_'
    return s

def generate_anchors(shape, size):
   def process_line(l, size):
      n = math.ceil(l/size)
      step = math.ceil(l/n)
      pos = 0
      pos_list = []
      while(pos < l):
         if (pos+size) <= l:
            pos_list.append(pos)
         else:
            pos_list.append(l-size)
         pos += step
      return pos_list
   h = shape[0]
   w = shape[1]
   pos_list = []
   h_list = process_line(h, size)
   w_list = process_line(w, size)
   for i in range(len(h_list)):
      for j in range(len(w_list)):
         pos_list.append((h_list[i], w_list[j]))
   return pos_list

def transform_npy(npy_path, data_size):
    npy = np.load(npy_path)
    npy = npy.transpose(0,3,1,2)
    pos_list = generate_anchors((npy.shape[2],npy.shape[3]), data_size)
    npa_list = []
    for i in range(npy.shape[0]):
        npa = npy[i]
        for pos in pos_list:
            _npa = npa[:,pos[0]:pos[0]+data_size, pos[1]:pos[1]+data_size]
            npa_list.append(_npa)
    new_npy = np.array(npa_list)
    print(new_npy.shape)
    return new_npy

def WriteProcess(version,str):
    with open("Process/"+version+".txt",'a',encoding='UTF-8') as f:
        f.write(str)
        f.close()

def gasuss_noise(image, mean=0, var=0.001):
    image = np.array(image/255, dtype=float)#将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
    noise = np.random.normal(mean, var / 255, image.shape)#创建一个均值为mean，方差为var呈高斯分布的图像矩阵
    out = image + noise#将噪声和原始图像进行相加得到加噪后的图像
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)#clip函数将元素的大小限制在了low_clip和1之间了，小于的用low_clip代替，大于1的用1代替
    out = np.uint8(out*255)#解除归一化，乘以255将加噪后的图像的像素值恢复
    return out