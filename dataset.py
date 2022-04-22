import torch.utils.data as data
import cv2
import torch
import os
import utils
import random
class DatasetFromFilelistSRTrain(data.Dataset):
    def __init__(self, scale, data_size):
        super(DatasetFromFilelistSRTrain, self).__init__()
        self.path = 'data/X'+str(scale)+'/train/'
        self.fl = os.listdir(self.path+'gt/')
        self.data_size = data_size
        self.scale = scale

    def __getitem__(self, index):
        _name = self.fl[index]
        valid_img = cv2.imread(self.path+'ns/' + _name) #低分辨率的图
        gt_img = cv2.imread(self.path+'gt/' + _name)  #高分辨率的图

        crop_x = random.randint(0, valid_img.shape[0] - self.data_size) #随机生成一个整数
        crop_y = random.randint(0, valid_img.shape[1] - self.data_size)


        valid_img = valid_img[crop_x:crop_x + self.data_size,crop_y:crop_y + self.data_size]
        gt_img = gt_img[self.scale * crop_x:self.scale * (crop_x + self.data_size),crop_y * self.scale:self.scale * (crop_y + self.data_size)]

        
        valid_img = utils.pre_process(valid_img) #valid_img表示输入图片,转化成y通道的图
        gt_img = utils.pre_process(gt_img) #gt_img表示目标图片


        _valid = torch.from_numpy(valid_img)
        _gt = torch.from_numpy(gt_img)
        return _valid, _gt

    def __len__(self):
        return len(self.fl)

class DatasetFromFilelistSRValid(data.Dataset):
    def __init__(self, scale, valid_path, tail):
        super(DatasetFromFilelistSRValid, self).__init__()
        self.vp="data/X{}/test/".format(scale)
        fl = os.listdir(self.vp+"gt/")
        self.fl = utils.list_filter(fl, tail) #为Set5图片的文件名
        self.s = scale
        self.tail = tail

    def __getitem__(self, index):
        gt_name = self.fl[index]
        ns_name = gt_name[0:len(gt_name)-4]+'.bmp'
        valid_img = cv2.imread(self.vp+'ns/' + ns_name)
        gt_img = cv2.imread(self.vp +'gt/'+ gt_name)
        w = gt_img.shape[1]
        h = gt_img.shape[0]
        w = w - w % self.s
        h = h - h % self.s
        gt_img = gt_img[0:h,0:w] #将边缘像素点裁掉

        valid_img = utils.pre_process(valid_img)
        gt_img = utils.pre_process(gt_img)

        _valid = torch.from_numpy(valid_img)
        _gt = torch.from_numpy(gt_img)
        return _valid, _gt

    def __len__(self):
        return len(self.fl)


class DatasetFromFilelistSRValidSet14(data.Dataset):
    def __init__(self, scale, valid_path, tail):
        super(DatasetFromFilelistSRValidSet14, self).__init__()
        self.vp = "data/X{}/TestSet14/".format(scale)
        fl = os.listdir(self.vp + "gt/")
        self.fl = utils.list_filter(fl, tail)
        self.s = scale
        self.tail = tail

    def __getitem__(self, index):
        gt_name = self.fl[index]
        ns_name = gt_name[0:len(gt_name) - 4] + '.png'
        valid_img = cv2.imread(self.vp + 'ns/' + ns_name)
        gt_img = cv2.imread(self.vp + 'gt/' + gt_name)
        w = gt_img.shape[1]
        h = gt_img.shape[0]
        w = w - w % self.s
        h = h - h % self.s
        gt_img = gt_img[0:h, 0:w]

        valid_img = utils.pre_process(valid_img)
        gt_img = utils.pre_process(gt_img)

        _valid = torch.from_numpy(valid_img)
        _gt = torch.from_numpy(gt_img)
        return _valid, _gt

    def __len__(self):
        return len(self.fl)


class DatasetFromFilelistSRValidB100(data.Dataset):
    def __init__(self, scale, valid_path, tail):
        super(DatasetFromFilelistSRValidB100, self).__init__()
        self.vp = "data/X{}/TestB100/".format(scale)
        fl = os.listdir(self.vp + "gt/")
        self.fl = utils.list_filter(fl, tail)  # 为Set5图片的文件名
        self.s = scale
        self.tail = tail

    def __getitem__(self, index):
        gt_name = self.fl[index]
        ns_name = gt_name[0:len(gt_name) - 4] + '.png'
        valid_img = cv2.imread(self.vp + 'ns/' + ns_name)  # .convert('L')
        gt_img = cv2.imread(self.vp + 'gt/' + gt_name)  # .convert('L')
        w = gt_img.shape[1]
        h = gt_img.shape[0]
        w = w - w % self.s
        h = h - h % self.s
        gt_img = gt_img[0:h, 0:w]  # 将边缘像素点裁掉

        valid_img = utils.pre_process(valid_img)
        gt_img = utils.pre_process(gt_img)

        _valid = torch.from_numpy(valid_img)
        _gt = torch.from_numpy(gt_img)
        return _valid, _gt

    def __len__(self):
        return len(self.fl)


class DatasetFromFilelistSRValidUrban100(data.Dataset):
    def __init__(self, scale, valid_path, tail):
        super(DatasetFromFilelistSRValidUrban100, self).__init__()
        self.vp = "data/X{}/TestUrban100/".format(scale)
        fl = os.listdir(self.vp + "gt/")
        self.fl = utils.list_filter(fl, tail)  # 为Set5图片的文件名
        self.s = scale
        self.tail = tail

    def __getitem__(self, index):
        gt_name = self.fl[index]
        ns_name = gt_name[0:len(gt_name) - 4] + '.png'
        valid_img = cv2.imread(self.vp + 'ns/' + ns_name)
        gt_img = cv2.imread(self.vp + 'gt/' + gt_name)
        w = gt_img.shape[1]
        h = gt_img.shape[0]
        w = w - w % self.s
        h = h - h % self.s
        gt_img = gt_img[0:h, 0:w]  # 将边缘像素点裁掉

        valid_img = utils.pre_process(valid_img)
        gt_img = utils.pre_process(gt_img)

        _valid = torch.from_numpy(valid_img)
        _gt = torch.from_numpy(gt_img)
        return _valid, _gt

    def __len__(self):
        return len(self.fl)


class DatasetFromFilelistSRValidManga109(data.Dataset):
    def __init__(self, scale, valid_path, tail):
        super(DatasetFromFilelistSRValidManga109, self).__init__()

        self.vp = "data/X{}/TestManga109/".format(scale)

        fl = os.listdir(self.vp + "gt/")
        self.fl = utils.list_filter(fl, tail)  # 为Set5图片的文件名
        self.s = scale
        self.tail = tail

    def __getitem__(self, index):
        gt_name = self.fl[index]
        ns_name = gt_name[0:len(gt_name) - 4] + '.png'
        valid_img = cv2.imread(self.vp + 'ns/' + ns_name)  # .convert('L')
        gt_img = cv2.imread(self.vp + 'gt/' + gt_name)  # .convert('L')
        w = gt_img.shape[1]
        h = gt_img.shape[0]
        w = w - w % self.s
        h = h - h % self.s
        gt_img = gt_img[0:h, 0:w]  # 将边缘像素点裁掉



        valid_img = utils.pre_process(valid_img)
        gt_img = utils.pre_process(gt_img)

        _valid = torch.from_numpy(valid_img)
        _gt = torch.from_numpy(gt_img)
        return _valid, _gt

    def __len__(self):
        return len(self.fl)