import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(self, num_channels=1, upscale_factor=2, line_weight=[12,1]):
        super(CNN, self).__init__()
        self.k_num_channels=get_k_num_channels(line_weight)
        self.upscale=upscale_factor
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=5, padding=5 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=3 // 2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=3 // 2)
        self.conv4 = nn.Conv2d(32, self.k_num_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.PReLU()
        if self.upscale==2:
            self.Distribute1 = Distribute([-0.25, 0.25], line_weight)
            self.Distribute2 = Distribute([0.25, 0.25], line_weight)
            self.Distribute3 = Distribute([-0.25, -0.25], line_weight)
            self.Distribute4 = Distribute([0.25, -0.25], line_weight)
        elif self.upscale==3:
            self.Distribute1 = Distribute([-0.333, 0.333], line_weight)
            self.Distribute2 = Distribute([0.0, 0.333], line_weight)
            self.Distribute3 = Distribute([0.333, 0.333], line_weight)
            self.Distribute4 = Distribute([-0.333, 0.0], line_weight)
            self.Distribute5 = Distribute([0.0, 0.0], line_weight)
            self.Distribute6 = Distribute([0.333, 0.0], line_weight)
            self.Distribute7 = Distribute([-0.333, -0.333], line_weight)
            self.Distribute8 = Distribute([0.0, -0.333], line_weight)
            self.Distribute9 = Distribute([0.333, -0.333], line_weight)
        else:
            self.Distribute1 = Distribute([-0.375, 0.375], line_weight)
            self.Distribute2 = Distribute([-0.125, 0.375], line_weight)
            self.Distribute3 = Distribute([0.125,0.375], line_weight)
            self.Distribute4 = Distribute([0.375, 0.375], line_weight)
            self.Distribute5 = Distribute([-0.375, 0.125], line_weight)
            self.Distribute6 = Distribute([-0.125, 0.125], line_weight)
            self.Distribute7 = Distribute([0.125, 0.125], line_weight)
            self.Distribute8 = Distribute([0.375, 0.125], line_weight)
            self.Distribute9 = Distribute([-0.375, -0.125], line_weight)
            self.Distribute10 = Distribute([-0.125, -0.125], line_weight)
            self.Distribute11 = Distribute([0.125, -0.125], line_weight)
            self.Distribute12 = Distribute([0.375, -0.125], line_weight)
            self.Distribute13 = Distribute([-0.375, -0.375], line_weight)
            self.Distribute14 = Distribute([-0.125, -0.375], line_weight)
            self.Distribute15 = Distribute([0.125, -0.375], line_weight)
            self.Distribute16 = Distribute([0.375, -0.375], line_weight)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)


    def forward(self, input):
        x = self.relu(self.conv1(input))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        weight = self.conv4(x)
        if self.upscale==2:
            out1 = self.Distribute1(input, weight)
            out2 = self.Distribute2(input, weight)
            out3 = self.Distribute3(input, weight)
            out4 = self.Distribute4(input, weight)
            out = torch.cat([out1, out2, out3, out4],axis=1)
        elif self.upscale==3:
            out1 = self.Distribute1(input, weight)
            out2 = self.Distribute2(input, weight)
            out3 = self.Distribute3(input, weight)
            out4 = self.Distribute4(input, weight)
            out5 = self.Distribute5(input, weight)
            out6 = self.Distribute6(input, weight)
            out7 = self.Distribute7(input, weight)
            out8 = self.Distribute8(input, weight)
            out9 = self.Distribute9(input, weight)
            out = torch.cat([out1, out2, out3, out4,out5, out6, out7, out8, out9], axis=1)
        else:
            out1 = self.Distribute1(input, weight)
            out2 = self.Distribute2(input, weight)
            out3 = self.Distribute3(input, weight)
            out4 = self.Distribute4(input, weight)
            out5 = self.Distribute5(input, weight)
            out6 = self.Distribute6(input, weight)
            out7 = self.Distribute7(input, weight)
            out8 = self.Distribute8(input, weight)
            out9 = self.Distribute9(input, weight)
            out10 = self.Distribute10(input, weight)
            out11 = self.Distribute11(input, weight)
            out12 = self.Distribute12(input, weight)
            out13 = self.Distribute13(input, weight)
            out14 = self.Distribute14(input, weight)
            out15 = self.Distribute15(input, weight)
            out16 = self.Distribute16(input, weight)
            out = torch.cat([out1, out2, out3, out4, out5, out6, out7, out8, out9,out10,out11,out12,out13,out14,out15,out16], axis=1)
        out = self.pixel_shuffle(out)
        return out

def get_k_num_channels(line_weight):
    sum=1
    for i in range(len(line_weight)):
        if i==0:
            sum+=2*line_weight[0]
        else:
            sum+=line_weight[i-1]*line_weight[i]
    return sum



def func(W, N, coor):
    k_list = []
    for i in range(len(N)):
        if i == 0:
            k_list.append(N[i] * 2)
        else:
            k_list.append(N[i] * N[i-1])

    for i in range(len(k_list)):
        _temp_output_list = []
        u0 = 0
        u1 = 0
        if i == 0:
            w = W[:,0:k_list[i],:,:]
            delta = 2
            _input = coor.reshape(1,2,1,1)

        else:
            for j0 in range(1, i+1):
                u0 = u0 + k_list[j0 - 1]
                u1 = u0 + k_list[j0]
            w = W[:,u0:u1,:,:]
            delta = N[i-1]
            relu = nn.ReLU(inplace=True)
            _input = relu(_temp)

        for j in range(N[i]):
            wj = w[:,j*delta:(j+1)*delta,:,:]
            _temp_output = torch.mul(wj, _input)
            _temp_output = torch.sum(_temp_output, dim=1, keepdim=True)
            _temp_output_list.append(_temp_output)
        _temp = torch.cat(_temp_output_list ,1)
    return _temp.add_(W[:,-1:,:,:])


class Distribute(nn.Module):
    def __init__(self, coor, N):
        super(Distribute, self).__init__()
        coor = torch.Tensor(coor)
        coor = coor.cuda()
        self.coor = coor
        self.N = N

    def forward(self,input,weight):
        out = func(weight, self.N, self.coor)
        return out
