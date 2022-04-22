import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class loss_func(nn.Module):
    def __init__(self, args):
        super(loss_func, self).__init__()
        self.multi = False
        self.filtering_loss_list = ['sobel', 'laplace', 'a_sobel']
        self.args = args
        if isinstance(args, tuple) or isinstance(args, list):
            self.multi = True

    def filtering_loss(self, x, y, arg):
        def kernels_filtering(x, kernels):

            def _kernel_filtering(x, kernel):
                kernel = np.expand_dims(kernel, 0)
                kernel = np.expand_dims(kernel, 0)
                kernel = np.repeat(kernel, x.shape[1], 0)
                conv_kernel = torch.from_numpy(kernel.astype('float32')).cuda()
                padded = F.pad(x, (1,1,1,1), mode='reflect')
                conved = torch.conv2d(padded, conv_kernel, groups=x.shape[1])
                return torch.unsqueeze(conved, 1)

            res_list = []
            for i in range(kernels.shape[0]):
                res_list.append(_kernel_filtering(x,kernels[i]))
            return torch.cat(res_list, 1)

        if arg == 'a_sobel':
            kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
                        [[0, -1, -2], [1, 0, -1], [2, 1, 0]]]
            s = 0.25

        if arg == 'sobel':
            kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
            s = 0.5
        
        if arg == 'laplace':
            kernels = [[[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                        [[-1, 0, -1], [0, 4, 0], [-1, 0, -1]]]
            s = 0.5

        kernels = np.array(kernels)
        res_x = kernels_filtering(x, kernels)

        with torch.no_grad():
            res_y = kernels_filtering(y, kernels)

        for i in range(kernels.shape[0]):
            _loss = s*F.l1_loss(res_x[:,i,:,:,:], res_y[:,i,:,:,:]) if i==0 else _loss + s*F.l1_loss(res_x[:,i,:,:,:], res_y[:,i,:,:,:])
        return _loss

    def get_loss(self, x, y, arg):
        if arg in self.filtering_loss_list:
            return self.filtering_loss(x, y, arg)
        else:
            if arg == 'mae':
                return F.l1_loss(x[0],y)
            if arg == 'mse':
                return F.mse_loss(x[0],y)

    def forward(self, x, y):
        if self.multi:
            idx = 0
            for arg in self.args:
                _loss = self.get_loss(x,y,arg) if idx == 0 else _loss + self.get_loss(x,y,arg)
                idx += 1
        else:
            _loss = self.get_loss(x,y,self.args)

        return _loss
