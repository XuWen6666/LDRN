import dataset,utils
import argparse
import numpy as np
from torch.utils.data import DataLoader
import torch
from loss import loss_func
from PIL import Image
import LDRN


# Training settings
parser = argparse.ArgumentParser(description='LDRN')
parser.add_argument("--line_weight",type=int,nargs='+',required=True,help="line weight")
parser.add_argument('--scale', type=int, required=True, help='bicubic scale')
parser.add_argument('--data_size', type=int, default=32, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true',help='use cuda?')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--training_log_count', type=int, default=10, help='the frequency of print in training stage')
parser.add_argument('--pre_train', type=str, default='', help='the path of pre-trained model')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda:0" if opt.cuda else "cpu")
print(device)

data_size = opt.data_size

print('===> Loading datasets')


#Loading dataset
train_set = dataset.DatasetFromFilelistSRTrain(opt.scale, data_size)
valid_set = dataset.DatasetFromFilelistSRValid(opt.scale, 'F:\\dataset\\Set5\\', '.bmp') #Set5测试集
# valid_set = dataset.DatasetFromFilelistSRValidSet14(opt.scale, 'F:\\dataset\\Set5\\', '.png') #Set14测试集
# valid_set = dataset.DatasetFromFilelistSRValidB100(opt.scale, 'F:\\dataset\\Set5\\', '.png') #B100测试集
# valid_set = dataset.DatasetFromFilelistSRValidUrban100(opt.scale, 'F:\\dataset\\Set5\\', '.png') #Urban100测试集
# valid_set = dataset.DatasetFromFilelistSRValidManga109(opt.scale, 'F:\\dataset\\Set5\\', '.png') #Manga109测试集


train_dataloader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True, num_workers=opt.threads)
valid_dataloader = DataLoader(dataset=valid_set, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = LDRN.CNN(upscale_factor=opt.scale,line_weight=opt.line_weight).to(device)
train_criterion = loss_func('mae')
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15,  threshold=5e-3,
                                                       threshold_mode='abs',  cooldown=0, min_lr=1e-6, eps=1e-8,verbose=True)

if opt.pre_train != '':
    ckpt = torch.load(opt.pre_train)
    model.load_state_dict(ckpt.state_dict())

def train():
    best_res = valid(False)
    model_head = utils.get_model_name('LDRN_X'+str(opt.scale))


    for epochs in range(opt.nEpochs):
        epoch_loss = 0
        for iteration, batch in enumerate(train_dataloader, 1):
            input, target = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = train_criterion(model(input), target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if iteration % opt.training_log_count == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epochs+1, iteration, len(train_dataloader), loss.item()), end='\r')
        avg_psnr = valid()
        scheduler.step(avg_psnr)
        if avg_psnr > best_res:
            best_res = avg_psnr
            check_point(model_head, best_res)
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epochs+1, epoch_loss / len(train_dataloader)))



def valid(write=False):
    avg_psnr = 0
    avg_ssim = 0
    count = 0
    with torch.no_grad():
        for batch in valid_dataloader:
            count = count + 1
            _input, target = batch[0].to(device), batch[1].to(device)

            prediction,weight = model(_input)

            psnr = utils.calc_psnr(prediction, target, opt.scale, 1)
            ssim = utils.calc_ssim(prediction,target)
            avg_psnr += psnr
            avg_ssim += ssim
            if write == True:
                _r = utils.pos_process(prediction)
                _r = Image.fromarray(np.uint8(_r))
                _r.save('test_images/'+str(count)+'.png')
    avg_psnr = avg_psnr / len(valid_dataloader)
    avg_ssim = avg_ssim / len(valid_dataloader)
    print("===> Avg. PSNR: {:.4f} dB, Avg. SSIM: {:.4f}".format(avg_psnr,avg_ssim))
    return avg_psnr

def check_point(head, bet_res):
    model_out_path = head + "bestpsnr_{}.pth".format(np.round(bet_res,2))
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    train()