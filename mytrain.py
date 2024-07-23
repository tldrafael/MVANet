import  os, argparse
from tqdm import tqdm
import torch
from datetime import datetime
from model.MVANet import MVANet
from utils.dataset_strategy_fpn import get_loader
from utils.misc import adjust_lr, AvgMeter
import torch.nn.functional as F
from torch.autograd import Variable
from torch.backends import cudnn
from torchvision import transforms
import torch.nn as nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
cudnn.benchmark = True
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
import torch.distributed as dist
import pandas as pd
import sys
sys.path.append('/home/rafael/workspace/PX-Matting-Training/')
import dataset as ds
import torchvision.transforms as T
import numpy as np
import cv2


get_gtpath = lambda x: x.replace('/im/', '/gt/').replace('.jpg', '.png')
cv2_imread = lambda x: cv2.imread(x, cv2.IMREAD_UNCHANGED)


T_mva = T.Compose([
    T.ToTensor(),
    T.Resize((1024,1024)),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict(p, generator, input_size=1024):
    im = cv2_imread(p)
    oldh, oldw = im.shape[:2]
    newim, (padh, padw) = ds.resize_im(
        im, long_length=input_size, fl_pad=True)

    input_ = T_mva(newim)[None].cuda()
    pred = generator(input_).sigmoid()

    pred = T.functional.resize(pred, (input_size, input_size))
    dim = input_size
    pred = pred[..., :(dim-padh), :(dim-padw)]
    pred = T.functional.resize(pred, (oldh, oldw))
    pred = pred[0, 0].cpu().numpy()
    pred = (pred * 255).astype('uint8')
    return pred


def evaluate_evalset_by_cat(generator):
    dfcat = pd.read_csv("/home/rafael/datasets/evalsets/evalset-multicat-v0.2-long2048/dfcat-for-training.csv")
    dfcat['sad'] = np.nan
    dfcat = dfcat.sample(10)

    for i, r in tqdm(dfcat.iterrows()):
        gtpath = get_gtpath(r.path)

        gt = cv2_imread(gtpath) / 255
        pred = predict(r.path, generator) / 255

        sad = (pred - gt).__abs__().sum()
        dfcat.loc[i, 'sad'] = sad / 1000

    dfcat['sadlog'] = dfcat['sad'].apply(np.log2)
    return dfcat.groupby('cat')['sadlog'].mean().mean()


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))


    pred  = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))

    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou  = 1-(inter+1)/(union-inter+1)

    return (wbce+wiou).mean()


from types import SimpleNamespace
opt = SimpleNamespace()
opt.epoch = 80
opt.lr_gen = 1e-5
opt.batchsize = 1
opt.trainsize = 1024
opt.decay_rate = 0.9
opt.decay_epoch = 80


def train():
    n_gpus = torch.cuda.device_count()
    rank = dist.get_rank()
    gpu_id = rank % n_gpus
    print(f'gpu_id: {gpu_id}, rank: {rank}, n_gpus: {n_gpus}')

    if gpu_id == 0:
        time_now = datetime.now().strftime("%Y%m%d_%H%M")
        ckpt_dir = f'runs/{time_now}+test1'
        os.makedirs(ckpt_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=ckpt_dir)


    print('Generator Learning Rate: {}'.format(opt.lr_gen))
    # build models
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    generator = MVANet(gpu_id=gpu_id)
    generator.to(gpu_id)
    generator = DDP(generator, device_ids=[gpu_id])

    generator_params = generator.parameters()
    generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)

    image_root = './data/DIS5K/DIS-TR/images/'
    gt_root = './data/DIS5K/DIS-TR/masks/'
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)
    to_pil = transforms.ToPILImage()

    CE = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
    size_rates = [1]
    criterion = nn.BCEWithLogitsLoss().to(gpu_id)
    criterion_mae = nn.L1Loss().to(gpu_id)
    criterion_mse = nn.MSELoss().to(gpu_id)
    use_fp16 = True
    scaler = amp.GradScaler(enabled=use_fp16)

    for epoch in range(1, opt.epoch+1):
        torch.cuda.empty_cache()
        generator.train()
        i = 1
        pbar = tqdm(train_loader, disable=gpu_id != 0)
        losses = []
        for pack in pbar:
            for rate in size_rates:
                generator_optimizer.zero_grad()
                images, gts = pack
                images = Variable(images)
                gts = Variable(gts)
                images = images.to(gpu_id)
                gts = gts.to(gpu_id)
                trainsize = int(round(opt.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear',
                                              align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                b, c, h, w = gts.size()
                target_1 = F.upsample(gts, size=h // 4, mode='nearest')
                target_2 = F.upsample(gts,  size=h // 8, mode='nearest').to(gpu_id)
                target_3 = F.upsample(gts,  size=h // 16, mode='nearest').to(gpu_id)
                target_4 = F.upsample(gts,  size=h // 32, mode='nearest').to(gpu_id)
                target_5 = F.upsample(gts, size=h // 64, mode='nearest').to(gpu_id)

                with amp.autocast(enabled=use_fp16):
                    sideout5, sideout4, sideout3, sideout2, sideout1, final, glb5, glb4, glb3, glb2, glb1, tokenattmap4, tokenattmap3, tokenattmap2, tokenattmap1 = generator.forward(images)
                    loss1 = structure_loss(sideout5, target_4)
                    loss2 = structure_loss(sideout4, target_3)
                    loss3 = structure_loss(sideout3, target_2)
                    loss4 = structure_loss(sideout2, target_1)
                    loss5 = structure_loss(sideout1, target_1)
                    loss6 = structure_loss(final, gts)
                    loss7 = structure_loss(glb5, target_5)
                    loss8 = structure_loss(glb4, target_4)
                    loss9 = structure_loss(glb3, target_3)
                    loss10 = structure_loss(glb2, target_2)
                    loss11 = structure_loss(glb1, target_2)
                    loss12 = structure_loss(tokenattmap4, target_3)
                    loss13 = structure_loss(tokenattmap3, target_2)
                    loss14 = structure_loss(tokenattmap2, target_1)
                    loss15 = structure_loss(tokenattmap1, target_1)
                    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + 0.3*(loss7 + loss8 + loss9 + loss10 + loss11)+ 0.3*(loss12 + loss13 + loss14 + loss15)
                    Loss_loc = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                    Loss_glb = loss7 + loss8 + loss9 + loss10 + loss11
                    Loss_map = loss12 + loss13 + loss14 + loss15

                generator_optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(generator_optimizer)
                scaler.update()
                losses.append(loss.item())

            i += 1
            if i % 10 == 0 or i == total_step and gpu_id == 0:
                pbar.set_description('Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, opt.epoch, losses[-1]))

            if i % 10 == 0:
                break
                torch.cuda.empty_cache()

        # adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
        if gpu_id == 0:
            with torch.no_grad():
                generator.eval()
                sadlog = evaluate_evalset_by_cat(generator)
                generator.train()

            writer.add_scalar('train_loss', np.mean(losses), epoch)
            writer.add_scalar('sadlog', sadlog, epoch)
            print(f'Epoch: {epoch}, Loss: {np.mean(losses):.3f}, sadlog: {sadlog:.2f}')

            savepath = os.path.join(ckpt_dir, f'Model_{epoch}.pth')
            print(f'Saving Model to {savepath}')
            torch.save(generator.state_dict(), savepath)


@record
def torchrun_startup():
    dist.init_process_group(backend="nccl")
    train()
    dist.destroy_process_group()


if __name__ == '__main__':
    torchrun_startup()
