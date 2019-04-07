import torch
import numpy as np
import sys
import glob
sys.path.append('/home/thl/Desktop/3D/')
from data import class_reduce
import time
import os
import gc
from PIL import Image
import torch.backends.cudnn as cudnn
import itertools
import opts
from models import create_model
from utils import checkpoints
from utils import NetFunction
from utils import map
from utils import smooth
cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datasets import get_dataset,get_dataset_train, get_dataset_val, get_dataset_video
from train import AverageMeter, accuracy, submission_file

first_model_dir = '../model/01_model.pth'  # all the model are same
model_num_all = 20


def val_fm(model, model_dir, criterion,dataset, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    trained_model = torch.load(model_dir)
    model.load_state_dict(trained_model)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    criterion.eval()

    def part(x):
        return itertools.islice(x, int(len(x) * args.val_size))
    data_loader = part(dataset)
    for i, (input, target, meta, vid, Auxili_info, raw_test) in enumerate(data_loader):
        gc.collect()
        # meta['epoch'] = epoch
        target = target.long().cuda(async=True)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target.float().cuda(async=True))
        Auxili_info = torch.autograd.Variable(Auxili_info)
        output = model(input_var, Auxili_info)
        loss = criterion(output, target_var)
        output = torch.nn.Sigmoid()(output)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        if i % int(0.1 * args.val_size * len(dataset)) == 0:
            print('Test: [{0}/{1} ({2})]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, int(len(dataset) * args.val_size), len(dataset),
                loss=losses,top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def validate_video(model, model_dir, criterion, loader, args):
    # with torch.no_grad():
    outputs = []
    gts = []
    ids = []

    trained_model = torch.load(model_dir)
    model.load_state_dict(trained_model)
    model = torch.nn.DataParallel(model).cuda()
    # switch to evaluate mode
    model.eval()
    criterion.eval()

    kernelsize = 1
    print('applying smoothing with kernelsize {}'.format(kernelsize))
    for i, (input, target, meta, vid, Auxili_info, raw_test) in enumerate(loader):
        gc.collect()
        # meta['epoch'] = epoch
        target = target.long().cuda(async=True)
        assert target[0, :].eq(target[1, :]).all(), "val_video not synced"
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target.float().cuda(), volatile=True)
        Auxili_info = torch.autograd.Variable(Auxili_info)
        output = model(input_var, Auxili_info)

        # store predictions
        output = torch.nn.Sigmoid()(output)
        output = smooth.winsmooth(output, kernelsize=kernelsize)
        output_video = output.max(dim=0)[0]
        outputs.append(output_video.cpu().numpy())
        gts.append(target[0, :])
        ids.append(meta['id'][0])
        if i % int(len(loader) * 0.07) == 0:
            print('Test2: [{0}/{1}]\t'.format(i, len(loader)))

    mAP, _, ap = map.charades_map(np.vstack(outputs), np.vstack(gts))
    prec1, prec5 = accuracy(torch.Tensor(np.vstack(outputs)), torch.Tensor(np.vstack(gts)), topk=(1, 5))
    # print(ap)
    print(' * mAP {:.3f}'.format(mAP))
    print(' * prec1 {:.3f} * prec5 {:.3f}'.format(prec1[0], prec5[0]))
    return mAP, prec1[0], prec5[0]


def write_log(epoch, info, dir):
    epoch_info = 'epoch:\t ' + str(epoch+1) + '\n'
    val_info = 'top1val\t' + str(info['top1val']) + '\ttop5val\t' + str(info['top5val']) + '\tval_loss\t' + str(info['val_loss']) + '\n'
    map_info = 'mAP\t' + str(info['mAP']) + '\tvideoprec1\t' + str(info['videoprec1']) + '\tvideoprec5\t' + str(info['videoprec5']) + '\n'
    end = '====================************============================ \n'
    file = epoch_info + val_info + map_info + end
    if not os.path.exists(dir):
        with open(dir, 'w') as f:
            f.write(file)
    else:
        with open(dir, 'a') as f:
            f.write(file)


if __name__ == '__main__':
    while not os.path.exists(first_model_dir):
        time.sleep(30)
        print('The first model not generate wait to val and get mAP file')

    raw_num, part_num, Original_list, Part_class_list = class_reduce.Class_num()
    global opt
    opt = opts.parse(part_num)
    opt.vdlist, opt.BBOX_dir = opts.vd_list('val')

    model,_ = create_model(opt, val=1)  # model OK

    # loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    # model = torch.nn.DataParallel(model).cuda()

    val_loader = get_dataset_val(opt, Part_class_list)
    video_loader = get_dataset_video(opt, Part_class_list)

    for model_No in range(1, model_num_all):
        model_name = '%02d_' % (model_No) + 'model.pth'
        model_dir = '../model/' + model_name
        while not os.path.exists(model_dir):
            time.sleep(30)
        top1val, top5val, val_loss = val_fm(model, model_dir, criterion, val_loader, opt)
        mAP, prec1, prec5 = validate_video(model, model_dir, criterion, video_loader, opt)
        scores = {'top1val': round(top1val, 3), 'top5val': round(top5val, 3), 'val_loss': round(val_loss, 3),
                  'mAP': round(mAP, 3), 'videoprec1': round(prec1, 3), 'videoprec5': round(prec5, 3)}
        write_log(model_No, scores, '../val.txt')
