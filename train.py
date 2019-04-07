""" Defines the Trainer class which handles train/validation/validation_video
"""
import time
import torch
import itertools
import numpy as np
from utils import map
import gc
from utils import smooth
from PIL import Image


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(startlr, decay_rate, optimizer, epoch, ratio=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = startlr * (ratio ** (epoch // decay_rate))
    # lr = startlr * (0.1 ** (epoch // decay_rate))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def Two_stage_learning_rate(startlr, decay_rate, optimizer, epoch, ratio=0.1, threshold=2):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch <= threshold:
        # lr = startlr * (ratio ** (epoch // 1))
        lr = startlr * (0.8 ** (epoch // 1))  # resnet baseline
    else:
        lr = startlr * ((ratio) ** ((epoch-threshold) // decay_rate)) * 0.7
    # lr = startlr * (0.1 ** (epoch // decay_rate))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def key_out(epoch, bkstep, bk, key):
    key = key
    cur = sum(epoch >= np.array(bkstep))
    if cur > 0 :
        if cur > len(bk):
            cur = len(bk)
            key = bk[cur - 1]
        else:
            key = bk[cur - 1]
    return key


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = torch.zeros(*pred.shape)
    for i in range(correct.shape[0]):
        for j in range(correct.shape[1]):
            correct[i, j] = target[j, pred[i, j]] > 0.5

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def submission_file(ids, outputs, filename):
    """ write list of ids and outputs to filename"""
    with open(filename, 'w') as f:
        for vid, output in zip(ids, outputs):
            scores = ['{:g}'.format(x)
                      for x in output]
            f.write('{} {}\n'.format(vid, ' '.join(scores)))


class Trainer():
    def train(self, loader, model, criterion, optimizer, epoch, args):
        adjust_learning_rate(args.lr, args.lr_decay_step, optimizer, epoch, ratio=args.lr_decay_ratio)
        print('This batch lr:\t', optimizer.param_groups[0]['lr'])

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        model.train()
        criterion.train()
        optimizer.zero_grad()
        key = torch.autograd.Variable(torch.IntTensor([1]))

        def part(x): return itertools.islice(x, int(len(x)*args.train_size))
        data_loader = part(loader)
        end = time.time()
        for i, (input, target, meta, vid, Auxili_info, raw_test) in enumerate(data_loader):
            # Image._show(Image.fromarray(raw_test[0, :, :, :].numpy().astype(np.uint8)))
            gc.collect()
            data_time.update(time.time() - end)
            meta['epoch'] = epoch
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.float().cuda(async=True))
            Auxili_info = torch.autograd.Variable(Auxili_info)
            output = model(input_var, Auxili_info, key)
            loss = criterion(output, target_var)
            output = torch.nn.Sigmoid()(output)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()

            if i % int(0.05*args.train_size*len(loader)) == 0:
                print('Epoch: [{0}][{1}/{2}({3})]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          epoch, i, int(
                              len(loader)*args.train_size), len(loader),
                          batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5))
        return top1.avg, top5.avg, losses.avg

    def validate(self, loader, model, criterion, epoch, args):
        # with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()
        criterion.eval()
        key = torch.autograd.Variable(torch.IntTensor([2]))

        def part(x): return itertools.islice(x, int(len(x)*args.val_size))
        end = time.time()
        data_loader = part(loader)
        for i, (input, target, meta, vid, Auxili_info, raw_test)in enumerate(data_loader):
            gc.collect()
            meta['epoch'] = epoch
            target = target.long().cuda(async=True)
            input_var = torch.autograd.Variable(input.cuda(), volatile=True)
            target_var = torch.autograd.Variable(target.float().cuda(async=True))
            Auxili_info = torch.autograd.Variable(Auxili_info)
            output = model(input_var, Auxili_info, key)
            loss = criterion(output, target_var)
            output = torch.nn.Sigmoid()(output)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % int(0.1*args.val_size*len(loader)) == 0:
                print('Test: [{0}/{1} ({2})]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, int(len(loader)*args.val_size), len(loader),
                          batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        return top1.avg, top5.avg, losses.avg

    def validate_video(self, loader, model, criterion, epoch, args):
        """ Run video-level validation on the Charades test set"""
        # with torch.no_grad():
        batch_time = AverageMeter()
        outputs = []
        gts = []
        ids = []

        # switch to evaluate mode
        model.eval()
        criterion.eval()
        key = torch.autograd.Variable(torch.IntTensor([3]))

        end = time.time()
        kernelsize = 1
        print('applying smoothing with kernelsize {}'.format(kernelsize))

        for i, (input, target, meta, vid, Auxili_info, raw_test) in enumerate(loader):
            gc.collect()
            meta['epoch'] = epoch
            target = target.long().cuda(async=True)
            assert target[0,:].eq(target[1,:]).all(), "val_video not synced"
            input_var = torch.autograd.Variable(input.cuda(), volatile=True)
            target_var = torch.autograd.Variable(target.float().cuda(), volatile=True)
            Auxili_info = torch.autograd.Variable(Auxili_info)
            output = model(input_var, Auxili_info, key)
            # loss = criterion(output, target_var)

            # store predictions
            output = torch.nn.Sigmoid()(output)
            output = smooth.winsmooth(output, kernelsize=kernelsize)
            output_video = output.max(dim=0)[0]
            outputs.append(output_video.cpu().numpy())
            gts.append(target[0,:])
            ids.append(meta['id'][0])
            batch_time.update(time.time() - end)
            end = time.time()

            if i % int(len(loader)*0.07) == 0:
                print('Test2: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                          i, len(loader), batch_time=batch_time))

        mAP, _, ap = map.charades_map(np.vstack(outputs), np.vstack(gts))
        prec1, prec5 = accuracy(torch.Tensor(np.vstack(outputs)), torch.Tensor(np.vstack(gts)), topk=(1, 5))
        # print(ap)
        print(' * mAP {:.3f}'.format(mAP))
        print(' * prec1 {:.3f} * prec5 {:.3f}'.format(prec1[0], prec5[0]))
        return mAP, prec1[0], prec5[0]
