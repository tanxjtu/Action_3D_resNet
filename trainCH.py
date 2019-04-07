import torch
import numpy as np
from data import class_reduce
import time
import os
import gc
from PIL import Image
import torch.backends.cudnn as cudnn
import itertools
from utils import opts
from models import create_model
from utils import checkpoints
from utils import NetFunction
cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
from datasets import get_dataset,get_dataset_train
from train import AverageMeter, adjust_learning_rate, accuracy, submission_file, Two_stage_learning_rate, key_out


if __name__ == '__main__':
    raw_num, part_num, Original_list, Part_class_list = class_reduce.Class_num()
    global opt, best_mAP
    opt = opts.parse(part_num)
    opt.vdlist, opt.BBOX_dir = opts.vd_list('train')

    version = 4  # change every time
    Tr_log_dir, Md_dir = checkpoints.Make_ProjDir(opt, version)

    model, two_stage = create_model(opt)  # model OK

    # Net fixed ---------
    NetFunction.lock_Bn(model, keys=7, Exc=1)
    NetFunction.Fix_block(model, keys=7, Exc=1)
    oldkey = 13

    param = filter(lambda p: p.requires_grad, model.parameters())

    # loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(param, opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()

    # switch to train mode
    model.train()
    optimizer.zero_grad()

    train_loader = get_dataset_train(opt, Part_class_list)

    # for i, (input, target, meta, vid, Auxili_info, raw_test) in enumerate(train_loader):
    #     print('test')


    for epoch in range(opt.start_epoch, opt.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        threshold = 6
        # key = key_out(epoch, [3, 5, 7, 9], [12, 8, 7], oldkey)

        if two_stage == 0:
            adjust_learning_rate(opt.lr, opt.lr_decay_step, optimizer, epoch, ratio=opt.lr_decay_ratio)
            print('This batch lr:\t', optimizer.param_groups[0]['lr'])
        else:
            exc_key = epoch > threshold
            NetFunction.Open_block(model, keys=7, Exc=exc_key)
            NetFunction.Lock_BN_Dur_train(model, keys=7, Exc=exc_key)
            param = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = torch.optim.SGD(param, opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
            optimizer.zero_grad()
            Two_stage_learning_rate(opt.lr, opt.lr_decay_step, optimizer, epoch, opt.lr_decay_ratio, threshold)
            print('This batch lr:\t', optimizer.param_groups[0]['lr'])

        def part(x): return itertools.islice(x, int(len(x)*opt.train_size))
        data_loader = part(train_loader)
        end = time.time()
        for i, (input, target, meta, vid, Auxili_info, raw_test) in enumerate(data_loader):
            # Image._show(Image.fromarray(raw_test[0, :, :, :].numpy().astype(np.uint8)))
            gc.collect()
            data_time.update(time.time() - end)
            meta['epoch'] = epoch
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.float().cuda(async=True))
            Auxili_info = torch.autograd.Variable(Auxili_info)
            output = model(input_var, Auxili_info)
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

            if i % int(0.05*opt.train_size*len(train_loader)) == 0:
                print('Epoch: [{0}][{1}/{2}({3})]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          epoch, i, int(
                              len(train_loader)*opt.train_size), len(train_loader),
                          batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5))
        #
        scores = {'top1train': round(top1.avg, 3), 'top5train': round(top5.avg, 3), 'losstrain': round(losses.avg, 3)}
        checkpoints.save(epoch, opt, model, optimizer, 1, scores, Tr_log_dir, Md_dir)
