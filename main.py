'''
    author: Haoliang Tan
    Data: 22/1/2019
    Name: Object Auxiliary
'''
import torch
import numpy as np
from data import class_reduce
import train
from models import create_model
from datasets import get_dataset
from utils import checkpoints
from utils import NetFunction
from utils import opts
import os
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

best_mAP = 0


def main():
    raw_num, part_num, Original_list, Part_class_list = class_reduce.Class_num()
    global opt, best_mAP
    opt = opts.parse(part_num)

    version = 1  # change every time
    # Tr_log_dir, Md_dir = checkpoints.Make_ProjDir(opt, version)
    Tr_log_dir = './SV_version/2_CharadesSub_resnet50/trainlog'
    Md_dir = './SV_version/2_CharadesSub_resnet50/model'
    model = create_model(opt)

    # Net fixed ---------
    NetFunction.lock_Bn(model, keys=6, Exc=1)
    NetFunction.Fix_block(model, keys=5, Exc=1)
    param = filter(lambda p: p.requires_grad, model.parameters())
    # Net fixed ---------

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(param, opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    model = torch.nn.DataParallel(model).cuda()

    trainer = train.Trainer()
    train_loader, val_loader, valvideo_loader = get_dataset(opt, Part_class_list)

    for epoch in range(opt.start_epoch, opt.epochs):
        top1, top5, loss_train = trainer.train(train_loader, model, criterion, optimizer, epoch, opt)

        top1val, top5val, val_loss = trainer.validate(val_loader, model, criterion, epoch, opt)

        mAP, prec1, prec5 = trainer.validate_video(valvideo_loader, model, criterion, epoch, opt)
        #
        # top1 = 24.32445354324
        # top5 = 84.32445354324
        # loss_train = 0.243235234
        # top1val = 5.32445354324
        # top5val = 21.32445354324
        # val_loss = 0.75446532
        # mAP = 0.0832445354324
        # prec1 = 21.32445354324
        # prec5 = 50.32445354324

        is_best = mAP > best_mAP
        best_mAP = max(mAP, best_mAP)
        scores = {'top1train': round(top1, 3), 'top5train': round(top5, 3), 'losstrain': round(loss_train, 3),
                  'top1val': round(top1val, 3), 'top5val': round(top5val, 3), 'val_loss':  round(val_loss, 3),
                  'mAP': round(mAP, 3), 'videoprec1': round(prec1, 3), 'videoprec5': round(prec5, 3)}
        checkpoints.save(epoch, opt, model, optimizer, is_best, scores, Tr_log_dir, Md_dir)


if __name__ == '__main__':
    main()