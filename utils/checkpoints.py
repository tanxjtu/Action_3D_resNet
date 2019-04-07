""" Defines functions used for checkpointing models and storing model scores """
import os
import torch
import shutil
from collections import OrderedDict
import sys


def ordered_load_state(model, chkpoint):
    """ 
        Wrapping the model with parallel/dataparallel seems to
        change the variable names for the states
        This attempts to load normally and otherwise aligns the labels
        of the two statese and tries again.
    """
    try:
        model.load_state_dict(chkpoint)
    except KeyError:  # assume order is the same, and use new labels
        print('keys do not match model, trying to align')
        modelkeys = model.state_dict().keys()
        fixed = OrderedDict([(z,y) 
                             for (x,y),z in zip(chkpoint.items(), modelkeys)])
        model.load_state_dict(fixed)


def load(args, model, optimizer):
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            chkpoint = torch.load(args.resume)
            if isinstance(chkpoint, dict) and 'state_dict' in chkpoint:
                args.start_epoch = chkpoint['epoch']
                mAP = chkpoint['mAP']
                ordered_load_state(model, chkpoint['state_dict'])
                optimizer.load_state_dict(chkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, chkpoint['epoch']))
                return mAP
            else:
                ordered_load_state(model, chkpoint)
                print("=> loaded checkpoint '{}' (just weights)"
                      .format(args.resume))
                return 0
        else:
            print("=> no checkpoint found, starting from scratch: '{}'".format(args.resume))
    return 0


def score_file(scores, filename):
    with open(filename, 'w') as f:
        for key, val in sorted(scores.items()):
            f.write('{} {}\n'.format(key, val))


def Make_ProjDir(args, version):
    version = version  # 1
    data_name = args.dataname  # str 'Charades'
    arch = args.arch  # Resnet
    dir_name = os.path.join(args.sv_dir, str(version) + '_' + data_name + '_' + arch)

    train_log_dir = os.path.join(dir_name, 'trainlog')
    model_dir = os.path.join(dir_name, 'model')
    code_dir = os.path.join(dir_name, 'code')
    train_py = sys.argv[0]
    charadesrgb = './datasets/charadesrgb.py'
    resnet_py = './models/resnet.py'
    opts = './utils/opts.py'
    val = './val_all.py'
    if os.path.exists(dir_name):
        print('The Dir exist Continue? ', dir_name)
        a = str.lower(input("Continue? YES or NO"))
        assert a =='yes', "Check the path Correctly"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        os.mkdir(train_log_dir)
        os.mkdir(model_dir)
        os.mkdir(code_dir)
    shutil.copyfile(train_py, os.path.join(code_dir, train_py.split('/')[-1]))
    shutil.copyfile(charadesrgb, os.path.join(code_dir, 'charadesrgb.py'))
    shutil.copyfile(resnet_py, os.path.join(code_dir, 'resnet.py'))
    shutil.copyfile(opts, os.path.join(code_dir, 'opts.py'))
    shutil.copyfile(val, os.path.join(code_dir, 'val_all.py'))

    return train_log_dir, model_dir


def train_log_txt(dir, scores, epoch):
    file = 'epoch: ' + str(epoch+1)+' \n ' + 'top1:\t' + \
          str(scores['top1train']) + '\ttop5:\t' + str(scores['top5train']) + '\tloss:\t' + str(scores['losstrain']) + ' \n ' + \
          '====================************============================ \n'
    if not os.path.exists(dir):
        with open(dir, 'w') as f:
            f.write(file)
    else:
        with open(dir, 'a') as f:
            f.write(file)


def save(epoch, args, model, optimizer, is_best, scores, Tr_log_dir, Md_dir):
    train_log_dir = Tr_log_dir+'/train_log.txt'
    train_log_txt(train_log_dir, scores, epoch)
    state = model.module.state_dict()
    file_name = '%02d_' % (epoch+1) + 'model.pth'
    file_dir = os.path.join(Md_dir, file_name)
    torch.save(state, file_dir)
