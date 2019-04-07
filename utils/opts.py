""" Define and parse commandline arguments """
import argparse
import os
import csv


def parse(class_num=157):
    print('parsing arguments')
    parser = argparse.ArgumentParser(description='PyTorch Charades Training')

    # net info
    parser.add_argument('--arch', default='resnet101', type=str,help='resnet [34, 50, 101, 152]')

    parser.add_argument('--pretrained', default=1, type=int)

    # data info
    parser.add_argument('--dataname', default='CharadesSub')

    parser.add_argument('--nclass', default=class_num, type=int)

    parser.add_argument('--data', metavar='DATA_DIR', default='/VIDEO_DATA/Charades_v1_rgb', help='path to dataset')

    parser.add_argument('--dataset', default='charadesrgb', help='Data Name')

    parser.add_argument('--train-file', default='./data/data_csv/Charades_v1_train.csv')

    parser.add_argument('--val-file', default='./data/data_csv/Charades_v1_test.csv')

    # optim info
    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='total epochs to run')

    parser.add_argument('--start_epoch', default=0, type=int, help='Start Epoch')

    parser.add_argument('--lr', '--learning-rate', default=0.013, type=float, metavar='LR', help='init learning rate')

    parser.add_argument('--lr-decay-step', default=2, type=int)

    parser.add_argument('--lr-decay-ratio', default=0.8, type=float)

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')

    parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')

    # data loader info
    parser.add_argument('-j', '--workers', default=6, type=int, help='data loading workers (default: 4)')

    parser.add_argument('-b', '--batch-size', default=20, type=int, help='mini-batch size (default: 256)')

    parser.add_argument('--train-size', default=0.012, type=float)

    parser.add_argument('--val-size', default=0.003, type=float)

    parser.add_argument('--inputsize', default=224, type=int)

    parser.add_argument('--image_num', default=16, type=int)

    parser.add_argument('--duration', default=3, type=int)

    # cache and save
    parser.add_argument('--cache-dir', default='./cache')

    parser.add_argument('--sv-dir', default='./SV_version')

    parser.add_argument('--resume', default='model.pth', type=str)

    # 3D resnet
    parser.add_argument('--sample_size', default=224)



    args = parser.parse_args()

    # args.distributed = args.world_size > 1
    # args.cache = args.cache_dir + args.name
    args.cache = args.cache_dir
    if not os.path.exists(args.cache):
        os.makedirs(args.cache)

    return args


def vd_list(phase='train'):
    if phase == 'train':
        csvfile = './data/data_csv/Charades_v1_train.csv'
        box_dir = './cache/BOX/Charades_train.json'
    elif phase == 'val':
        csvfile = os.path.join(os.getcwd(), '../../..', './data/data_csv/Charades_v1_test.csv')
        box_dir = './cache/BOX/Charades_val.json'
    elif phase == 'map':
        csvfile = os.path.join(os.getcwd(), '../../..', './data/data_csv/Charades_v1_test.csv')
        box_dir = './cache/BOX/Charades_video.json'

    vd_list = []
    with open(csvfile) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['id']
            vd_list.append(vid)

    return vd_list, box_dir

