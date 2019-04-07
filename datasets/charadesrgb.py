""" Dataset loader for the Charades dataset """
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
from glob import glob
import csv
import pickle
import os
from utils import trasmy
import cv2

def parse_charades_csv(filename):
    labels = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['id']
            actions = row['actions']
            length = row['length']
            if actions == '':
                actions = []
            else:
                actions = [a.split(' ') for a in actions.split(';')]
                actions = [{'class': x, 'start': float(
                    y), 'end': float(z)} for x, y, z in actions]
            labels[vid] = [actions, length]
    return labels


def cls2int(x):
    return int(x[1:])


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def cache(cachefile):
    """ Creates a decorator that caches the result to cachefile """
    def cachedecorator(fn):
        def newf(*args, **kwargs):
            print('cachefile {}'.format(cachefile))
            if os.path.exists(cachefile):
                with open(cachefile, 'rb') as f:
                    print("Loading cached result from '%s'" % cachefile)
                    return pickle.load(f)
            res = fn(*args, **kwargs)
            with open(cachefile, 'wb') as f:
                print("Saving result to cache '%s'" % cachefile)
                pickle.dump(res, f)
            return res
        return newf
    return cachedecorator


class Charades(data.Dataset):
    def __init__(self, root, split, labelpath, cachedir, transform=None, target_transform=None,
                 label_list=[],vdlist=None, args=None):
        self.num_classes = len(label_list)
        self.transform = transform
        self.target_transform = target_transform
        self.labels = parse_charades_csv(labelpath)
        self.root = root
        # self.testGAP = 50  # test sample num in each video  if  for 3D ConV using 15
        self.testGAP = 15
        self.vdlist = vdlist
        self.duration = args.duration
        self.imagenum = args.image_num
        self.half_duration = int(self.duration * (self.imagenum-1) * 0.5)
        cachename = '{}/{}_{}.pkl'.format(cachedir,
                                          self.__class__.__name__, split)
        self.data = self.prepare(root, self.labels, split, label_map=label_list, cache_name=cachename)

    def prepare(self, path, labels, split, label_map=[], cache_name=''):
        if os.path.exists(cache_name):
            with open(cache_name, 'rb') as f:
                print("Loading cached result from '%s'" % cache_name.split('/')[-1])
                dataset = pickle.load(f)
                print('Find ', len(dataset), ' data named', cache_name.split('/')[-1])
                return dataset
        else:
            FPS, GAP, testGAP = 24, 4, self.testGAP
            datadir = path
            dataset = []
            video_num = 0
            for i, (vid, VD_info) in enumerate(labels.items()):
                label = VD_info[0]
                VD_length = VD_info[1]
                Vid_key = 0
                iddir = datadir + '/' + vid
                lines = glob(iddir+'/*.jpg')
                n = len(lines)
                FPS = n/float(VD_length)
                assert 24 < FPS < 25, 'FPS ERROR'
                if split == 'val_video':
                    fm_No = []
                    target = np.zeros((self.num_classes, 1)).astype(np.int8)
                    for x in label:
                        if cls2int(x['class']) in label_map:
                            label_No = label_map.index(cls2int(x['class']))
                            target[label_No] = 1
                            Vid_key = 1  # Effice video
                    if Vid_key == 1:
                        for ii in range(0, n-1):
                            for x in label:
                                if x['start'] < ii/float(FPS) < x['end'] and cls2int(x['class']) in label_map:
                                    fm_order = ii + 1
                                    if fm_order not in fm_No:
                                        fm_No.append(fm_order)

                        spacing_index = np.linspace(0, len(fm_No)-1, testGAP).astype(np.int16)
                        FM_list = np.array(fm_No)[spacing_index].tolist()
                        for No in FM_list:
                            # dataset.append((vid, target, '{:06d}'.format(No+1)))
                            dataset.append((vid, target, '{:06d}'.format(No+1), n))
                        video_num = video_num + 1
                    if i % 99 == 0:
                        print('Process ', video_num, ' / ', i+1)

                else:
                    for ii in range(0, n-1, GAP):
                        target = np.zeros((self.num_classes, 1)).astype(np.int8)
                        for x in label:
                            if x['start'] < ii/float(FPS) < x['end'] and cls2int(x['class']) in label_map:
                                label_No = label_map.index(cls2int(x['class']))
                                target[label_No] = 1
                                im_NO = '{:06d}'.format(ii+1)
                                # dataset.append((vid, target, im_NO))
                                dataset.append((vid, target, im_NO, n))
                                Vid_key = 1
                    if Vid_key == 1:
                        video_num = video_num + 1
                    if i % 99 == 0:
                        print('Process ', video_num, ' / ', i+1)
            print('Find ', i+1, ' videos ', video_num, 'is effective video')
            dataset_name = cache_name
            with open(dataset_name, 'wb') as file:
                pickle.dump(dataset, file)
            return dataset

    def pre_img(self, vid, fm_NO, total_num, out_num_vd):
        vd_fm_num = out_num_vd
        fm_index = int(fm_NO)
        total_num = total_num
        out_img = []
        if 1 < (fm_index-self.half_duration) and (fm_index + self.half_duration) < total_num:
            index = np.linspace(fm_index-self.half_duration, fm_index+self.half_duration, vd_fm_num).astype(np.int)
            for i in index:
                path = '{}/{}-{:06d}.jpg'.format(os.path.join(self.root, vid), vid, i)
                img = default_loader(path)
                out_img.append(img)
        elif 1 >= (fm_index-self.half_duration):
            if fm_index + 2 * self.half_duration +2 > total_num:
                index = np.linspace(1, total_num, vd_fm_num).astype(np.int)
            else:
                index = np.linspace(fm_index, fm_index + 2 * self.half_duration, vd_fm_num).astype(np.int)
            for i in index:
                path = '{}/{}-{:06d}.jpg'.format(os.path.join(self.root, vid), vid, i)
                img = default_loader(path)
                out_img.append(img)
        elif (fm_index + self.half_duration) >= total_num:
            if fm_index - 2 * self.half_duration < 1:
                index = np.linspace(1, total_num, vd_fm_num).astype(np.int)
            else:
                index = np.linspace(fm_index - 2 * self.half_duration, fm_index, vd_fm_num).astype(np.int)
            for i in index:
                path = '{}/{}-{:06d}.jpg'.format(os.path.join(self.root, vid), vid, i)
                # if not os.path.exists(path):
                #     print('wait')
                img = default_loader(path)
                out_img.append(img)
        return out_img, index

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, target, fm_NO, total_num = self.data[index]
        images, Img_index = self.pre_img(vid, fm_NO, total_num, self.imagenum)
        target = torch.from_numpy(target[:, 0].astype(np.float32))
        raw_test = torch.FloatTensor(cv2.resize(np.array(images[2]), (224, 224), interpolation=cv2.INTER_LINEAR))
        meta = {}
        meta['id'] = vid
        meta['time'] = int(fm_NO)

        Org_W, Org_H= images[0].size[0], images[0].size[1]
        Auxili_info = [self.vdlist.index(vid)]
        [Auxili_info.insert(1, im_No) for im_No in reversed(Img_index)]
        Auxili_info.insert(1, Org_H)
        Auxili_info.insert(2, Org_W)

        out_process_img = []
        if self.transform is not None:
            for img in images:
                img_out = self.transform(img).unsqueeze(0)
                out_process_img.append(img_out)
            final_out = torch.cat(out_process_img, 0).contiguous()
        if self.target_transform is not None:
            target = self.target_transform(target)
        final_out = final_out.permute(1, 0, 2, 3).contiguous()

        return final_out, target, meta, vid, torch.FloatTensor(np.array(Auxili_info)).contiguous(), raw_test

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get(args, Part_class_list):
    """ Entry point. Call this function to get all Charades dataloaders """
    normalize = trasmy.Norm_my(mean=[114.77, 107.74, 99.48], std=[1, 1, 1])   # for resnet 3D

    train_file = args.train_file
    val_file = args.val_file
    args.data = '/VIDEO_DATA/Charades_v1_rgb'

    train_dataset = Charades(args.data, 'train', train_file, args.cache,
                             transform=transforms.Compose([trasmy.IMG_resize(args.inputsize, args.inputsize),
                                                           transforms.ToTensor(),
                                                           normalize]),
                             label_list=Part_class_list)

    val_dataset = Charades(args.data, 'val', val_file, args.cache,
                           transform=transforms.Compose([trasmy.IMG_resize(240, 240),
                                                         trasmy.CenterCrop(args.inputsize),
                                                         transforms.ToTensor(), normalize]),
                           label_list=Part_class_list)

    valvideo_dataset = Charades(args.data, 'val_video', val_file, args.cache,
                                transform=transforms.Compose([trasmy.IMG_resize(240, 240),
                                                              trasmy.CenterCrop(args.inputsize),
                                                              transforms.ToTensor(), normalize]),
                                label_list=Part_class_list)

    return train_dataset, val_dataset, valvideo_dataset


def get_train(args, Part_class_list):
    """ Entry point. Call this function to get all Charades dataloaders """
    normalize = trasmy.Norm_my(mean=[114.77, 107.74, 99.48], std=[1, 1, 1])  # for resnet 3D
    train_file = args.train_file
    args.data = '/VIDEO_DATA/Charades_v1_rgb'
    train_dataset = Charades(args.data, 'train', train_file, args.cache,
                             transform=transforms.Compose([trasmy.IMG_resize(args.inputsize, args.inputsize),
                                                           trasmy.To_Tensor_My(),
                                                           normalize]),
                             label_list=Part_class_list, vdlist=args.vdlist, args=args)
    return train_dataset


def get_val(args, Part_class_list):
    """ Entry point. Call this function to get all Charades dataloaders """
    normalize = trasmy.Norm_my(mean=[114.77, 107.74, 99.48], std=[1, 1, 1])  # for resnet 3D

    val_file = os.path.join(os.getcwd(), '../../..', './data/data_csv/Charades_v1_test.csv')
    args.data = '/VIDEO_DATA/Charades_v1_rgb'
    resize_shape = args.inputsize + 16
    val_dataset = Charades(args.data, 'val', val_file, args.cache,
                           transform=transforms.Compose([trasmy.IMG_resize(resize_shape, resize_shape),
                                                         trasmy.CenterCrop(args.inputsize),
                                                         trasmy.To_Tensor_My(), normalize]),
                           label_list=Part_class_list,vdlist=args.vdlist, args=args)
    return val_dataset


def get_video(args, Part_class_list):
    """ Entry point. Call this function to get all Charades dataloaders """
    normalize = trasmy.Norm_my(mean=[114.77, 107.74, 99.48], std=[1, 1, 1])  # for resnet 3D

    val_file = os.path.join(os.getcwd(), '../../..', './data/data_csv/Charades_v1_test.csv')
    args.data = '/VIDEO_DATA/Charades_v1_rgb'
    resize_shape = args.inputsize + 16
    valvideo_dataset = Charades(args.data, 'val_video', val_file, args.cache,
                                transform=transforms.Compose([trasmy.IMG_resize(resize_shape, resize_shape),
                                                              trasmy.CenterCrop(args.inputsize),
                                                              trasmy.To_Tensor_My(), normalize]),
                                label_list=Part_class_list,vdlist=args.vdlist, args=args)

    return valvideo_dataset
