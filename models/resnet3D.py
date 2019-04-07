import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from lib.model.roi_align.modules.roi_align import RoIAlignAvg, RoIAlignMax
from lib.model.roi_pooling.modules.roi_pool import _RoIPooling
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import os
import pickle
import json
import numpy as np

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Load_fm_BBoxV3(object):    # mulity box   2D mulitiple frames
    def __init__(self, vdlist, im_num):  # (self.vdlist, self.img_name)
        self.vd_list = vdlist
        self.img_num = im_num
        self.bx_dir = '/VIDEO_DATA/tar_box/'
        print('Box and train info loaded')

    def load_fm_Box(self, vid, fm_list):
        vd_bx_dir = os.path.join(self.bx_dir, vid) + '.json'
        vd_bbox = json.load(open(vd_bx_dir, 'r'))
        vd_out = []
        for fm_No in fm_list:
            det_all = vd_bbox['%06d' % fm_No]
            hum_num, obj_num = det_all[-2], det_all[-1] - det_all[-2]
            h_Box, O_Box = [], []
            for i in range(0, hum_num):
                h_Box.append(det_all[i])
            for j in range(hum_num, len(det_all)-2):
                O_Box.append(det_all[j])
            vd_out.append([h_Box, O_Box, hum_num, obj_num, fm_No])
        return vd_out

    def __call__(self, vd_info):
        fm_info = vd_info.data.cpu().numpy().astype(np.int).tolist()
        bc_out = []
        for ec_fm_info in fm_info:
            index, Im_H, Im_W, fm_No = ec_fm_info[0], ec_fm_info[1],ec_fm_info[2],ec_fm_info[3:]
            vid = self.vd_list[index]
            vd_box = self.load_fm_Box(vid, fm_No)
            bc_out.append([vd_box, Im_H, Im_W, vid, fm_No])
        return bc_out


class Load_fm_BBoxV4(object):    # mulity box   3D mulitiple frames
    def __init__(self, vdlist, im_num,time):  # (self.vdlist, self.img_name)
        self.vd_list = vdlist
        self.img_num = im_num
        self.time = time
        self.time_scale = int(self.img_num / self.time)
        self.start = int(0.5 * self.time_scale)
        self.bx_dir = '/VIDEO_DATA/tar_box/'
        self.fm_sl_idx = [i * self.time_scale + self.start for i in range(self.time)]
        print('Box and train info loaded')

    def load_fm_Box(self, vid, fm_list):
        vd_bx_dir = os.path.join(self.bx_dir, vid) + '.json'
        vd_bbox = json.load(open(vd_bx_dir, 'r'))
        vd_out = []
        for fm_No in fm_list:
            det_all = vd_bbox['%06d' % fm_No]
            hum_num, obj_num = det_all[-2], det_all[-1] - det_all[-2]
            h_Box, O_Box = [], []
            for i in range(0, hum_num):
                h_Box.append(det_all[i])
            for j in range(hum_num, len(det_all)-2):
                O_Box.append(det_all[j])
            vd_out.append([h_Box, O_Box, hum_num, obj_num, fm_No])
        return vd_out

    def __call__(self, vd_info):
        fm_info = vd_info.data.cpu().numpy().astype(np.int).tolist()
        bc_out = []
        for ec_fm_info in fm_info:
            index, Im_H, Im_W, fm_No = ec_fm_info[0], ec_fm_info[1],ec_fm_info[2],ec_fm_info[3:]
            vid = self.vd_list[index]
            fm_selected = np.array(fm_No)[self.fm_sl_idx]
            vd_box = self.load_fm_Box(vid, fm_selected)
            # vd_box = self.load_fm_Box(vid, fm_No)
            bc_out.append([vd_box, Im_H, Im_W, vid, fm_No])
        return bc_out


class RoI_layer_mulity(nn.Module):
    def __init__(self, out_size, in_im_sz):
        """Initializes RoI_layer module."""
        super(RoI_layer_mulity, self).__init__()
        self.out_size = out_size
        self.in_img_sz = in_im_sz

        # define rpn
        self.ROI_Align = RoIAlignAvg(self.out_size, self.out_size, 1 / 16.0)       # 224->14 : 16
        # self.ROI_Pool = _RoIPooling(self.out_size, self.out_size, 1 / 16.0)        # 224->14 : 16
        # self.Ptorch_ROI = Torch_ROI(feature_scal=(self.in_img_sz / 16))  # 224->14 : 16
        # self.Scens_Sparse = np.array([[0, 0, 0, self.in_img_sz-32, self.in_img_sz-32]])

    def forward(self, input, BBox_info=None):
        batch_out = []
        for vd_No, box_info in enumerate(BBox_info):
            VD_ft = torch.index_select(input, 0, Variable(torch.LongTensor([vd_No])).cuda()).squeeze(0)
            im_h, im_w, vid, fn_No = box_info[1], box_info[2], box_info[3], box_info[4]
            vd_out = []
            vd_out.append([im_h, im_w, vid, fn_No])
            for bc_No, Box_fm in enumerate(box_info[0]):  # box_info[0] frame box info
                Each_bc = torch.index_select(VD_ft, 0, Variable(torch.LongTensor([bc_No])).cuda())

                # ----------init----------
                out_key = [True, Box_fm[2] > 0, Box_fm[3] > 0, Box_fm[2] > 0 and Box_fm[3] > 0]  # scene human obj HOI
                #  whether exist this kind node
                out_num = [1, Box_fm[2], Box_fm[3], Box_fm[2]*Box_fm[3]]  # scene human obj HOI
                Score = [[0], [], [], []]
                H_area_FM_ratio = []
                O_area_FM_ratio = []
                HO_area_FM_ratio = []
                Object_cls = []
                geometry = []

                # -----------video info------------
                H_Box, O_Box, h_hum, o_num, fn_No  = Box_fm  # original ratio
                ratio_H, ratio_W = np.round(self.in_img_sz / im_h, 3), np.round(self.in_img_sz / im_w, 3)

                #  ---------------Scene node---------------
                S_node_pre = Each_bc.clone()
                Center_S = S_node_pre[:, :, 1:13, 1:13].contiguous()
                # S_node = S_node_pre
                Fake_H_S = Center_S
                S_node = S_node_pre.mean(2).mean(2)
                # Fake_H_S = Center_S.mean(2).mean(2)

                # -----------Human node------------
                if out_key[1]:
                    Score[1] = np.array(H_Box)[:, -2]  # human det score
                    H_area_FM_ratio = area_sacale_ratio(H_Box, im_h, im_w)
                    Box = np.insert(np.array(H_Box)[:, :4], 0, values=0, axis=1)
                    Nm_box = np.round(Box * [1, ratio_W, ratio_H, ratio_W, ratio_H], 3)
                    Hn_box = Variable(torch.from_numpy(Nm_box).float().cuda())
                    # H_Node = self.ROI_Pool(Each_bc, Hn_box)
                    H_Node = self.ROI_Align(Each_bc, Hn_box)
                    # H_Node = self.Ptorch_ROI(Each_bc, Hn_box)
                    # H_Node = H_Node.mean(2).mean(2)
                else: H_Node = None

                # -----------Object node-----------
                if out_key[2]:
                    Score[2] = np.array(O_Box)[:, -2]
                    O_area_FM_ratio = area_sacale_ratio(O_Box, im_h, im_w)
                    Box = np.insert(np.array(O_Box)[:, :4], 0, values=0, axis=1)
                    Nm_box = np.round(Box * [1, ratio_W, ratio_H, ratio_W, ratio_H], 3)
                    On_box = Variable(torch.from_numpy(Nm_box).float().cuda())
                    # O_Node = self.ROI_Pool(Each_bc, On_box)
                    O_Node = self.ROI_Align(Each_bc, On_box)
                    # O_Node = self.Ptorch_ROI(Each_bc, On_box)
                    # O_Node = O_Node.mean(2).mean(2)
                    Object_cls.append(np.array(O_Box)[:, -1])

                    # -----------Human_object node-----------
                    if out_key[3]:
                        H_O_ROI, geometry = H_O_BBOX(H_BOX=H_Box, O_BOX=O_Box, Im_H=im_h, Im_W=im_w)
                        HO_area_FM_ratio = area_sacale_ratio(H_O_ROI, im_h, im_w)
                        Score[3] = np.array(O_Box)[:, -2]  # equal to object score
                        Box = np.insert(np.array(H_O_ROI)[:, :4], 0, values=0, axis=1)
                        Nm_box = np.round(Box * [1, ratio_W, ratio_H, ratio_W, ratio_H], 3)
                        H_O_box = Variable(torch.from_numpy(Nm_box).float().cuda())
                        # H_O_Node = self.ROI_Pool(Each_bc, H_O_box)
                        H_O_Node = self.ROI_Align(Each_bc, H_O_box)
                        # H_O_Node = self.Ptorch_ROI(Each_bc, H_O_box)
                        # H_O_Node = H_O_Node.mean(2).mean(2)
                    else:
                        H_O_Node = None
                else:
                    O_Node = None
                    H_O_Node = None

                area_ratio_fm = [H_area_FM_ratio,  O_area_FM_ratio, HO_area_FM_ratio, fn_No, Object_cls]
                vd_out.append([S_node, H_Node, O_Node, H_O_Node, out_key, out_num, Score, geometry, area_ratio_fm, Fake_H_S])
            batch_out.append(vd_out)
        return batch_out


class ResNet(nn.Module):

    def __init__(self, block, layers, sample_size,
                 sample_duration, shortcut_type='B', num_classes=400, **kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.In_IM_size = kwargs['kwargs'].inputsize
        self.vdlist = kwargs['kwargs'].vdlist
        self.BOX_dir = kwargs['kwargs'].BBOX_dir
        self.img_num = kwargs['kwargs'].image_num
        self.ROI_size = int(math.ceil(self.In_IM_size / 16))
        self.last_size = int(math.ceil(self.In_IM_size / 32))
        self.time_scle = int(self.img_num/8)

        self.conv1 = nn.Conv3d(
            3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1)
        self.ConDown = nn.Conv3d(2048, 512, (1, 1, 1), bias=True)
        self.Drop = nn.Dropout()

        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d( (last_duration, last_size, last_size), stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc = nn.Linear(512, num_classes)

        # -----------------graph -----------------
        # load box
        self.Load_BBox_infoV3 = Load_fm_BBoxV4(self.vdlist, self.img_num, self.time_scle)
        # ROI layer
        self.RoI_layer = RoI_layer_mulity(out_size=self.last_size, in_im_sz=self.In_IM_size)

        # -----------------graph -----------------

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                                          nn.Conv3d(self.inplanes, planes * block.expansion,
                                          kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, Auxiliary=None):

        # ------load bbox----------
        bbox = self.Load_BBox_infoV3(Auxiliary)
        # ------load bbox----------

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.ConDown(x)

        # ---------original------
        # x = x.mean(2).mean(2).mean(2)
        # x = self.Drop(x)
        # x = self.fc(x)
        # ---------original------

        # ---------graph --------
        ROI = self.RoI_layer(x, bbox)
        x = self.fcdown(x)
        # ---------graph --------


        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(pretrained=False,**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(pretrained=False,**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
