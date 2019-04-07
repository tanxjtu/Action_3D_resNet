import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import os
import pickle
import json
import numpy as np
from lib.model.roi_align.modules.roi_align import RoIAlignAvg, RoIAlignMax
from lib.model.roi_pooling.modules.roi_pool import _RoIPooling
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
              'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
              'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
              'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
              'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}

COCO_classes = ['Back', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']  # COCO class list

Sub_Object = ['Back', 'person', 'bench', 'backpack', 'handbag', 'bottle', 'wine glass',
              'cup', 'bowl', 'chair', 'couch', 'bed', 'dining table',
              'tv', 'laptop', 'remote', 'cell phone', 'refrigerator', 'book']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)


def box_encode(h_bx, O_BOX, Im_H, Im_W):
    encode = []
    h_x1, h_y1, h_x2, h_y2, score, _ = h_bx
    h_cx, h_cy, h_w, h_h = (h_x1 + h_x2) / 2, (h_y1 + h_y2) / 2, h_x2 - h_x1, h_y2 - h_y1
    # assert h_w > 0 and h_w > 0, 'Box error'
    for o_box in O_BOX:
        # H -> O
        o_x1, o_y1, o_x2, o_y2, sor, cls = o_box
        o_cx, o_cy, o_w, o_h = (o_x1 + o_x2) / 2, (o_y1 + o_y2) / 2, o_x2 - o_x1, o_y2-o_y1
        # assert o_w > 0 and o_w > 0, 'Object Box error'
        # dx/h_w , dy/h_h
        d_x, d_y, d_xy = (abs(h_cx - o_cx)/h_w)*100, (abs(h_cy - o_cy) / h_h)*100, \
                         (math.sqrt((h_cx - o_cx)**2 + (h_cy - o_cy)**2)/(h_w + h_h))*100
        # dx_ratio, dy_ratio
        dx_ratio, dy_ratio = (abs(h_cx - o_cx) / Im_W)*100, (abs(h_cy - o_cy) / Im_H) * 100
        # angle
        if abs(h_cy - o_cy) > 1e-5:
            det_y = h_cy - o_cy
        elif (h_cy - o_cy) > 0:
            det_y = 1e-5
        else: det_y = -1e-5
        angle = math.atan((h_cx - o_cx)/(det_y))
        # angle = math.atan((h_cx - o_cx)/(h_cy - o_cy + 0.0000001))
        # area scale
        o_h_area_scale = (o_w*o_h) / (h_w * h_h)
        # object sacle
        o_scale = o_h/o_w
        # IoU
        IoU = compute_iou(h_bx[:4], o_box[:4])
        code = {'d_x': d_x, 'd_y': d_y, 'd_xy': d_xy, 'dx_ratio': dx_ratio, 'dy_ratio': dy_ratio,
                'angle': angle, 'o_h_area_scale': o_h_area_scale, 'o_scale': o_scale,
                'IoU': IoU, 'Obj_class': int(cls), 'Obj_name': Sub_Object[int(cls)], 'Obj_score': sor}
        encode.append(code)
    return encode


def H_O_BBOX(H_BOX, O_BOX, Im_H, Im_W):
    # H_num = len(H_BOX)
    # O_num = len(O_BOX)
    # if H_num > 1 and O_num > 1:
    #     print('wait')
    O_BOX = np.array(O_BOX)
    H_O_ROI = []
    geometry =[]
    for h_bx in H_BOX:
        h_x1, h_y1, h_x2, h_y2, score, _ = h_bx
        O_x1, O_y1, O_x2, O_y2, Sor, cls = O_BOX[:, 0], O_BOX[:, 1], O_BOX[:, 2], O_BOX[:, 3], O_BOX[:, 4], O_BOX[:, 5]
        H_O_x1 = np.minimum(O_x1, h_x1)
        H_O_y1 = np.minimum(O_y1, h_y1)
        H_O_x2 = np.maximum(O_x2, h_x2)
        H_O_y2 = np.maximum(O_y2, h_y2)
        for h_o_box in zip(H_O_x1, H_O_y1, H_O_x2, H_O_y2):
            H_O_ROI.append(list(h_o_box))
        encode = box_encode(h_bx, O_BOX.tolist(), Im_H, Im_W)
        geometry.append(encode)
    return H_O_ROI, geometry


def area_sacale_ratio(BBox,im_h,im_w):
    '''
    Return BBox ratio of this frame
    :param BBox: type list each liat contain the x12,y12 og the bbox
    :param im_h:  uint8 such as 180
    :param im_w:  uint8 such as 240
    :return:  ratio scale
    '''
    im_area = im_h * im_w
    out = []
    for bx in BBox:
        x1, y1, x2, y2, *_ = bx
        area_box = (x2- x1) * (y2 - y1)
        ratio = (area_box/im_area)*100 # %
        out.append(ratio)
    return out


def HOI_ins_out(HOI_ins):
    object_code = np.zeros(19)
    object_code[HOI_ins['Obj_class']] = 1
    IOU = HOI_ins['IoU']
    Obj_Sco = HOI_ins['Obj_score']
    angle = HOI_ins['angle']
    dx, dy, d_xy = HOI_ins['d_x']/100, HOI_ins['d_y']/100, HOI_ins['d_xy']/100
    HOI_area, objscale = HOI_ins['o_h_area_scale'], HOI_ins['o_scale']
    vet1 = np.array([IOU, Obj_Sco, angle, dx, dy, d_xy, HOI_area, objscale])
    sum_vet1 = vet1.sum()
    norm_vet1 = vet1 / sum_vet1
    out = np.concatenate([object_code, norm_vet1])
    return out


def HOI_ins_8(HOI_ins):
    IOU = HOI_ins['IoU']
    Obj_Sco = HOI_ins['Obj_score']
    angle = HOI_ins['angle']
    dx, dy, d_xy = HOI_ins['d_x']/100, HOI_ins['d_y']/100, HOI_ins['d_xy']/100
    HOI_area, objscale = HOI_ins['o_h_area_scale'], HOI_ins['o_scale']
    vet1 = np.array([IOU, Obj_Sco, angle, dx, dy, d_xy, HOI_area, objscale])
    sum_vet1 = vet1.sum()
    norm_vet1 = vet1 / sum_vet1
    return norm_vet1


def Obj_int_out(HOI_ins):
    object_code = np.zeros(19)
    object_code[HOI_ins['Obj_class']] = 1
    return object_code


def global_Object(encode):
    object_code = np.zeros((1, 19))
    for h_No in encode:
        for HOI_ins in h_No:
            object_code[0, HOI_ins['Obj_class']] = 1
    sum = object_code.sum()
    out_code = object_code/sum
    return out_code


def HOI_encode(encode):
    # Obj_code = global_Object(encode)
    out = []
    for h_No in encode:
        for HOI_ins in h_No:
            code = HOI_ins_out(HOI_ins)
            out.append(code)
    return np.array(out)


def HOI_encode_mulity(encode, object_code):
    # Obj_code = global_Object(encode)
    out = []
    for h_No in encode:
        for HOI_ins in h_No:
            code = HOI_ins_8(HOI_ins)
            out.append(code)
    code_geo = np.array(out)
    final = np.concatenate([object_code, code_geo], 1)
    return final


def Object_encode(encode):
    # Obj_code = global_Object(encode)
    out = []
    for h_No in encode:
        for HOI_ins in h_No:
            code = Obj_int_out(HOI_ins)
            out.append(code)
    return np.array(out)


class Torch_ROI(nn.Module):
    def __init__(self, feature_scal=14):
        super(Torch_ROI, self).__init__()
        self.Adp_Avg_Pool = nn.AdaptiveAvgPool2d((7, 7))
        self.Adp_Max_Pool = nn.AdaptiveMaxPool2d((7, 7))
        self.scale = 1/16.0
        self.fea_scal = feature_scal

    def forward(self, tensor, ROI):
        ROI = (ROI.data.cpu().numpy()).tolist()
        out = []
        for Sg_ROI in ROI:
            fm = tensor.clone()
            # fm = torch.index_select(tensor, 0, Variable(torch.LongTensor([0])).cuda())
            x1, y1, x2, y2 = self.get_Cord(np.array(Sg_ROI[1:])*self.scale)
            # ROI_fea = fm[:, :, x1:x2, y1:y2].contiguous() # Bug
            ROI_fea = fm[:, :, y1:y2, x1:x2].contiguous()
            Pooled_feat = self.Adp_Avg_Pool(ROI_fea)
            out.append(Pooled_feat)
        final_out = torch.cat(out, 0)
        return final_out

    def get_Cord(self,float_ROI):
        x1, y1, x2, y2 = float_ROI
        x1 = max(math.floor(x1), 0)
        y1 = max(math.floor(y1), 0)
        x2 = min(math.ceil(x2), self.fea_scal)
        y2 = min(math.ceil(y2), self.fea_scal)  # I think using this function x1 < x2-1 and  y1 < y2-1
        return x1, y1, x2, y2


class RoI_layer(nn.Module):
    def __init__(self, out_size, in_im_sz):
        """Initializes RoI_layer module."""
        super(RoI_layer, self).__init__()
        self.out_size = out_size
        self.in_img_sz = in_im_sz

        # define rpn
        self.ROI_Align = RoIAlignAvg(self.out_size, self.out_size, 1 / 16.0)       # 224->14 : 16
        self.ROI_Pool = _RoIPooling(self.out_size, self.out_size, 1 / 16.0)        # 224->14 : 16
        self.Ptorch_ROI = Torch_ROI(feature_scal=(self.in_img_sz / 16))  # 224->14 : 16
        self.Scens_Sparse = np.array([[0, 0, 0, self.in_img_sz-32, self.in_img_sz-32]])

    def forward(self, input, BBox_info=None):
        batch_out = []
        for bc_No, Box_fm in enumerate(BBox_info):
            Each_bc = torch.index_select(input, 0, Variable(torch.LongTensor([bc_No])).cuda())

            # ----------init----------
            out_key = [True, len(Box_fm[0]) > 0, len(Box_fm[1]) > 0, len(Box_fm[0]) > 0 and len(Box_fm[1]) > 0]
            #  whether exist this kind node
            out_num = [1, Box_fm[6], Box_fm[7], Box_fm[6]*Box_fm[7]]  # each node num
            Score = [[0], [], [], []]
            H_area_FM_ratio = []
            O_area_FM_ratio = []
            HO_area_FM_ratio = []
            Object_cls = []
            geometry = []

            # -----------video info------------
            H_Box, O_Box, im_h, im_w, vid, fn_No, h_hum, o_num = Box_fm  # original ratio
            ratio_H, ratio_W = np.round(self.in_img_sz / im_h, 3), np.round(self.in_img_sz / im_w, 3)

            #  ---------------Scene node---------------
            S_node = Each_bc.clone()

            # -----------test ROI align----------------
            # test_box = Variable(torch.from_numpy(self.Scens_Sparse).float().cuda())
            # test_out_tensor1 = self.ROI_Align(S_node, test_box)
            # test_out_tensor2 = self.ROI_Pool(S_node, test_box)
            # -----------test ROI align----------------

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
                else:
                    H_O_Node = None

            else:
                O_Node = None
                H_O_Node = None

            area_ratio_fm = [H_area_FM_ratio,  O_area_FM_ratio, HO_area_FM_ratio, vid, fn_No, Object_cls]

            batch_out.append([S_node, H_Node, O_Node, H_O_Node, out_key, out_num, Score, geometry, area_ratio_fm])

        return batch_out


class RoI_layer_mulity(nn.Module):
    def __init__(self, out_size, in_im_sz):
        """Initializes RoI_layer module."""
        super(RoI_layer_mulity, self).__init__()
        self.out_size = out_size
        self.in_img_sz = in_im_sz

        # define rpn
        self.ROI_Align = RoIAlignAvg(self.out_size, self.out_size, 1 / 16.0)       # 224->14 : 16
        # self.ROI_Pool = _RoIPooling(self.out_size, self.out_size, 1 / 16.0)        # 224->14 : 16
        self.Ptorch_ROI = Torch_ROI(feature_scal=(self.in_img_sz / 16))  # 224->14 : 16
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


class RoI_layer_Pt(nn.Module):
    def __init__(self, out_size, in_im_sz):
        """Initializes RoI_layer module."""
        super(RoI_layer_Pt, self).__init__()
        self.out_size = out_size
        self.in_img_sz = in_im_sz

        # define rpn
        self.ROI_Align = RoIAlignAvg(self.out_size, self.out_size, 1 / 16.0)       # 224->14 : 16
        self.ROI_Pool = _RoIPooling(self.out_size, self.out_size, 1 / 16.0)        # 224->14 : 16
        self.Ptorch_ROI = Torch_ROI(feature_scal=(self.in_img_sz / 16))  # 224->14 : 16
        self.Scens_Sparse = np.array([[0, 0, 0, self.in_img_sz-32, self.in_img_sz-32]])

    def forward(self, input, BBox_info=None):
        # batch_size = len(BBox_info)
        batch_out = []
        for bc_No, Box_fm in enumerate(BBox_info):
            Each_bc = torch.index_select(input, 0, Variable(torch.LongTensor([bc_No])).cuda())

            # ----------init----------
            out_key = [True, len(Box_fm[0]) > 0, len(Box_fm[1]) > 0, len(Box_fm[0]) > 0 and len(Box_fm[1]) > 0]
            #  whether exist this kind node
            out_num = [1, Box_fm[6], Box_fm[7], Box_fm[6]*Box_fm[7]]  # each node num
            Score = [[0], [], [], []]
            H_area_FM_ratio = []
            O_area_FM_ratio = []
            HO_area_FM_ratio = []
            Object_cls = []
            geometry = []

            # -----------video info------------
            H_Box, O_Box, im_h, im_w, vid, fn_No, h_hum, o_num = Box_fm  # original ratio
            ratio_H, ratio_W = np.round(self.in_img_sz / im_h, 3), np.round(self.in_img_sz / im_w, 3)

            #  ---------------Scene node---------------
            S_node = Each_bc.clone()

            # -----------test ROI align----------------
            # test_box = Variable(torch.from_numpy(self.Scens_Sparse).float().cuda())
            # test_out_tensor1 = self.ROI_Align(S_node, test_box)
            # test_out_tensor2 = self.ROI_Pool(S_node, test_box)
            # -----------test ROI align----------------

            # -----------Human node------------
            if out_key[1]:
                Score[1] = np.array(H_Box)[:, -2]  # human det score
                H_area_FM_ratio = area_sacale_ratio(H_Box, im_h, im_w)
                Box = np.insert(np.array(H_Box)[:, :4], 0, values=0, axis=1)
                Nm_box = np.round(Box * [1, ratio_W, ratio_H, ratio_W, ratio_H], 3)
                Hn_box = Variable(torch.from_numpy(Nm_box).float().cuda())
                # H_Node = self.ROI_Pool(Each_bc, Hn_box)
                # H_Node = self.ROI_Align(Each_bc, Hn_box)
                H_Node = self.Ptorch_ROI(Each_bc, Hn_box)
            else: H_Node = None

            # -----------Object node-----------
            if out_key[2]:
                Score[2] = np.array(O_Box)[:, -2]
                O_area_FM_ratio = area_sacale_ratio(O_Box, im_h, im_w)
                Box = np.insert(np.array(O_Box)[:, :4], 0, values=0, axis=1)
                Nm_box = np.round(Box * [1, ratio_W, ratio_H, ratio_W, ratio_H], 3)
                On_box = Variable(torch.from_numpy(Nm_box).float().cuda())
                # O_Node = self.ROI_Pool(Each_bc, On_box)
                # O_Node = self.ROI_Align(Each_bc, On_box)
                O_Node = self.Ptorch_ROI(Each_bc, On_box)
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
                    # H_O_Node = self.ROI_Align(Each_bc, H_O_box)
                    H_O_Node = self.Ptorch_ROI(Each_bc, H_O_box)
                else:
                    H_O_Node = None

            else:
                O_Node = None
                H_O_Node = None

            area_ratio_fm = [H_area_FM_ratio,  O_area_FM_ratio, HO_area_FM_ratio, vid, fn_No, Object_cls]

            batch_out.append([S_node, H_Node, O_Node, H_O_Node, out_key, out_num, Score, geometry, area_ratio_fm])

        return batch_out


class Load_fm_BBox(object):
    def __init__(self, vdlist, Box_dir):
        self.vd_list = vdlist
        self.BOX_fl = json.load(open(Box_dir, 'r'))
        print('Box loaded')

    def __call__(self, vd_info):
        fm_info = vd_info.data.cpu().numpy().astype(np.int).tolist()
        # bc_sz = len(fm_info)
        bc_out = []
        for ec_fm in fm_info:
            # h_Box, o_Box = [], []
            vd_index, im_h, im_w, st = ec_fm
            vd = self.vd_list[vd_index]
            box = self.BOX_fl[vd]['%06d' % st]
            h_hum, o_num = box[1], box[2]-box[1]
            h_Box = box[0][:h_hum]
            o_Box = box[0][h_hum:]
            bc_out.append([h_Box, o_Box, im_h, im_w, vd, st, h_hum, o_num])
        return bc_out


class Load_fm_BBoxV2(object):
    def __init__(self, vdlist):
        self.vd_list = vdlist
        self.bx_dir = '/VIDEO_DATA/tar_box/'
        print('Box and train info loaded')

    def load_fm_Box(self, vid, fm_No):
        h_Box = []
        O_Box = []
        vd_bx_dir = os.path.join(self.bx_dir, vid) + '.json'
        vd_bbox = json.load(open(vd_bx_dir, 'r'))
        det_all = vd_bbox['%06d' % fm_No]
        hum_num, obj_num = det_all[-2], det_all[-1] - det_all[-2]
        for i in range(0, hum_num):
            h_Box.append(det_all[i])
        for j in range(hum_num, len(det_all)-2):
            O_Box.append(det_all[j])
        return h_Box, O_Box

    def __call__(self, vd_info):
        fm_info = vd_info.data.cpu().numpy().astype(np.int).tolist()
        bc_out = []
        for ec_fm_info in fm_info:
            index, Im_H, Im_W, fm_No = ec_fm_info
            vid = self.vd_list[index]
            h_Box, O_Box = self.load_fm_Box(vid, fm_No)
            h_hum, o_num = len(h_Box), len(O_Box)
            # print('human\t', h_hum, '\tall\t', h_hum+o_num)
            bc_out.append([h_Box, O_Box, Im_H, Im_W, vid, fm_No, h_hum, o_num])
        return bc_out


class Load_fm_BBoxV3(object):    # mulity box
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


class ConvGraph(nn.Module):
    def __init__(self):
        super(ConvGraph, self).__init__()
        # self.Update_S_Fn = Update_S()
        # self.adj_mat = adj_function()
        self.propagation_step = 1
        self.area_ratio_thre = 7
        self.inchannel = 512
        self.sigmoid = torch.nn.Sigmoid()
        self.Softmax = torch.nn.Softmax(dim=1)
        self.Conv1 = torch.nn.Conv2d(self.inchannel, self.inchannel, (1, 1), 1, 0, bias=True)
        self.Conv2 = torch.nn.Conv2d(self.inchannel, self.inchannel, (1, 1), 1, 0, bias=True)
        self.dropout = torch.nn.Dropout()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0, std=0.01)
                m.bias.data.normal_(mean=0, std=0.01)

    def forward(self, input):
        # batch_sz = len(input)
        batch_out = []
        for ba_No, batch in enumerate(input):  # process each video frame
             # ready node info
            s_node, h_node, o_node, ho_node, key, num_info, score, encode, HOI_fm_area_ratio, Fake_H_S = batch
            h_area, o_area, ho_area, vid, fn_No, Obj_cls = HOI_fm_area_ratio

            add_key = 0
            ts_colt = []
            # select to process which
            if key[1]:   # exist human
                hum_index = np.where(np.array(h_area) > self.area_ratio_thre)[0]
                if len(hum_index):
                    h_sed = torch.index_select(h_node, 0, Variable(torch.LongTensor(hum_index)).cuda())
                    ts_colt.append(h_sed)
                    add_key = 1

            if key[2]:  # exist object
                obj_index = np.where(np.array(o_area) > self.area_ratio_thre)[0]
                if len(obj_index):
                    o_sed = torch.index_select(o_node, 0, Variable(torch.LongTensor(obj_index)).cuda())
                    ts_colt.append(o_sed)
                    add_key = 1

                if key[3]:
                    HOI_index = np.where(np.array(ho_area) > self.area_ratio_thre)[0]
                    if len(HOI_index):
                        HOI_sed = torch.index_select(ho_node, 0, Variable(torch.LongTensor(HOI_index)).cuda())
                        ts_colt.append(HOI_sed)
            # if add_key == 0:
            ts_colt.append(Fake_H_S)

            clt_tensor = torch.cat(ts_colt, 0).unsqueeze(2).unsqueeze(2)
            fm_pred = []
            fm_pred.append(clt_tensor.squeeze(2).squeeze(2))
            for step in range(self.propagation_step):
                state = fm_pred[step]
                mat1 = self.Conv1(state.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2)
                mat2 = self.Conv2(state.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2)
                adj_mat = mat1.mm(mat2.t())
                soft_adj = self.Softmax(adj_mat)
                update = soft_adj.mm(state)
                fm_pred.append(update)
            # gp_out = fm_pred[-1].mean(0).unsqueeze(0)
            # gp_out = self.dropout(fm_pred[-1]).mean(0).unsqueeze(0)
            gp_out = self.dropout(fm_pred[-1].mean(0)).unsqueeze(0)
            fm_out = torch.cat([s_node, gp_out], 1)
            batch_out.append(fm_out)
        out = torch.cat(batch_out, 0)
        return out

'''
class ConvGraphV1(nn.Module):
    def __init__(self):
        super(ConvGraphV1, self).__init__()
        # self.Update_S_Fn = Update_S()
        # self.adj_mat = adj_function()
        self.propagation_step = 1
        self.area_ratio_thre = 7
        self.inchannel = 512
        self.sigmoid = torch.nn.Sigmoid()
        self.Softmax = torch.nn.Softmax(dim=1)
        self.Conv1 = torch.nn.Conv2d(self.inchannel, self.inchannel, (1, 1), 1, 0, bias=True)
        self.Conv2 = torch.nn.Conv2d(self.inchannel, self.inchannel, (1, 1), 1, 0, bias=True)
        self.dropout = torch.nn.Dropout()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0, std=0.01)
                m.bias.data.normal_(mean=0, std=0.01)

    def forward(self, input):
        # batch_sz = len(input)
        batch_out = []
        for vd_feat in input:
            vd_info = vd_feat[0]
            vd_scene = []
            vd_nodes = []
            for ba_No, batch in enumerate(vd_feat[1:]):  # process each video frame
                # ready node info
                s_node, h_node, o_node, ho_node, key, num_info, score, encode, HOI_fm_area_ratio, Fake_H_S = batch
                h_area, o_area, ho_area, fn_No, Obj_cls = HOI_fm_area_ratio

                add_key = 0
                ts_colt = []
                # select to process which
                if key[1]:   # exist human
                    hum_index = np.where(np.array(h_area) > self.area_ratio_thre)[0]
                    if len(hum_index):
                        h_sed = torch.index_select(h_node, 0, Variable(torch.LongTensor(hum_index)).cuda())
                        ts_colt.append(h_sed)
                        add_key = 1

                if key[2]:  # exist object
                    obj_index = np.where(np.array(o_area) > self.area_ratio_thre)[0]
                    if len(obj_index):
                        o_sed = torch.index_select(o_node, 0, Variable(torch.LongTensor(obj_index)).cuda())
                        ts_colt.append(o_sed)
                        add_key = 1

                    if key[3]:
                        HOI_index = np.where(np.array(ho_area) > self.area_ratio_thre)[0]
                        if len(HOI_index):
                            HOI_sed = torch.index_select(ho_node, 0, Variable(torch.LongTensor(HOI_index)).cuda())
                            ts_colt.append(HOI_sed)
                # if add_key == 0:
                ts_colt.append(Fake_H_S)

                clt_tensor = torch.cat(ts_colt, 0).unsqueeze(2).unsqueeze(2)
                fm_pred = []
                fm_pred.append(clt_tensor.squeeze(2).squeeze(2))
                for step in range(self.propagation_step):
                    state = fm_pred[step]
                    mat1 = self.Conv1(state.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2)
                    mat2 = self.Conv2(state.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2)
                    adj_mat = mat1.mm(mat2.t())
                    soft_adj = self.Softmax(adj_mat)
                    update = soft_adj.mm(state)
                    # update += state
                    new_state = update + state
                    # fm_pred.append(update)
                    fm_pred.append(new_state)
                gp_out = fm_pred[-1].mean(0).unsqueeze(0)
                # gp_out = self.dropout(fm_pred[-1].mean(0)).unsqueeze(0)

                vd_scene.append(s_node)
                vd_nodes.append(gp_out)
            scene_batch = torch.cat(vd_scene, 0)
            node_batch = torch.cat(vd_nodes, 0)
            batch_out.append([scene_batch, node_batch])
        return batch_out

'''


class ConvGraphV1(nn.Module):
    def __init__(self):
        super(ConvGraphV1, self).__init__()
        # self.Update_S_Fn = Update_S()
        # self.adj_mat = adj_function()
        self.propagation_step = 1
        self.area_ratio_thre = 7
        self.inchannel = 512
        self.sigmoid = torch.nn.Sigmoid()
        self.Softmax = torch.nn.Softmax(dim=1)
        self.Conv1 = torch.nn.Conv2d(self.inchannel, self.inchannel, (1, 1), 1, 0, bias=True)
        self.Conv2 = torch.nn.Conv2d(self.inchannel, self.inchannel, (1, 1), 1, 0, bias=True)
        self.dropout = torch.nn.Dropout()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0, std=0.01)
                m.bias.data.normal_(mean=0, std=0.01)

    def forward(self, input):
        # batch_sz = len(input)
        batch_out = []
        for vd_feat in input:
            vd_info = vd_feat[0]
            ts_colt = []
            vd_scene = []
            for ba_No, batch in enumerate(vd_feat[1:]):  # process each video frame
                # ready node info
                s_node, h_node, o_node, ho_node, key, num_info, score, encode, HOI_fm_area_ratio, Fake_H_S = batch
                h_area, o_area, ho_area, fn_No, Obj_cls = HOI_fm_area_ratio
                # select to process which
                if key[1]:   # exist human
                    hum_index = np.where(np.array(h_area) > self.area_ratio_thre)[0]
                    if len(hum_index):
                        h_sed = torch.index_select(h_node, 0, Variable(torch.LongTensor(hum_index)).cuda())
                        ts_colt.append(h_sed)

                if key[2]:  # exist object
                    obj_index = np.where(np.array(o_area) > self.area_ratio_thre)[0]
                    if len(obj_index):
                        o_sed = torch.index_select(o_node, 0, Variable(torch.LongTensor(obj_index)).cuda())
                        ts_colt.append(o_sed)

                    if key[3]:
                        HOI_index = np.where(np.array(ho_area) > self.area_ratio_thre)[0]
                        if len(HOI_index):
                            HOI_sed = torch.index_select(ho_node, 0, Variable(torch.LongTensor(HOI_index)).cuda())
                            ts_colt.append(HOI_sed)

                ts_colt.append(Fake_H_S)
                vd_scene.append(s_node)

            clt_tensor = torch.cat(ts_colt, 0).unsqueeze(2).unsqueeze(2)
            fm_pred = []
            fm_pred.append(clt_tensor.squeeze(2).squeeze(2))
            for step in range(self.propagation_step):
                state = fm_pred[step]
                mat1 = self.Conv1(state.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2)
                mat2 = self.Conv2(state.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2)
                adj_mat = mat1.mm(mat2.t())
                soft_adj = self.Softmax(adj_mat)
                update = soft_adj.mm(state)
                # update += state
                new_state = update + state
                # fm_pred.append(update)
                fm_pred.append(new_state)
            gp_out = fm_pred[-1].mean(0).unsqueeze(0)
            # gp_out = self.dropout(fm_pred[-1].mean(0)).unsqueeze(0)

            scene_mean = torch.cat(vd_scene, 0).mean(0).unsqueeze(0)
            scene_vd = self.dropout(scene_mean)
            vd_final = torch.cat([scene_vd, gp_out], 1)
            batch_out.append(vd_final)
        final_out = torch.cat(batch_out, 0)

        return final_out


class TimeGraph(nn.Module):
    def __init__(self):
        super(TimeGraph, self).__init__()
        self.propagation_step = 1
        self.area_ratio_thre = 7
        self.inchannel = 512
        self.sigmoid = torch.nn.Sigmoid()
        self.Softmax = torch.nn.Softmax(dim=1)
        self.Conv1 = torch.nn.Conv2d(self.inchannel, self.inchannel, (1, 1), 1, 0, bias=True)
        self.Conv2 = torch.nn.Conv2d(self.inchannel, self.inchannel, (1, 1), 1, 0, bias=True)
        self.dropout = torch.nn.Dropout()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0, std=0.01)
                m.bias.data.normal_(mean=0, std=0.01)

    def forward(self, input):
        # batch_sz = len(input)
        batch_out = []
        for vd_s, vd_nodes in input:
            out_scene = vd_s.mean(0).unsqueeze(0)
            fm_pred = []
            fm_pred.append(vd_nodes)
            for step in range(self.propagation_step):
                state = fm_pred[step]
                mat1 = self.Conv1(state.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2)
                mat2 = self.Conv2(state.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2)
                adj_mat = mat1.mm(mat2.t())
                soft_adj = self.Softmax(adj_mat)
                update = soft_adj.mm(state)
                # update += state
                new_state = update + state
                # fm_pred.append(update)
                fm_pred.append(new_state)
            gp_out = fm_pred[-1].mean(0).unsqueeze(0)
            # gp_out = self.dropout(fm_pred[-1].mean(0)).unsqueeze(0)

            final_vd_out = torch.cat([out_scene, gp_out], 1)
            batch_out.append(final_vd_out)
        out = torch.cat(batch_out, 0)
        return out


class ConvGraph_SC(nn.Module):
    def __init__(self):
        super(ConvGraph_SC, self).__init__()
        # self.Update_S_Fn = Update_S()
        # self.adj_mat = adj_function()
        self.propagation_step = 1
        self.area_ratio_thre = 7
        self.inchannel = 512
        self.sigmoid = torch.nn.Sigmoid()
        self.Softmax = torch.nn.Softmax(dim=1)
        self.Conv1 = torch.nn.Conv2d(self.inchannel, self.inchannel, (1, 1), 1, 0, bias=True)
        self.Conv2 = torch.nn.Conv2d(self.inchannel, self.inchannel, (1, 1), 1, 0, bias=True)
        self.dropout = torch.nn.Dropout()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0, std=0.01)
                m.bias.data.normal_(mean=0, std=0.01)

    def forward(self, input):
        batch_sz = input.size()[0]
        batch_out = []
        for batch in range(batch_sz):
            fm_pred = []
            video_batch = torch.index_select(input, 0, Variable(torch.LongTensor([batch])).cuda())
            process_vd_batch = video_batch.mean(0).mean(2).mean(2)
            fm_pred.append(process_vd_batch)
            for step in range(self.propagation_step):
                state = fm_pred[step]
                mat1 = self.Conv1(state.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2)
                mat2 = self.Conv2(state.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2)
                adj_mat = mat1.mm(mat2.t())
                # soft_adj = self.Softmax(adj_mat)
                adj_norm = ((adj_mat-adj_mat.mean())/adj_mat.std())
                soft_adj = self.Softmax(adj_norm)
                update = soft_adj.mm(state)
                # update += state
                new_state = update + state
                # fm_pred.append(update)
                fm_pred.append(new_state)
            # gp_out = fm_pred[-1].mean(0).unsqueeze(0)
            gp_out = self.dropout(fm_pred[-1].mean(0)).unsqueeze(0)
            batch_out.append(gp_out)
        out = torch.cat(batch_out, 0)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, **kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.In_IM_size = kwargs['kwargs'].inputsize
        self.ROI_size = int(math.ceil(self.In_IM_size / 16))
        self.last_size = int(math.ceil(self.In_IM_size / 32))
        self.vdlist = kwargs['kwargs'].vdlist
        self.BOX_dir = kwargs['kwargs'].BBOX_dir
        self.img_name = kwargs['kwargs'].image_num

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.downConv = nn.Conv2d(2048, 512, 1, 1, padding=0, bias=True)
        # self.downConv = nn.Conv2d(2048, 1024, 1, 1, padding=0, bias=True)

        # self.avgpool = nn.AvgPool2d(self.last_size, stride=1)
        self.avgpool = nn.AvgPool2d(14, stride=1)
        self.dropout = torch.nn.Dropout()

        # self.Load_BBox_info = Load_fm_BBox(self.vdlist, self.BOX_dir)
        # self.Load_BBox_infoV2 = Load_fm_BBoxV2(self.vdlist)
        self.Load_BBox_infoV3 = Load_fm_BBoxV3(self.vdlist, self.img_name)

        #  ---------ROI model -----------------
        # self.RoI_layer = RoI_layer(out_size=self.last_size, in_im_sz=self.In_IM_size)
        self.RoI_layer = RoI_layer_mulity(out_size=self.last_size, in_im_sz=self.In_IM_size)
        # self.RoI_layer_Pt = RoI_layer_Pt(out_size=self.last_size, in_im_sz=self.In_IM_size)
        # self.Graph = Graph_Enhance_model()
        # self.Graph = Graph_Enhance_modelV1()
        # self.GNN = ConvGraph()
        self.GNNV1 = ConvGraphV1()    # resnet  model
        # self.TimeGraph = TimeGraph()
        # self.GNNSC = ConvGraph_SC()    # resnet  model

        # ---------ROI model -----------------

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def node_mulity_Max(self, input):
        batch_out = []
        for vd_feat in input:
            vd_out = []
            vd_out.append(vd_feat[0])
            im_h, im_w, vid, fm_index = vd_feat[0]
            for bc in vd_feat[1:]:
                s_fet, h_fet, o_fet, ho_fet, key, num, score, encode, area_ratio_fm, Center_S = bc
                s_node = s_fet

                if key[1]: h_node = h_fet.max(2)[0].max(2)[0]  # human
                else: h_node = None

                if key[2]: o_node = o_fet.max(2)[0].max(2)[0]  # object
                else: o_node = None

                if key[3]: ho_node = ho_fet.max(2)[0].max(2)[0]  # # human object interaction
                else: ho_node = None

                # Center_S = s_fet[:, :, 1:13, 1:13].contiguous()
                # Center_S = s_fet
                Fake_H_S = Center_S.max(2)[0].max(2)[0]
                fm_out = [s_node, h_node, o_node, ho_node, key, num, score, encode, area_ratio_fm, Fake_H_S]
                vd_out.append(fm_out)
            batch_out.append(vd_out)
        return batch_out

    def drop_cat(self, tensor):
        dropout = self.dropout(tensor)
        no_dropout = tensor
        out = torch.cat([dropout, no_dropout], 1).contiguous()
        return out

    def forward(self, x, Auxili_info=None):
        # ------load bbox----------
        bbox = self.Load_BBox_infoV3(Auxili_info)
        # ------load bbox----------

        x = x.view(-1, 3, self.In_IM_size, self.In_IM_size)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.downConv(x)
        x = x.view(-1, self.img_name, 512, self.ROI_size, self.ROI_size)   # self.img_name 5

        # ------resnet down size ---------
        # x = self.avgpool(x)
        # x = self.dropout(x)
        # x = x.view(-1, 5, 512, 1, 1).mean(1)
        # x = x.view(x.size(0), -1)
        # x = self.fcdown(x)
        # ------resnet down size ---------

        # ----------Conv Graph------------
        # ROI = self.RoI_layer(x, bbox)
        # ROI = self.RoI_layer_Pt(x, bbox)
        # nodesinfo = self.generate_nodeV5_mulity(ROI)
        # enhanced_node = self.GNN(nodesinfo)
        # enhanced_node = self.GNNV1(nodesinfo)
        # x = self.fcdown(enhanced_node)
        # ----------Conv Graph------------

        # ----------Conv Graph Only Scent------------ mulity frame
        ROI = self.RoI_layer(x, bbox)
        # ROI = self.RoI_layer_Pt(x, bbox)
        nodesinfo = self.node_mulity_Max(ROI)
        out = self.GNNV1(nodesinfo)
        x = self.fcdown(out)
        # ----------Conv Graph Only Scent------------ mulity frame

        # ----------Conv Graph Only Scent------------ mulity frame
        # enhanced_node = self.GNNSC(x)
        # x = self.fcdown(enhanced_node)
        # ----------Conv Graph Only Scent------------ mulity frame

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
