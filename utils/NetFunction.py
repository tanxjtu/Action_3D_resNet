import torch.nn as nn


def lock_Bn(model, keys=7, Exc=1):

    block = list(model._modules.keys())
    keys = block[:keys]
    if Exc == 1:
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm3d) and name.split('.')[0] in keys:
                for p in m.parameters(): p.requires_grad = False
            if isinstance(m, nn.BatchNorm2d) and name.split('.')[0] in keys:
                for p in m.parameters(): p.requires_grad = False
    else:
        print('No to Lock Batch Normlize layer')


def Fix_block(model, keys=7, Exc=1):
    block = list(model._modules.keys())
    keys = block[:keys]
    if Exc == 1:
        for name, value in model.named_parameters():
            if name.split('.')[0] in keys:
                value.requires_grad = False
    else:
        print('Not to Fix any block')


def Open_block(model, keys=7, Exc=1):
    block = list(model.module._modules.keys())         # dataparaline so using mnodel.module
    keys = block[keys:]
    if Exc == 1:
        for name, value in model.module.named_parameters():
            if name.split('.')[0] in keys:     # dataparaline so using [1]
                value.requires_grad = True
        print('Two-Stage:   start to train more', keys)
    else:
        print('===============Two stage still lock===============')


def Lock_BN_Dur_train(model, keys=7, Exc=1):
    block = list(model.module._modules.keys())         # dataparaline so using mnodel.module
    keys = block[:keys]
    if Exc == 1:
        for name, m in model.module.named_modules():
            if isinstance(m, nn.BatchNorm3d) and name.split('.')[0] in keys:
                for p in m.parameters(): p.requires_grad = False
            if isinstance(m, nn.BatchNorm2d) and name.split('.')[0] in keys:
                for p in m.parameters(): p.requires_grad = False
        print('Two-Stage:   Fix some Bn to train')
    else:
        print('===============Two stage still lock===============')