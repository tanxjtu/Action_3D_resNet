""" Initilize the datasets module
    New datasets can be added with python scripts under datasets/
"""
import torch
import torch.utils.data
import torch.utils.data.distributed
import importlib


def get_dataset(args, Part_class_list=None):
    dataset = importlib.import_module('.'+args.dataset, package='datasets')
    train_dataset, val_dataset, valvideo_dataset = dataset.get(args, Part_class_list)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    valvideo_loader = torch.utils.data.DataLoader(
        valvideo_dataset, batch_size=valvideo_dataset.testGAP, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, valvideo_loader


def get_dataset_train(args, Part_class_list=None):
    dataset = importlib.import_module('.'+args.dataset, package='datasets')
    train_dataset = dataset.get_train(args, Part_class_list)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    return train_loader


# def get_dataset_val(args, Part_class_list=None):
#     dataset = importlib.import_module('.' + args.dataset, package='datasets')
#     val_dataset = dataset.get_val(args, Part_class_list)
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=args.batch_size, shuffle=True,
#         num_workers=args.workers, pin_memory=True)
#     return val_loader


def get_dataset_val(args, Part_class_list=None):
    dataset = importlib.import_module('.' + args.dataset, package='datasets')
    val_dataset = dataset.get_val(args, Part_class_list)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=15, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    return val_loader


def get_dataset_video(args, Part_class_list=None):
    dataset = importlib.import_module('.' + args.dataset, package='datasets')
    valvideo_dataset = dataset.get_video(args, Part_class_list)
    valvideo_loader = torch.utils.data.DataLoader(
        valvideo_dataset, batch_size=valvideo_dataset.testGAP, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return valvideo_loader