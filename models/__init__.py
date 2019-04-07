'''
    author: Haoliang Tan
    Data: 22/1/2019
    Name: Object Auxiliary
'''
import torch.nn as nn
import torch



def create_model(args, val=0):
    if val == 0:  # train mode
        two_stage = 0
        # from models import resnext
        from models import resnet
        # build model args.inputsize
        if args.arch == 'resnet18':
            model = resnet.resnet18(pretrained=False, sample_size=args.sample_size, sample_duration=args.image_num,
                                     kwargs=args)
            pretrained_model = ''
        elif args.arch == 'resnet50':
            model = resnet.resnet50(pretrained=False, sample_size=args.sample_size, sample_duration=args.image_num,
                                     kwargs=args)
            # pretrained_model = './models/Imgnet_pretrained/resnet50.pth'
            pretrained_model = './models/Imgnet_pretrained/3D224_50preained.pth'
        elif args.arch == 'resnet101':
            model = resnet.resnet101(pretrained=False, sample_size=args.sample_size, sample_duration=args.image_num,
                                     kwargs=args)
            pretrained_model = './models/Imgnet_pretrained/resnet101.pth'
            # pretrained_model = './models/Imgnet_pretrained/274_101_224.pth'
        elif args.arch == 'resnet152':
            model = resnet.resnet152(pretrained=False, sample_size=args.sample_size, sample_duration=args.image_num,
                                     kwargs=args)
            pretrained_model = './models/Imgnet_pretrained/resnet101-5d3b4d8f.pth'

        if pretrained_model != './models/Imgnet_pretrained/resnet50.pth' or \
             pretrained_model != './models/Imgnet_pretrained/resnet101.pth':  # fine tune
            model.fcdown = nn.Linear(in_features=512, out_features=args.nclass)  # fintunde effiect only
            two_stage = 1  # using fine tune model to train

        pretrained_dict = torch.load(pretrained_model)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        two_stage = 0  # no use
        import resnet
        if args.arch == 'resnet18':
            model = resnet.resnet18(pretrained=False, kwargs=args)
        elif args.arch == 'resnet50':
            model = resnet.resnet50(pretrained=False, sample_size=args.sample_size, sample_duration=args.image_num,
                                     kwargs=args)
        elif args.arch == 'resnet101':
            model = resnet.resnet101(pretrained=False, sample_size=args.sample_size, sample_duration=args.image_num,
                                     kwargs=args)
        elif args.arch == 'resnet152':
            model = resnet.resnet152(pretrained=False, kwargs=args)

    # replace last layer
    # model.fc = nn.Linear(in_features=2048, out_features=args.nclass)
    model.fcdown = nn.Linear(in_features=512 , out_features=args.nclass)
    return model, two_stage
