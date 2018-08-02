import argparse
import os
import sys
from pkgutil import iter_modules
import numpy as np

import torch
from torchvision import transforms as torchtransforms

from fetalnav.transforms import itk_transforms as itktransforms
from fetalnav.transforms import tensor_transforms as tensortransforms
from fetalnav.datasets.itk_metadata_classification import ITKMetaDataClassification

from fetalnav.models.spn_models import *

import engine as torchengine


def module_exists(module_name):
    return module_name in (name for loader, name, ispkg in iter_modules())


parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. ../data/')
parser.add_argument('--image-size', '-i', default=224, type=int,
                    metavar='N', help='image size (default: 128)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save_model_path', default='logs/ITKMetaDataClassification', type=str, metavar='PATH',
                    help='path to logs (default: none)')
parser.add_argument('--model', default='resnet18', type=str, metavar='MD',
                    help='Model Type to use')
parser.add_argument('--spn', '--soft-proposal', dest='spn', action='store_true',
                    help='Use Soft Proposal Layer')
parser.add_argument('--aspect-ratio', '--aspect', default=1., type=float, metavar='A',
                    help='Natural cropped aspect ratio of the image (default: 1.), tune to 1.5 for Polar images')
parser.add_argument('--tensorboard_path', default=None, type=str, metavar='PATH',
                    help='path to tensorboard logs (default: none)')


def main():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    # create transformation and data augmentation schemes
    resample = itktransforms.Resample(new_spacing=[.5, .5, 1.])
    tonumpy  = itktransforms.ToNumpy(outputtype='float')
    totensor = torchtransforms.ToTensor()
    crop     = tensortransforms.CropToRatio(outputaspect=args.aspect_ratio)
    resize   = tensortransforms.Resize(size=[args.image_size, args.image_size], interp='bilinear')
    rescale  = tensortransforms.Rescale(interval=(0,1))
    flip     = tensortransforms.Flip(axis=2)

    transform = torchtransforms.Compose(
                    [resample,
                     tonumpy,
                     totensor,
                     crop,
                     resize,
                     rescale,
                     flip])
    validation_transform = torchtransforms.Compose(
                            [resample,
                             tonumpy,
                             totensor,
                             crop,
                             resize,
                             rescale])

    # load datasets
    train_dataset = ITKMetaDataClassification(root=args.data, mode='train',    transform=transform)
    val_dataset   = ITKMetaDataClassification(root=args.data, mode='validate', transform=validation_transform)

    # estimate the samples' weights
    train_cardinality = train_dataset.get_class_cardinality()
    val_cardinality = val_dataset.get_class_cardinality()
    train_sample_weights = torch.from_numpy(train_dataset.get_sample_weights())

    print('')
    print('train-dataset: ')
    for idx, c in enumerate(train_dataset.get_classes()):
        print('{}: \t{}'.format(train_cardinality[idx], c))
    print('')
    print('validate-dataset: ')
    for idx, c in enumerate(val_dataset.get_classes()):
        print('{}: \t{}'.format(val_cardinality[idx], c))
    print('')

    # create samplers weighting samples according to the occurence of their respective class
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_sample_weights,
                                                                   int(np.median(train_cardinality)),
                                                                   replacement=True)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers)

    # class labels
    classes = train_dataset.get_classes()
    num_classes = len(classes)

    if   args.model == 'resnet18':
        model = resnet18_sp(num_classes, num_maps=512, in_channels=1)
    elif args.model == 'resnet34':
        model = resnet34_sp(num_classes, num_maps=512, in_channels=1)
    elif args.model == 'vgg13':
        model = vgg13_sp(num_classes, batch_norm=False, num_maps=512, in_channels=1)
    elif args.model == 'vgg13_bn':
        model = vgg13_sp(num_classes, batch_norm=True, num_maps=512, in_channels=1)
    elif args.model == 'vgg16':
        model = vgg16_sp(num_classes, batch_norm=False, num_maps=512, in_channels=1)
    elif args.model == 'vgg16_bn':
        model = vgg16_sp(num_classes, batch_norm=True, num_maps=512, in_channels=1)
    elif args.model == 'alexnet':
        model = alexnet_sp(num_classes, num_maps=512, in_channels=1)
    else:
        print('No network known: {}, possible choices are ffnet|vgg11|vgg16|alexnet|resnet18|densenet'.format(args.model))
        sys.exit(0)

    print(model)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    criterion = nn.MultiLabelSoftMarginLoss()

    logger = None
    if args.tensorboard_path is not None:
        if not os.path.exists(args.tensorboard_path):
            os.makedirs(args.tensorboard_path)
        logger = None
        if module_exists('tensorboardX'):
            from tensorboardX import SummaryWriter
            logger = SummaryWriter(log_dir=args.tensorboard_path)

    state = {'batch_size': args.batch_size,
             'image_size': args.image_size,
             'max_epochs': args.epochs,
             'evaluate': args.evaluate,
             'resume': args.resume,
             'train_transform': transform,
             'val_transform': transform,
             'save_model_path': args.save_model_path,
             'epoch': args.start_epoch,
             'arch': args.model,
             'workers': args.workers,
             'Logger': logger,
             'classes': classes
             }

    if not os.path.exists(state['save_model_path']):
        os.makedirs(state['save_model_path'])

    # launch learning procedure
    engine = torchengine.MultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_loader, val_loader, optimizer)


if __name__ == '__main__':
    main()
