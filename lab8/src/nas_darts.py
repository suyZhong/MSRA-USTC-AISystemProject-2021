import json
import logging
import functools
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from model import CNN
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = dict()
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res



def data_preprocess(args):
    def cutout_fn(img, length):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img

    augmentation = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    cutout = [functools.partial(
        cutout_fn, length=args.cutout)] if args.cutout else []
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ]

    transform_train = transforms.Compose(augmentation + normalize + cutout)
    transform_test = transforms.Compose(normalize)

    trainset = torchvision.datasets.CIFAR10(
        root=args.dataset, train=True, download=True, transform=transform_train)


    testset = torchvision.datasets.CIFAR10(
        root=args.dataset, train=False, download=True, transform=transform_test)
 
    return trainset, testset

logger = logging.getLogger('nni')

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=8, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--channels", default=16, type=int)
    parser.add_argument('--cutout', default=0, type=int,
                        help='cutout length in data augmentation')
    parser.add_argument("--unrolled", default=False, action="store_true")
    parser.add_argument("--visualization", default=False, action="store_true")
    parser.add_argument("--v1", default=False, action="store_true")
    parser.add_argument('--dataset', default="./data",
                        type=str, help='choose the dataset path')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='number of workers to preprocess data')
    parser.add_argument('--output', default='/output/',type=str,help='output path')

    args = parser.parse_args()
     

    dataset_train, dataset_valid = data_preprocess(args)

    model = CNN(32, 3, args.channels, 10, args.layers)
    criterion = nn.CrossEntropyLoss()

    optim = torch.optim.SGD(model.parameters(), 0.025,
                            momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, args.epochs, eta_min=0.001)

    if args.v1:
        from nni.algorithms.nas.pytorch.darts import DartsTrainer
        trainer = DartsTrainer(model,
                               loss=criterion,
                               metrics=lambda output, target: accuracy(
                                   output, target, topk=(1,)),
                               optimizer=optim,
                               num_epochs=args.epochs,
                               dataset_train=dataset_train,
                               dataset_valid=dataset_valid,
                               batch_size=args.batch_size,
                               log_frequency=args.log_frequency,
                               unrolled=args.unrolled,
                               callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint("./checkpoints")])
        if args.visualization:
            trainer.enable_visualization()

        trainer.train()
    else:
        from nni.retiarii.oneshot.pytorch import DartsTrainer
        trainer = DartsTrainer(
            model=model,
            loss=criterion,
            metrics=lambda output, target: accuracy(output, target, topk=(1,)),
            optimizer=optim,
            num_epochs=args.epochs,
            dataset=dataset_train,
            batch_size=args.batch_size,
            log_frequency=args.log_frequency,
            unrolled=args.unrolled
        )
        trainer.fit()
        final_architecture = trainer.export()
        print('Final architecture:', trainer.export())
        json.dump(trainer.export(), open('/output/checkpoint.json', 'w'))
