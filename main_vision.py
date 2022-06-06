from __future__ import print_function

import argparse
import os
from tqdm import tqdm
import time
import random
import numpy as np
import wandb

import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from models import prompters
from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint, cosine_lr, refine_classname


def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for Vision Models')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epoch5s')

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=40,
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--patience', type=int, default=1000)

    # model
    parser.add_argument('--model', type=str, default=None,
                        choices=['rn50', 'instagram_resnext101_32x8d', 'bit_m_rn50'],
                        help='choose pre-trained model')
    parser.add_argument('--method', type=str, default='padding',
                        choices=['padding', 'random_patch', 'fixed_patch'],
                        help='choose visual prompting method')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')

    # dataset
    parser.add_argument('--root', type=str, default='./data',
                        help='dataset')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')

    # other
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models',
                        help='path to save models')
    parser.add_argument('--image_dir', type=str, default='./save/images',
                        help='path to save images')
    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1,
                        help='number of trials')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--gpu', type=int, default=None,
                        help='gpu to use')
    parser.add_argument('--use_wandb', default=False,
                        action="store_true",
                        help='whether to use wandb')

    args = parser.parse_args()

    args.filename = '{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'. \
        format(args.method, args.prompt_size, args.dataset, args.model,
               args.optim, args.learning_rate, args.weight_decay, args.batch_size, args.trial)

    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    args.image_folder = os.path.join(args.image_dir, args.filename)
    if not os.path.isdir(args.image_folder):
        os.makedirs(args.image_folder)

    return args

best_acc1 = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    global best_acc1, device

    args = parse_option()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # create model
    if args.model == 'rn50':
        model = models.__dict__['resnet50'](pretrained=True).to(device)

    elif args.model == 'instagram_resnext101_32x8d':
        model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl').to(device)

    elif args.model == 'bit_m_rn50':
        import big_transfer.bit_pytorch.models as bit_models
        model = bit_models.KNOWN_MODELS['BiT-M-R50x1'](zero_head=True)
        model.load_from(np.load('BiT-M-R50x1.npz'))
        model = model.to(device)

    model.eval()

    prompter = prompters.__dict__[args.method](args).to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            prompter.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # create data
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CIFAR100(args.root, transform=preprocess,
                             download=True, train=True)

    val_dataset = CIFAR100(args.root, transform=preprocess,
                           download=True, train=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, pin_memory=True,
                              num_workers=args.num_workers, shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size, pin_memory=True,
                            num_workers=args.num_workers, shuffle=False)

    class_names = train_dataset.classes
    class_names = refine_classname(class_names)
    indices = list(range(len(class_names)))

    # define criterion and optimizer
    optimizer = torch.optim.SGD(prompter.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

    cudnn.benchmark = True

    # wandb
    if args.use_wandb:
        wandb.init(project='Visual Prompting')
        wandb.config.update(args)
        wandb.run.name = args.filename
        wandb.watch(prompter, criterion, log='all', log_freq=10)

    if args.evaluate:
        acc1 = validate(indices, val_loader, model, prompter, criterion, args)
        return

    epochs_since_improvement = 0

    for epoch in range(args.epochs):

        # train for one epoch
        train(indices, train_loader, model, prompter, optimizer, scheduler, criterion, epoch, args)

        # evaluate on validation set
        acc1 = validate(indices, val_loader, model, prompter, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': prompter.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best=is_best)

        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"There's no improvement for {epochs_since_improvement} epochs.")

            if epochs_since_improvement >= args.patience:
                print("The training halted by early stopping criterion.")
                break

    wandb.run.finish()


def train(indices, train_loader, model, prompter, optimizer, scheduler, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    prompter.train()

    num_batches_per_epoch = len(train_loader)

    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        images = images.to(device)
        target = target.to(device)

        prompted_images = prompter(images)
        output = model(prompted_images)
        if indices:
            output = output[:,indices]
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            if args.use_wandb:
                wandb.log({
                    'training_loss': losses.avg,
                    'training_acc': top1.avg
                     })

        if i % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': prompter.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, args)

    return losses.avg, top1.avg


def validate(indices, val_loader, model, prompter, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_org = AverageMeter('Original Acc@1', ':6.2f')
    top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1_org, top1_prompt],
        prefix='Validate: ')

    # switch to evaluation mode
    prompter.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):

            images = images.to(device)
            target = target.to(device)
            prompted_images = prompter(images)

            # compute output
            output_prompt = model(prompted_images)
            output_org = model(images)
            if indices:
                output_prompt = output_prompt[:, indices]
                output_org = output_org[:, indices]
            loss = criterion(output_prompt, target)

            # measure accuracy and record loss
            acc1_org = accuracy(output_org, target, topk=(1,))
            acc1_prompt = accuracy(output_prompt, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1_org.update(acc1_org[0].item(), images.size(0))
            top1_prompt.update(acc1_prompt[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
              .format(top1_prompt=top1_prompt, top1_org=top1_org))

        if args.use_wandb:
            wandb.log({
                'val_loss': losses.avg,
                'val_acc_prompt': top1_prompt.avg,
                'val_acc_org': top1_org.avg,
            })

    return top1_prompt.avg


if __name__ == '__main__':
    main()