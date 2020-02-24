import datetime
import os
import time
import json

import torch
from torch import nn
import torchvision

import mlflow
import mlflow.pytorch

import input_target_transforms as TT
import distributed_utils

from faim_mlflow import get_run_manager
from ml_args import parse_args
from evaluation import evaluate
from models import UNetAuto
from visualize import visualize_outputs, argmax_to_rgb

from datasets import GameImagesDataset, GameFoldersDataset, OverfitDataset, get_dataset

# Reference Training Script and Utils: https://github.com/pytorch/vision/tree/master/references


def train_one_epoch(model, criterion, data_loader, device, optimizer, lr_scheduler, epoch, print_freq, do_logging=False):
    model.train()
    metric_logger = distributed_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', distributed_utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    for data, i in metric_logger.log_every(data_loader, print_freq, header):
        step = epoch * len(data_loader.dataset) + i

        image, target = data['image'], data['target']
        image, target = image.to(device), target.to(device)
        output = model(image)
        total_loss = criterion(output, target)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        lr_scheduler.step()
        metric_logger.update(total_loss=total_loss.item(),
                             lr=optimizer.param_groups[0]["lr"])
        if do_logging and i % print_freq == 0:
            mlflow.log_metric('Loss_Sum/Training',
                              total_loss.item(), global_step=step)
            mlflow.log_metric(
                'Learning_Rate', optimizer.param_groups[0]["lr"], global_step=step)


def main(args):
    if args.output_dir:
        distributed_utils.mkdir(args.output_dir)
    # Setup for Distributed if Available, Else set args.distributed to False
    distributed_utils.init_distributed_mode(args)
    # Use device from args. Locally CPU, with GPU 'cuda', with Distributed 'cuda:x' where x is gpu number
    if not torch.cuda.is_available():
        args.device = 'cpu'
    print(args)
    device = torch.device(args.device)

    # train=True applies augmentations to inputs such as flips and crops
    if args.no_augmentation:
        dataset = get_dataset(args.dataset, "train",
                              TT.get_transform(train=False))
    else:
        dataset = get_dataset(args.dataset, "train",
                              TT.get_transform(train=True))
    dataset_test = get_dataset(
        args.dataset, "val", TT.get_transform(train=False))
    print(f'len train set: {len(dataset)} ; len test set: {len(dataset_test)}')

    # Distributed mode chunks the dataset so that each worker does equal work but doesn't do extra work
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # Configured to fetch the correct batched data
    # Pin Memory should help with shared CUDA data resources
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        pin_memory=True, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        pin_memory=True)

    # Initialize Model, handling distributed as needed
    model = UNetAuto(max_features=512)
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Fetch Model weights from a checkpoint. Checkpoint saved in distributed_utils.py
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    # For analyzing model parameters and saving the master weights
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Add a training image and it's target to tensorboard
    # rand_select = torch.randint(0, len(dataset), (6,)).tolist()
    # train_images = []
    # for idx in rand_select:
    #     data = dataset[idx]
    #     image, target = data['image'], data['target']

    #     train_images.append(image)
    #     train_images.append(target)

    # img_grid = torchvision.utils.make_grid(
    #     train_images, nrow=6, normalize=True)
    # writer.add_image('Random_Train_Sample', img_grid)

    # Same functionality available using evaluation.py
    if args.test_only:
        eval_result = evaluate(model, multi_loss, data_loader_test, device,
                               1, epoch=0, writer=writer, post_visualize=args.do_visualize)
        print(eval_result)
        return

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.parameters()
                    if p.requires_grad]},
    ]

    if args.distributed:
        args.lr = args.lr * args.world_size

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # Don't log to mlflow on distributed training and not master thread
    if ('rank' in args and args.rank != 0):
        args.log_mlflow = False
    run_manager = get_run_manager(args)

    with run_manager as run:
        start_time = time.time()
        for epoch in range(args.epochs):
            visualize_flag = (args.do_visualize and epoch == (args.epochs-1))

            if args.distributed:
                train_sampler.set_epoch(epoch)
            train_one_epoch(model, loss_fn, data_loader, device,
                            optimizer, lr_scheduler, epoch, args.print_freq, args.log_mlflow)
            if data_loader_test is not None:
                result_metric_logger = evaluate(
                    model, loss_fn, data_loader_test, device, args.print_freq, epoch, args.log_mlflow, post_visualize=visualize_flag)
            else:
                result_metric_logger = None

            distributed_utils.save_on_master(
                {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args
                },
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        print('Training time {}'.format(total_time_str))
        world = args.world_size if 'rank' in args else 0

        param_dict = {
            'hparam/epochs': args.epochs,
            'hparam/num_samples': len(dataset),
            'hparam/batch_size': args.batch_size,
            'hparam/lr_start': args.lr,
            'hparam/momentum': args.momentum,
            'hparam/weight_decay': args.weight_decay,
            'hparam/distributed': int(args.distributed),
            'hparam/world_size': world,
        }
        if result_metric_logger is not None:
            result_dict = {
                'results/total_loss': getattr(result_metric_logger, "total_loss").value,
                'results/train_time': total_time_str
            }
        else:
            result_dict = {}
        print(json.dumps(param_dict, indent=2, sort_keys=True))
        print(json.dumps(result_dict, indent=2, sort_keys=True))
        if args.log_mlflow:

            mlflow.log_params(param_dict)
            mlflow.log_metrics(result_dict)


if __name__ == "__main__":
    args = parse_args()

    main(args)
