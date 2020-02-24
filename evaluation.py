
import torch
import torchvision
import mlflow

import input_target_transforms as TT
import distributed_utils

from faim_mlflow import get_run_manager
from datasets import get_dataset
from visualize import argmax_to_rgb
from ml_args import parse_args
from models import UNetAuto


def evaluate(model, criterion, data_loader, device, print_freq, epoch=0, do_logging=False, post_visualize=False):
    model.eval()
    metric_logger = distributed_utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    with torch.no_grad():
        if post_visualize:
            inputs = []
            targets = []
            reconstructions = []
        for data, i in metric_logger.log_every(data_loader, print_freq, header):

            step = epoch * len(data_loader.dataset) + i

            image, target = data['image'], data['target']
            image, target = image.to(device), target.to(device)
            output = model(image)
            total_loss = criterion(output, target)

            loss_item = total_loss.item()
            metric_logger.update(total_loss=loss_item)

            if do_logging:
                mlflow.add_metric('Loss_Total/Validation',
                                  loss_item, global_step=step)

                # TODO: write function to visualize mask and output in eval loop
                # if i < 5:
                #     input_images = [image, target]
                #     img_grid = torchvision.utils.make_grid(
                #         torch.cat(input_images), nrow=2, normalize=True)
                #     writer.add_image(
                #         f'Validation_Sample_{i}/Input_And_Target', img_grid, global_step=step)

                #     reconstruction = TT.img_norm(reconstruction)
                #     writer.add_image(
                #         f'Validation_Sample_{i}/AE_Reconstruction', reconstruction, global_step=step)

                #     if post_visualize:
                #         inputs.append(image)
                #         targets.append(target)
                #         reconstructions.append(reconstruction)

        if post_visualize:
            from visualize import visualize_outputs
            visualize_outputs(inputs, targets, reconstructions, titles=[
                              'Input', 'Target', 'AutoEncoder'])

    return metric_logger


def main(args):
    if args.output_dir:
        distributed_utils.mkdir(args.output_dir)
    # Setup for Distributed if Available, Else set args.distributed to False
    distributed_utils.init_distributed_mode(args)
    print(args)
    # Use device from args. Locally CPU, with GPU 'cuda', with Distributed 'cuda:x' where x is gpu number
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
    model = UNetAuto()
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Fetch Model weights from a checkpoint. Checkpoint saved in distributed_utils.py
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    # Don't log to mlflow on distributed training and not master thread
    if ('rank' in args and args.rank != 0):
        args.log_mlflow = False
    run_manager = get_run_manager(args)
    with run_manager as run:
        eval_result = evaluate(model, nn.MSELoss(), data_loader_test, device,
                            1, epoch=0, writer=writer, post_visualize=args.do_visualize)
        print(eval_result)
    return 1


if __name__ == "__main__":
    args = parse_args()

    main(args)
