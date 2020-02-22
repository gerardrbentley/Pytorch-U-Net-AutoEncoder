import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Unsupervised Game Segmentation Training')

    # Data and Model
    parser.add_argument('-d', '--dataset', default='overfit', help='dataset')
    parser.add_argument('-m', '--model', default='wnet', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')

    # Hyper Parameters
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('-e', '--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate')
    parser.add_argument('-mo', '--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    # Logging and Saving
    parser.add_argument('-p', '--print-freq', default=5,
                        type=int, help='print frequency')
    parser.add_argument('-o', '--output-dir',
                        default='./output/', help='path where to save')
    parser.add_argument('-t', '--tensorboard', default='',
                        help='project name to save tensorboard output')
    parser.add_argument('-c', '--comment', default='',
                        help='extra comment for tensorboard file name')
    parser.add_argument('-cp', '--checkpoint', default='',
                        help='resume from checkpoint')

    # Boolean Flags
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--no-aug",
        dest="no_augmentation",
        help="Don't augment training images",
        action="store_true",
    )
    parser.add_argument(
        "--log-tensorboard",
        dest="log_tensorboard",
        help="Don't record training to tensorboard",
        action="store_true",
    )
    parser.add_argument(
        "--visualize",
        dest="do_visualize",
        help="Visualize the model outputs after train / test",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    args = parser.parse_args()
    return args
