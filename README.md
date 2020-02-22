# Game Texture Segmentation

Experiments using UNet Architectures for Video Game Image Auto-Encoding tasks

## Install
If working with [conda](https://docs.conda.io/en/latest/miniconda.html) you can use the following to set up a virtual python environment.
```
conda create --name mldev python=3.8
conda activate mldev
```
Then you can use pip install to get all the dependencies (this works with virtualenv as well)
```
pip install -r requirements.txt
```

## Overfit Training

A good practice of testing a new model is getting it to Overfit a sample dataset. In our case we want one image to be encoded, decoded, and segmented extremely well.

In `datasets.py` is an `OverfitDataset` that defaults to using the image `overfit.png` 2000 times per epoch (and 3 times for validation / evaluation loop because distributed training requires at least one sample per GPU).

Recommended transforms for this model: 
```
    import input_target_transforms as TT
    transforms = []

    # Input size of model is 224x224
    transforms.append(TT.Resize(224))
    # PIL Images to Torch Tensors
    transforms.append(TT.ToTensor())
    # Normalize Images for better Gradient Descent
    transforms.append(TT.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    transforms = TT.Compose(transforms)
```
** TODO ** Check the mean and std on our dataset, these are the values to Normalize over ImageNet (Or some other natural image datasets I believe)

Running the command `python train.py --dataset overfit --epochs 5 --no-aug` should start the training on your machine. `python train.py --help` will show all available Command Line Flags (or look at the `parse_args()` function of `ml_args.py`). You may need to lower --batch-size and --workers depending on your computer's computing abilities.

## Distributed Training

Training on multiple GPUS is simple using pytorch's distributed launch.py utility

`python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --use_env train.py --relevant --training --flags`

## Resources

Pytorch Training Utils and Distributed Utils references: https://github.com/pytorch/vision/tree/master/references/segmentation


## Papers
U-Net: https://arxiv.org/pdf/1505.04597.pdf
W-Net: https://arxiv.org/pdf/1711.08506.pdf
