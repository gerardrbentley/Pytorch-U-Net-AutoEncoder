from torch import Tensor
from PIL import Image
import glob
import os

import torch

import input_target_transforms as TT
from visualize import visualize_outputs

# Expects directory of .png's as root
# Transforms should include ToTensor (also probably Normalize)
# Can apply different transform to output, returns image as input and label
class GameImagesDataset(torch.utils.data.Dataset):
    def __init__(self, root='/faim/datasets/per_game_screenshots/super_mario_bros_3_usa', train_or_val="train", transform=TT.ToTensor()):
        self.image_dir = os.path.join(root)
        # Get abs file paths
        self.image_list = glob.glob(f'{self.image_dir}/*.png')
        
        if train_or_val == "val":
            self.image_list = self.image_list[:int(len(self.image_list) * 0.2)]
        # self.image_folders = next(os.walk(self.image_dir))[1]
        self.length = len(self.image_list)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        screenshot_file = self.image_list[idx]

        image = Image.open(screenshot_file).convert('RGB')
        target = image.copy()

        if self.transform:
            image, target = self.transform(image, target)

        sample = {'image': image, 'target': target}
        return sample

class GameFoldersDataset(torch.utils.data.Dataset):
    def __init__(self, root='/faim/datasets/per_game_screenshots/', train_or_val="train", transform=TT.ToTensor()):
        self.image_folders = next(os.walk(root))[1]
        sets = []
        for game_folder in self.image_folders:
            sets.append(GameImagesDataset(root=os.path.join(root, game_folder), train_or_val=train_or_val, transform=transform))
        
        self.full_dataset = torch.utils.data.ConcatDataset(sets)
        if train_or_val == "val":
            to_chunk = int(0.2 * len(self.full_dataset))
            self.full_dataset, _ = torch.utils.data.random_split(self.full_dataset, [to_chunk, len(self.full_dataset) - to_chunk])
        self.transform = transform

    def __len__(self):
        return len(self.full_dataset)

    def __getitem__(self, idx):
        return self.full_dataset[idx]

class OverfitDataset(torch.utils.data.Dataset):
    def __init__(self, root='./overfit.png', train_or_val="train", transform=TT.ToTensor(), num_images=2000):
        self.image_file = root
        self.length = num_images
        if train_or_val == "val":
            self.length = 3
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        image = Image.open(self.image_file).convert('RGB')

        target = image.copy()

        if self.transform:
            image, target = self.transform(image, target)

        sample = {'image': image, 'target': target}
        return sample

def get_dataset(name, train_or_val, transform):
    paths = {
        "overfit": ('./overfit.png', OverfitDataset),
        "test_mario": ('/faim/datasets/test_mario', GameImagesDataset),
        "mario": ('/faim/datasets/mario_images', GameFoldersDataset),
        "blap": ('/faim/datasets/blap_images', GameFoldersDataset),
        "icarus": ('/faim/datasets/per_game_screenshots/kid_icarus_usa_europe', GameImagesDataset),
    }
    p, ds_fn = paths[name]

    ds = ds_fn(root=p, train_or_val=train_or_val, transform=transform)
    return ds


if __name__ == "__main__":
    print('test Game Image Dataset')

    trainset = GameImagesDataset(root='/faim/datasets/mario_images', train_or_val='train', transform=None)
    print(f'len trainset: {len(trainset)}')
    data = trainset[1]
    # data['image'].show()
    image = data['image']
    target = data['target']
    
    print(f'Image and Target with Transform = None')
    print(f'types: {type(image)}, {type(target)}')
    print(f'shapes: {(image.size)}, {(target.size)}')
    print(f'extrema: [{image.getextrema()}], [{target.getextrema()}]')

    do_transforms = TT.get_transform(False)
    image, target = do_transforms(image, target)
    
    print(f'Image and Target with Transform = val')
    print(f'types: {type(image)}, {type(target)}')
    print(f'shapes: {(image.shape)}, {(target.shape)}')
    print(f'ranges: [{torch.min(image).item()} - {torch.max(image).item()}], [{torch.min(target).item()} - {torch.max(target).item()}]')

    image, target = image.unsqueeze(0), target.unsqueeze(0)
    
    print(f'Image and Target post Batching')
    print(f'shapes: {(image.shape)}, {(target.shape)}')


    image = image.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    print(f'Image and Target post detach, cpu, numpy for viz')
    print(f'types: {type(image)}, {type(target)}')
    print(f'shapes: {(image.shape)}, {(target.shape)}')
    print(f'ranges: [{image.min()} - {image.max()}], [{target.min()} - {target.max()}]')
    
    visualize_outputs(image, target, titles=['Image', 'Target'])
