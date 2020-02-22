from typing import Tuple, Iterable

import matplotlib.pyplot as plt
import torch

from input_target_transforms import img_norm

"""
Original: https://github.com/fkodom/wnet-unsupervised-image-segmentation
Edited by Gerard
"""
def visualize_outputs(*args: Tuple[Iterable], titles: Iterable = ()) -> None:
    r"""Helper function for visualizing arrays of related images.  Each input argument is expected to be an Iterable of
    images -- shape:  (batch, nchan, nrow, ncol) or shape: (nchan, nrow, ncol).  Will handle both RGB and grayscale images. The i-th elements from all
    input arrays are displayed along a single row, with shared x- and y-axes for visualization.

    :param args: Iterables of related images to display. Suggested: List of images for each Title
    :param titles: Titles to display above each column.
    :return: None (plots the images with Matplotlib)
    """
    nrow, ncol = len(args[0]), len(args)
    fig, ax = plt.subplots(nrow, ncol, sharex='row', sharey='row', squeeze=False)

    for j, title in enumerate(titles[:ncol]):
        ax[0, j].set_title(title)

    for i, images in enumerate(zip(*args)):
        for j, image in enumerate(images):
            image = img_norm(image)
            if len(image.shape) < 3:
                ax[i, j].imshow(image.detach().cpu().numpy())
            else:
                ax[i, j].imshow(image.squeeze(0).permute(1,2,0).detach().cpu().numpy())

    plt.show()

# theory - https://eleanormaclure.files.wordpress.com/2011/03/colour-coding.pdf (page 5)
# kelly's colors - https://i.kinja-img.com/gawker-media/image/upload/1015680494325093012.JPG
# hex values - http://hackerspace.kinja.com/iscc-nbs-number-hex-r-g-b-263-f2f3f4-242-243-244-267-22-1665795040
# gist - https://gist.github.com/ollieglass/f6ddd781eeae1d24e391265432297538
kelly_colors = ['e6194b', '3cb44b', 'ffe119', '4363d8', 'f58231', '911eb4', '46f0f0', 'f032e6', 'bcf60c', 'fabebe', '008080', 'e6beff', '9a6324', 'fffac8', '800000', 'aaffc3', '808000', 'ffd8b1', '000075', '808080', 'ffffff', '000000']

def hex_to_rgb(hex_str):
    return list(int(hex_str[i:i+2], base=16) for i  in (0,2,4))

kelly_rgb = list(map(hex_to_rgb, kelly_colors))

def argmax_to_rgb(mask_pred):
    mask_3d = torch.cat([mask_pred, mask_pred, mask_pred])
    # print(mask_3d.shape)
    # print(mask_3d.unique())
    pred_image = torch.zeros(3, mask_3d.size(1), mask_3d.size(2), dtype=mask_pred.dtype)
    # print(pred_image.shape)
    for k in range(len(kelly_rgb)):
        idx_k = torch.tensor([k,k,k]).view(3,1)
        if k in mask_3d.unique().tolist():
            # print(k)
            # print(idx_k)
            # print(idx_k.shape)
            color_insert = torch.tensor(kelly_rgb[k]).float().view(3,1)
            # print(color_insert.dtype)
            # print(color_insert.shape)
            # print(color_insert)
            hits = torch.where(mask_pred == k)
            # print(hits)
            # for h in hits:
            #     print(h.shape)
            pred_image[:, hits[1], hits[2]] = color_insert
            # pred_image = mask_3d.where(mask_3d[:]==idx_k, color_insert)
            # pred_image[:, mask_3d==k] = torch.tensor(kelly_rgb[k]).view(3,1)

    return pred_image
