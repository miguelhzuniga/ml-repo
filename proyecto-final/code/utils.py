import torch
from torch.utils.data import random_split
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
GENERATOR = torch.Generator().manual_seed(SEED)

def dataset_resume( dataset, name ):
    print(f"\nNumber of images in { name } split: { len( dataset ) }")
    image, mask = dataset[0]
    print(f"[trainval] Image shape: { image.shape }, Mask shape: { mask.size }\n")
# end def

def load_data():
    transform = transforms.ToTensor( )

    trainval_dataset = OxfordIIITPet(
        root='./data/trainval',
        target_types='segmentation',
        transform=transform,
        download=True, 
        split='trainval'
    )
    test_dataset = OxfordIIITPet(
        root='./data/test',
        target_types='segmentation',
        transform=transform,
        download=True, 
        split='test'
    )

    train_size = int( 0.8 * len( trainval_dataset ) )
    val_size = len( trainval_dataset ) - train_size

    train_dataset, val_dataset = random_split( trainval_dataset, [ train_size, val_size ], generator=GENERATOR )

    dataset_resume( train_dataset, "train_dataset" )
    dataset_resume( val_dataset, "val_dataset" )
    dataset_resume( test_dataset, "test_dataset" )

    return ( train_dataset, val_dataset, test_dataset )
# end def

def plot_image( image, mask = None ):
    
    image_np = image.permute(1, 2, 0).numpy()

    if mask:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(image_np)
        ax1.set_title("Imagen Original")
        ax1.axis('off')
        mask_np = np.array(mask)
        ax2.imshow(mask_np, cmap='gray')
        ax2.set_title("Mascara de Segmentación")
        ax2.axis('off')
    else:
        _, ax1 = plt.subplots(1, figsize=(10, 5))
        ax1.imshow(image_np)
        ax1.set_title("Imagen Original")
        ax1.axis('off')

    plt.tight_layout()
    plt.show()
    # end def

# eof - utils.py