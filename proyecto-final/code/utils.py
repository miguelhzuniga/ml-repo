import torch
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as TF
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import random

SEED = 42
GENERATOR = torch.Generator( ).manual_seed( SEED )

'''
'''
def dataset_resume( dataset, name ):
    print( f"\nNumber of images in { name } split: { len( dataset ) }" )
    image, mask = dataset[ 0 ]
    print( f"[{ name }] Image shape: { image.shape }, Mask shape: { mask.shape }\n" )
# end def

'''
'''
def target_transform( mask ):
    mask_tensor = TF.pil_to_tensor( mask ).squeeze( 0 )
    binary_mask = ( ( mask_tensor == 1 ) | ( mask_tensor == 3 ) ).long( )
    return binary_mask
# end def

'''
'''
def find_low_info_masks( dataset, threshold = 0.99, plot_hist = False ):
    indices = [ ]
    zero_ratios = [ ]

    for idx in range( len( dataset ) ):
        _, mask = dataset[ idx ]
        total_pixels = mask.numel( )
        zero_pixels = ( mask == 0 ).sum( ).item( )
        zero_ratio = zero_pixels / total_pixels
        zero_ratios.append( zero_ratio )

        if ( zero_ratio >= threshold ):
            indices.append( idx )

    if plot_hist:
        plt.figure(figsize=(8, 4))
        plt.hist(zero_ratios, bins=30, color='skyblue', edgecolor='black')
        plt.title("Distribución del porcentaje de píxeles con valor 0 en las máscaras")
        plt.xlabel("Proporción de píxeles en 0 (fondo)")
        plt.ylabel("Número de imágenes")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    print(f"\nFound { len( indices ) } masks with ≥ { int( threshold * 100 ) }% zeros\n" )
    return indices
# end def

'''
'''
def exclude_indices(dataset, indices_to_exclude):
    all_indices = set(range(len(dataset)))
    keep_indices = sorted(all_indices - set(indices_to_exclude))
    return Subset(dataset, keep_indices)
# end def

'''
'''
def load_data( H = None, W = None, display_low_info = False ):
    transform = transforms.ToTensor( )

    trainval_dataset = OxfordIIITPet(
        root='./data/trainval',
        target_types='segmentation',
        transform=transform,
        target_transform=target_transform,
        download=True, 
        split='trainval'
    )
    test_dataset = OxfordIIITPet(
        root='./data/test',
        target_types='segmentation',
        transform=transform,
        target_transform=target_transform,
        download=True, 
        split='test'
    )

    dataset_resume( trainval_dataset, 'train_dataset' )
    dataset_resume( test_dataset, 'test_dataset' )

    low_info_train = find_low_info_masks( trainval_dataset, plot_hist = True )
    low_info_test = find_low_info_masks( test_dataset )
    
    if display_low_info:
        random.shuffle(low_info_train)
        for idx in low_info_train[:5]:
            image, mask = trainval_dataset[idx]
            plot_image(image, True, mask)

    trainval_dataset_clean = exclude_indices(trainval_dataset, low_info_train)
    test_dataset_clean = exclude_indices(test_dataset, low_info_test)

    dataset_resume(trainval_dataset_clean, 'trainval_dataset_clean')
    dataset_resume(test_dataset_clean, 'test_dataset_clean')

    return ( trainval_dataset_clean, test_dataset_clean )
# end def

'''
'''
def plot_image( image, plot_mask = False, mask = None ):
    
    image_np = image.permute( 1, 2, 0 ).numpy( )

    if plot_mask:
        _, ( ax1, ax2 ) = plt.subplots( 1, 2, figsize = ( 10, 5 ) )
        ax1.imshow( image_np )
        ax1.set_title( 'Imagen Original' )
        ax1.axis( 'off' )
        mask_np = np.array( mask )
        ax2.imshow( mask_np, cmap='gray' )
        ax2.set_title( 'Mascara de Segmentación' )
        ax2.axis('off')
    else:
        _, ax1 = plt.subplots(1, figsize=(10, 5))
        ax1.imshow(image_np)
        ax1.set_title( 'Imagen Original' )
        ax1.axis( 'off' )

    plt.tight_layout( )
    plt.show( )
    # end def

def flatten_dataset( dataset, batch_size = 64 ):
    loader = DataLoader( dataset, batch_size=batch_size, shuffle = False )
    all_images = [ ]
    all_masks = [ ]

    for images, masks in loader:
        images_flat = torch.flatten(images, start_dim=1)
        masks_flat = torch.flatten(masks, start_dim=1).float()

        all_images.append(images_flat)
        all_masks.append(masks_flat)

    X = torch.cat(all_images, dim=0)
    y = torch.cat(all_masks, dim=0)
    return X, y

# eof - utils.py