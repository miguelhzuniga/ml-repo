import sys
sys.path.append( './code' )
from utils import load_data, plot_image

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = load_data( )

    train_image, train_mask = train_dataset[0]
    plot_image( train_image, train_mask )
    plot_image( train_image )

# end if
#eof - main.py