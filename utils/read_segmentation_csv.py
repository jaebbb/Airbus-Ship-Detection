import pandas as pd
from data import *
import numpy as np

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).transpose()  # Needed to align to RLE direction




def get_segmentation(csvfile,size):

    nan = np.zeros((size[0], size[1]))

    csvfile = pd.read_csv(csvfile, na_filter = False)
    ship_dict = {}
    for index, row in csvfile.iterrows(): 
        if row['EncodedPixels'] == '':
            segmentation = nan
        else:
            segmentation = rle_decode(row['EncodedPixels'], (768,768))

        if row['ImageId'] in ship_dict:
            segmentation = segmentation + ship_dict[row['ImageId']]
        ship_dict[row['ImageId']] = segmentation
    print('get_segmentation load')
    return ship_dict
        

