import sys

import tensorflow as tf
from torch.utils.data import Dataset
import dask.dataframe as dd
import torch


# Define the Dataset class
class Waymo(Dataset):
    def __init__(self, path, col='[CameraImageComponent].image', resize=(128, 192), train=True):
        print(f"\n\nATTENTION! started initialization waymo", file=sys.stderr, flush=True)
        self.dask_df = dd.read_parquet(path + "/*.parquet", columns=[col]) # read all files
        self.data_iterator = iter(self.dask_df.iterrows())
        self.col = col
        self.resize = resize
        self.train = train
        print(f"\n\nATTENTION! initialized waymo", file=sys.stderr, flush=True)


    def __len__(self):
        print(f"\n\nATTENTION! : computing waymo length ", file=sys.stderr, flush=True)

        #return len(self.dask_df)
        if self.train:
            return 70000
        else:
            return 15000

    def __getitem__(self, idx):
        print(f"\n\nATTENTION! : {idx} ", file=sys.stderr, flush=True)

        _, row = next(self.data_iterator)

        image = tf.image.decode_jpeg(row[self.col])
        image = tf.image.resize(image, self.resize, method='nearest').numpy()

        print(f"\n\nATTENTION! : {image} ", file=sys.stderr, flush=True)
        image = torch.from_numpy(image).float() / 255
        image = image.permute(2, 0, 1)
        print(f"\n\nATTENTION! : {image} ", file=sys.stderr, flush=True)
        print(f"\n\nATTENTION! : {image.shape} ", file=sys.stderr, flush=True)

        item = {'image': image}
        return item

