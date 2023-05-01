import tensorflow as tf
from torch.utils.data import Dataset
import dask.dataframe as dd


# Define the Dataset class
class Waymo(Dataset):
    def __init__(self, path, col='[CameraImageComponent].image', resize=(128, 128)):
        self.dask_df = dd.read_parquet(path + "/*.parquet", columns=[col]) # read all files
        self.data_iterator = iter(self.dask_df.iterrows())
        self.col = col
        self.resize = resize

    def __len__(self):
        return len(self.dask_df)

    def __getitem__(self, idx):
        _, row = next(self.data_iterator)

        image = tf.image.decode_jpeg(row[self.col])
        image = tf.image.resize(image, self.resize, method='nearest')

        return image.numpy()

