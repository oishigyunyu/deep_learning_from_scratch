import numpy as np
import tensorflow_datasets as tfds

(ds_train, ds_test), info = tfds.load('mnist', split=['train', 'test'], shuffle_files=True,
                                      as_supervised=True,
                                      with_info=True
                                      )


print()
train_img_array = []
train_label_array = []


class MNIST():
    def __init__(self):
        (self.ds_train, self.ds_test), self.info = tfds.load('mnist',
                                                             split=['train', 'test'],
                                                             shuffle_files=True,
                                                             as_supervised=True,
                                                             with_info=True
                                                             )

    def ds_to_array(self, ds, info, split=None):
        img_array = []
        label_array = []
        if(split is None):
            print(split)
            raise ValueError('invalid value for splits')
        else:
            for image, label in ds.take(info.splits[split].num_examples):
                img_array.append(image.numpy())
                label_array.append(label.numpy())
        img_array = np.array(img_array)
        label_array = np.array(label_array)
        return (img_array, label_array)
    
    def get_dataset(self):
        ds_train = self.ds_to_array(self.ds_train, self.info, split='train')
        ds_test = self.ds_to_array(self.ds_train, self.info, split='test')

        return ds_train, ds_test

