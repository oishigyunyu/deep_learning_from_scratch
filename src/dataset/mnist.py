import tensorflow_datasets as tfds
import numpy as np


class MNIST():
    def __init__(self):
        (self.ds_train, self.ds_test), self.info = tfds.load('mnist',
                                                             split=['train', 'test'],
                                                             shuffle_files=True,
                                                             as_supervised=True,
                                                             with_info=True
                                                             )

    def ds_to_array(self, ds, info, flatten, normalize, one_hot, split=None):
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
        if(flatten):
            img_array = img_array.reshape(info.splits[split].num_examples, -1)
        
        if(normalize):
            img_array = img_array / 255.0
        if(one_hot):
            T = np.zeros((label_array.size, 10))
            for idx, row in enumerate(T):
                row[label_array[idx]] = 1

            label_array = T

        return (img_array, label_array)

    def get_dataset(self, flatten, normalize, one_hot):
        ds_train = self.ds_to_array(self.ds_train, self.info, flatten, normalize, one_hot, split='train')
        ds_test = self.ds_to_array(self.ds_train, self.info, flatten, normalize, one_hot, split='test')

        return ds_train, ds_test
