# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf

class CIFAR10_Idx:
    def __init__(self, root, data_idx=None, train=True, transform=None, download=False):
        """CIFAR-10 dataset with index to extract subset"""

        self.root = root
        self.data_idx = data_idx
        self.train = train
        self.transform = transform
        self.download = download

        self.data, self.labels = self.__build_cifar_subset__()

    def __build_cifar_subset__(self):
        # if index provided, extract subset, otherwise use the whole set
        (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.cifar10.load_data()

        # Choose the appropriate dataset
        data, labels = (train_data, train_labels) if self.train else (test_data, test_labels)

        # If data_idx is provided, extract the specified subset
        if self.data_idx is not None:
            data = data[self.data_idx]
            labels = labels[self.data_idx]

        return data, labels

    def get_tf_dataset(self, batch_size):
        """Converte il dataset in un tf.data.Dataset"""
        dataset = tf.data.Dataset.from_tensor_slices((self.data, self.labels))
        if self.transform:
            dataset = dataset.map(lambda x, y: (self.transform(x), y))
        return dataset.batch(batch_size)
