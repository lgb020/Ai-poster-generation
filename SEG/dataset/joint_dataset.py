import os
import cv2
import torch
from torch.utils import data
import numpy as np

mainroot = os.path.dirname(os.path.realpath(__file__))
mainroot = "/".join(mainroot.split('/')[:-1])
class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list):
        self.data_root = data_root
        self.data_list = data_list
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size = load_image_test(os.path.join(self.data_root, self.image_list[item]))
        image = torch.Tensor(image)

        return {'image': image, 'name': self.image_list[item % self.image_num], 'size': im_size}

    def __len__(self):
        return self.image_num

def get_loader(pin=False):
    shuffle = False
    test_root = mainroot
    test_list = mainroot + '/test.lst'
    dataset = ImageDataTest(test_root, test_list)
    data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle, num_workers=1, pin_memory=pin)
    return data_loader


def load_image_test(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2,0,1))
    return in_, im_size


