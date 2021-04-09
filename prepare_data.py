"""Read images from data directory, pass them through the pipeline. As the result obtain
the dataset for model training
"""

import os

import numpy as np
import pandas as pd
import torch
from cv2 import cv2

from face_crop import FaceCropper


def read_images():
    x = pd.concat([pd.read_csv('data/labels.csv'),
                   pd.read_csv('data/new_labels.csv')],
                  axis=0, ignore_index=True)

    label_dict = dict(zip(x['filename'], x['mood']))

    images = []
    labels = []
    fc = FaceCropper()

    for root, dirnames, filenames in os.walk('data/'):
        for filename in filenames:
            # filter JPG files
            if not filename.endswith('.JPG'):
                continue

            # get its label
            label = label_dict.get(filename)
            if label is None:
                print(f'label for {filename} not found;  skipping ..')
                continue

            filepath = f'{root}/{filename}'
            img = cv2.imread(filepath)
            if img is None:
                raise FileNotFoundError(filepath)

            # crop face
            img = fc.crop_face(img)

            # convert to gray
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # add to collection
            labels.append(label)
            images.append(img)

    labels = np.array(labels)
    return images, labels


if __name__ == '__main__':
    images, labels = read_images()

    assert images[0].ndim == 2
    assert images[0].dtype == np.uint8
    print('loaded', len(images), 'images')
    print('image shape:', images[0].shape)

    os.makedirs('resources', exist_ok=True)
    torch.save((images, labels), 'resources/myimages.pt')
    print('saved in:', 'myimages.pt')
