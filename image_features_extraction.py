import json

import numpy as np
import cv2
import chainer
from tqdm import tqdm

from RIA import VGGNet


def get_image_features(image_path, vgg):
    """Load an image and extract the image features.""" 
    # training images pixel mean
    mean = np.array([103.939, 116.779, 123.68])
    img = cv2.imread(image_path).astype(np.float32)
    img -= mean
    # [224, 224, 3] to [3, 224, 224]
    img = cv2.resize(img, (224, 224)).transpose((2, 0, 1))
    # input for VGGNet should be [1, 3, 224, 224]
    img = img[np.newaxis, :, :, :]
    img = chainer.cuda.cupy.asarray(img, dtype=np.float32)
    img = chainer.Variable(img, volatile='on')
    image_features = vgg(img).data
    return image_features


dataset = "iaprtc12"

with open(dataset + '/train_image_list.json') as f:
    train_image_list = json.load(f)

with open(dataset + '/test_image_list.json') as f:
    test_image_list = json.load(f)
    

# load pretrained VGG model
vgg = VGGNet()
chainer.serializers.load_hdf5('VGG.model', vgg)
vgg.to_gpu()

train_image_features = []
for image_path in tqdm(train_image_list):
    image_path = dataset + '/' + image_path
    image_features = get_image_features(image_path, vgg)
    image_features = chainer.cuda.to_cpu(image_features)
    train_image_features.append(image_features)
train_image_features = np.stack(train_image_features)
print train_image_features.shape

test_image_features = []
for image_path in tqdm(test_image_list):
    image_path = dataset + '/' + image_path
    image_features = chainer.cuda.to_cpu(get_image_features(image_path, vgg))
    test_image_features.append(image_features)
test_image_features = np.stack(test_image_features)
print test_image_features.shape

with open(dataset + "_train_image_features.npy", "w") as f:
    np.save(f, train_image_features)

with open(dataset + "_test_image_features.npy", "w") as f:
    np.save(f, test_image_features)
