"""Define the Recurrent Image Annotator model."""
from __future__ import division

import time
import json

import numpy as np
import chainer

from tqdm import tqdm
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class RIA(Chain):
    """Recurrent Image Annotator model."""

    def __init__(self, dict_size, embed_dim, hid_dim, image_dim):
        """Define the model.

        parameters:
            embed_dim: dimensionality of label embedding
            hid_dim: dimensionality of LSTM hidden layer
            image_dim: dimensionality of image features
        """
        super(RIA, self).__init__(
            embed=L.EmbedID(dict_size + 1, embed_dim),
            lstm=L.LSTM(embed_dim, hid_dim, forget_bias_init=1.0),
            fc1=L.Linear(hid_dim, hid_dim),
            fc2=L.Linear(hid_dim, dict_size + 1),
            image_embedding=L.Linear(image_dim, hid_dim),
        )

    def reset_state(self):
        """Reset h0 and c0."""
        self.lstm.reset_state()

    def initialize_state(self, image):
        """Embed image features to be used as h0."""
        h = self.image_embedding(image)
        self.lstm.h = h

    def __call__(self, label_input):
        """Make RIA callable.

        Input:
            label_input: list of labels in a single image
        Output:
            output: logits of each label candidate
        """
        embed_label = self.embed(label_input)
        h = self.lstm(embed_label)
        mid = F.relu(self.fc1(h))
        output = self.fc2(mid)
        return output
