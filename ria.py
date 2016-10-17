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


def compute_loss(inputs, targets):
    """Compute cross-entropy loss."""
    loss = 0
    correct_pred = 0
    for input, target in zip(inputs, targets):
        output = RIA(input)
        loss += F.loss.softmax_cross_entropy.softmax_cross_entropy(output, target)
        pred = output.data.argmax(1)
        if pred == target.data:
            correct_pred += 1
    loss /= len(targets)
    # accuracy over one training example/image
    accuracy = correct_pred / len(targets)
    return loss, accuracy


ria = RIA()
model = L.Classifier(ria)
model.to_gpu()
optimizer = optimizers.Adam(1e-3)
optimizer.setup(model)

def train(num_epoch, display_interval=500):
    """Train the model."""
    for epoch in xrange(num_epoch):
        start = time.clock();
        loss = 0
        display_loss = 0
        accuracy = 0
        display_accuracy = 0
        size = len(train_label_inputs)
        for i, (x, t) in tqdm(enumerate(zip(train_label_inputs, train_label_targets))):
            ria.reset_state()
            image_input = Variable(train_image_features[i].reshape([1,-1]))
            rnn.initialize_state(image_input)
            rnn.zerograds()
            display_loss, display_accuracy = compute_loss(x, t)
            display_loss += loss.data
            display_accuracy += accuracy
            if i % display_interval == 0 and i != 0:
                print "iteration: %4d, loss: %.3f, accuracy: %.3f, time used: %.3f" % (
                    i, display_loss / display_interval , display_accuracy / display_interval, time.clock() - start)
                display_start = time.clock()
                loss += display_loss
                accuracy += display_accuracy
                display_loss = 0
                display_accuracy = 0
            loss.backward()
            optimizer.update()
        print "Epoch %3d, loss: %.3f, accuracy: %.3f, time used: %.3f" % (
                epoch, display_loss / size, display_accuracy / size, time.clock() - start)
