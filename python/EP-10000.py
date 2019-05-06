# ==============================================================================
# MIT License
# 
# Copyright (c) 2019 Beyond-Data-llc
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

from __future__ import absolute_import, division, print_function, unicode_literals
from hyperdash import Experiment
from hyperdash import monitor
from hyperdash import monitor_cell
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

exp = Experiment("EP-10000")

print('###PROGRAM-START###')
print("Compiled using TensorFlow Version", tf.__version__)
print('')
print('###PARAMETERS-BEGIN###')
print('{ name: EP-10000 }')
exp.param("epochs", 10000)
exp.param("relu layer density", 10000)
exp.param("softmax layer density", 1000)
print('###PARAMETERS-END###')
print('')

print('###TEST-START###')


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape
len(train_labels)
train_labels
test_images.shape
len(test_labels)

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dense(1000, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10000)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('###TEST-END###')
print('')
print('###TEST-RESULTS###')
exp.metric("accuracy", (test_acc))
exp.metric("loss", (test_loss))
print('##################')

exp.end()

print('')
print('###PROGRAM-END###')
