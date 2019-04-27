#%%
import time

class Timing():
    def __init__(self):
        self.now = None
        self.task = None
    def start(self, task):
        # task should be a printable object, i.e. a string
        self.task = task
        self.now = time.process_time()
    def time_str(self, time):
        if (time > 60):
            return str(round(time/60, 2)) + "m"
        elif (time > 1):
            return str(round(time,2)) + "s"
        else:
            return str(round(time*1000, 2)) + "ms"
    def end(self):
        if self.now == None:
            return None
        else:
            delta = time.process_time() - self.now
            print("{:<8} required for [{}]".format(self.time_str(delta), self.task))
            return delta

t = Timing()

t.start("Import TensorFlow")
import tensorflow as tf
t.end()
t.start("Other package imports")
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt 
import numpy as np
import random
t.end()

#%%
t.start("Load MNIST Data")
(train_pics, train_labels), (test_pics, test_labels) = datasets.mnist.load_data()
t.end()
train_pics, test_pics = train_pics / 255, test_pics / 255

#%%
def explore_dataset():
    print("Number of training pictures: {}, Number of testing pictures: {}".format(
        len(train_pics), len(test_pics)
    ))
    """ The training data is stored as 8 bit integers via numpy """
    print(type(train_pics[0][0][0]))
    """ The max value for an 8 bit unsigned integer is 255"""
    print(np.amax(train_pics))

def preview(n):
    """ Take n^2 random items from the sample and visualize them. n must be less than 10 """
    assert n < 10 and n > 0 and int(n) == n
    random_indices = random.sample(range(0, len(train_pics)), n**2)
    print("Displaying items {} from the training pictures.".format(random_indices))

    fig = plt.figure()
    for i, pic_idx in enumerate(random_indices):
        a = fig.add_subplot(n,n,i+1) # number of rows, columns, then index
        plt.axis('off')
        plt.imshow(train_pics[pic_idx])
    plt.show()

explore_dataset()
preview(5)

""" Using the keras API to define the model and each layer """
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
""" the Conv2D layer is a predefined layer from the keras API. All layers are by definition
differentiable, and backpropagation can be carried out to train any sequence of layers in a model."""
