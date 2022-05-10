from importlib.resources import path
import numpy as np 

import PIL.Image as Image

import tensorflow as tf
import tensorflow_hub as hub

import labels
import os
import subprocess as sp

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from flowers import get_flower_paths, FLOWERS_LABELS

# Read labels
LABELS = labels.readLabels()

# Size of the image accepted by the classifier
IMAGE_SHAPE = (224,224)

# Adds one more dimension with 3 values for the RGB channel
CLASSIFIER_INPUT_SHAPE = IMAGE_SHAPE + (3,)

# Loads an image at a given path and converts it to a numpy array that can be
# passed as input to the network
def load_image(path: str) -> np.ndarray:
     img = Image.open(path).resize(IMAGE_SHAPE) # rescale the image
     img = np.array(img) / 255.0 # normalize the image
     return img

# Predicts the classification of a specific input image
def predict_classification(path: str, classifier: Sequential) -> str:
     img = load_image(path)
     # adds a new dimension, because classifier accepts multiple images as input
     img = img[np.newaxis, ...]
     result = classifier.predict(img)
     print(result)
     pos = np.argmax(result)
     classification = labels.getLabel(LABELS, pos)
     return classification

# Prints tensorflow info
tensorflow_version = tf.__version__
gpus = tf.config.list_physical_devices("GPU")
cuda = sp.getoutput("nvcc --version | grep \"Build cuda\"")
print(f"tensorflow: {tensorflow_version}\ngpu: {gpus}\ncuda: {cuda}")


print(tf.config.list_physical_devices("GPU"))
print(os.system("nvcc --version | grep \"Build cuda\""))
tf.test.gpu_device_name()


# Initialize the classifier
classifier = Sequential([
     hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=CLASSIFIER_INPUT_SHAPE)
])

# download flowers dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url,  cache_dir='.', untar=True)

paths = get_flower_paths(data_dir)
inputs, outputs = [], []

# generate the training dataset with all the flowers
for flower in paths:
     for flower_path in paths[flower]:
          inputs.append(load_image(flower_path))
          outputs.append(FLOWERS_LABELS[flower])


inputs = np.array(inputs)
outputs = np.array(outputs)

'''
Generating the new model
'''

from sklearn.model_selection import train_test_split

# generate teh training dataset out of the inputs and outputs
input_train, input_test, output_train, output_test = train_test_split(inputs, outputs, random_state=0)


# takes the frozen layers
feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

# initialize the model
pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

# create the last layer for the classifier
OUTPUT_LAYER_SIZE = 5

model = tf.keras.Sequential([
  pretrained_model_without_top_layer,
  tf.keras.layers.Dense(OUTPUT_LAYER_SIZE)
])
# data about the model
model.summary()

# training
model.compile(
  optimizer="adam",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

model.fit(np.array(input_train), output_train, epochs=10)



model.evaluate(input_test, output_test)
LABELS = labels.readFlowerLabels()
LABELS
predict_classification("assets\\rose.jpg", model)
predict_classification("assets\\daisy.jpg", model)
predict_classification("assets\\tulip.jpg", model)
predict_classification("assets\\dandelion.jpg", model)
predict_classification("assets\\sunflower.jpg", model)

'''
Trains the entire model
'''

from sklearn.model_selection import train_test_split

# generate teh training dataset out of the inputs and outputs
input_train, input_test, output_train, output_test = train_test_split(inputs, outputs, random_state=0)


# takes the frozen layers
feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

# initialize the model
pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=True)

# create the last layer for the classifier
OUTPUT_LAYER_SIZE = 5

model = tf.keras.Sequential([
  pretrained_model_without_top_layer,
  tf.keras.layers.Dense(OUTPUT_LAYER_SIZE)
])
# data about the model
model.summary()

# training
model.compile(
  optimizer="adam",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

model.fit(np.array(input_train), output_train, epochs=10)



model.evaluate(input_test, output_test)