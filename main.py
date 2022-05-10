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

import re

# Prints build information.
tensorflow_version = tf.__version__
gpus = tf.config.list_physical_devices("GPU")
cuda = sp.getoutput("nvcc --version | grep \"Build cuda\"")
print(f"tensorflow: {tensorflow_version}\ngpu: {gpus}\ncuda: {cuda}")
tf.test.gpu_device_name()

# The dimensions of the image accepted by the network. The last dimension
# allows to provide a value for each of the RGB channels of coloured images.
IMAGE_SHAPE = (224,224)
CLASSIFIER_INPUT_SHAPE = IMAGE_SHAPE + (3,)

# Loads an image at a given path and converts it to a numpy array that can be
# passed as input to the network
def load_image(path: str) -> np.ndarray:
	"""
	Loads an image at a given path and converts it to a numpy array that can be
	passed as input to the network
	"""
	img = Image.open(path).resize(IMAGE_SHAPE) # rescale the image
	img = np.array(img) / 255.0 # normalize the image
	return img

def predict_classification(path: str, labels_list: list[str], classifier: Sequential) -> str:
	"""
	Predicts the classification of a specific input image to a specific class
	from the list of labels
	"""
	img = load_image(path)
	
	# Adds a new dimension because the classifier accepts multiple images as
	# input.
	img = img[np.newaxis, ...]
	
	result = classifier.predict(img)
	pos = np.argmax(result)
	classification = labels.getLabel(labels_list, pos)

	return classification

# Set up the label lists.
MOBILENET_LABELS = labels.readLabels()
FLOWER_LABELS = labels.readFlowerLabels()

"""
Setting up the Mobilenet classifier
"""

# Initialize the MobileNet classifier
mobilenet_classifier = Sequential([
     hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=CLASSIFIER_INPUT_SHAPE)
])

# Predict classification for some sample inputs.
predict_classification("assets\\bananas.jpg", MOBILENET_LABELS, mobilenet_classifier)
predict_classification("assets\\peacock.jpg", MOBILENET_LABELS, mobilenet_classifier)
predict_classification("assets\\mushrooms.jpg", MOBILENET_LABELS, mobilenet_classifier)
predict_classification("assets\\violin.jpg", MOBILENET_LABELS, mobilenet_classifier)
predict_classification("assets\\desk.jpg", MOBILENET_LABELS, mobilenet_classifier)

'''
Generating the new model
'''

# Downloads the flowers dataset.
flowers_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
flowers_dir = tf.keras.utils.get_file('flower_photos', origin=flowers_url,  cache_dir='.', untar=True)
flowers_paths_dict = get_flower_paths(flowers_dir)

# This represents an input list with flower images and an output list with
# categories matching each flower.
inputs, outputs = [], []

# Generates the training dataset with all the flowers
for flower in flowers_paths_dict:
	for flower_path in flowers_paths_dict[flower]:
		inputs.append(load_image(flower_path))
		outputs.append(FLOWERS_LABELS[flower])

# Convert the inputs and outputs to np.arrays to pass them as training data
# to the model.
inputs = np.array(inputs)
outputs = np.array(outputs)


from sklearn.model_selection import train_test_split

# Generates a training and testing dataset for the input images and output 
# categories.
input_train, input_test, output_train, output_test = train_test_split(inputs, outputs, random_state=0)


# Retrieves the feature vector of the Mobilenet model.
feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

# Initializes the input and hidden layer of the new classifier.
pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

# Sets up the last layer for the classifier.

OUTPUT_LAYER_SIZE = 5

flower_classifier = tf.keras.Sequential([
  pretrained_model_without_top_layer,
  tf.keras.layers.Dense(OUTPUT_LAYER_SIZE)
])

# Prints data about the model parameters.
flower_classifier.summary()

# Training on the training dataset.
flower_classifier.compile(
  optimizer="adam",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

flower_classifier.fit(np.array(input_train), output_train, epochs=10)

# Evaluates the accuracy of the model on the testing dataset.
flower_classifier.evaluate(input_test, output_test)

# Sample flower predictions
predict_classification("assets\\rose.jpg", FLOWER_LABELS, flower_classifier)
predict_classification("assets\\daisy.jpg", FLOWER_LABELS, flower_classifier)
predict_classification("assets\\tulip.jpg", FLOWER_LABELS, flower_classifier)
predict_classification("assets\\dandelion.jpg", FLOWER_LABELS, flower_classifier)
predict_classification("assets\\sunflower.jpg", FLOWER_LABELS, flower_classifier)
