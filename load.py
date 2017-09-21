import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
import tensorflow as tf


def load_graph_weights(): 
	json_file = open('model/fashionMNISTModel.json','r')
	model_json = json_file.read()
	json_file.close()
	model = model_from_json(model_json)
	model.load_weights('model/fashionMNISTModel.h5')
	print("Loaded Model from disk")

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	graph = tf.get_default_graph()

	return model, graph
