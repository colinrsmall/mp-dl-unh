from pathlib import Path
from sklearn import preprocessing
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time

# TODO: Figure to reproduce: http://www.ssl.berkeley.edu/~moka/eva/img/2019/2019-04-02_132303_mms1.png

"""
Testing model selection and performance with Keras.

Future testing:
https://machinelearningmastery.com/gentle-introduction-backpropagation-time/ <- for long sequence inputs to LSTMs
https://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/
"""

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Model hyperparamters
SEQ_LEN = 250 # Defines the length of the input sequence to the LSTM, generally between 250 and 500: https://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/
EPOCHS = 10
BATCH_SIZE = 64
LAYER_SIZE = 128

def preprocess(mms_data):
	"""
	Preprocess the mms_data
	"""
	
	# Load data
	mms_data.dropna(inplace=True)
	scaler = preprocessing.StandardScaler()
	sitl_windows = pd.read_csv('https://lasp.colorado.edu/mms/sdc/public/service/latis/mms_events_view.csv?source=BDM&event_type=sitl_window',
							   infer_datetime_format=True, parse_dates=[0,1])
	
	# Standardize data
	index = mms_data.index
	selections = mms_data.pop("selected")
	column_names = mms_data.columns
	mms_data = scaler.fit_transform(mms_data)
	mms_data = pd.DataFrame(mms_data, index, column_names)
	mms_data = mms_data.join(selections)

	# Normalize data
	# TODO?

	# Break data up into orbits
	orbits = []
	for start, end in zip(sitl_windows.iloc[:, 0], sitl_windows.iloc[:, 1]):
		orbit = mms_data[start:end]
		if not orbit.empty:
			orbits.append(orbit)

	# Split orbits into sequences
	X_train, X_test, y_train, y_test = [], [], [], []

	sequences = []
	for i in range(len(orbits)):
		X_sequence = deque(maxlen=SEQ_LEN)
		y_sequence = deque(maxlen=SEQ_LEN)

		if i < int(0.8*len(orbits)): # First 80% of orbits for training data
			for value in orbits[i].values:
				X_sequence.append(value[:-1])
				y_sequence.append(value[-1])
				if len(X_sequence) == SEQ_LEN:
					X_train.append(X_sequence)
					y_train.append(y_sequence)

		else: # Remaining 20% of orbits for test data
			for value in orbits[i].values:
				X_sequence.append(value[:-1])
				y_sequence.append(value[-1])
				if len(X_sequence) == SEQ_LEN:
					X_test.append(X_sequence)
					y_test.append(y_sequence)

	return np.array(X_train), np.array(X_test), y_train, y_test


def lstm(epochs, batch_size, X_train, X_test, y_train, y_test, layer_size):
	"""
	Train the LSTM
	"""
	model_name = f"{SEQ_LEN}-SEQ_LEN-{batch_size}-BATCH_SIZE-{layer_size}-LAYER_SIZE-{int(time.time())}"
	model = Sequential()
	model.add(LSTM(layer_size, input_shape=(X_train.shape[1:]), return_sequences=True, activation="relu"))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(LSTM(layer_size, input_shape=(X_train.shape[1:]), return_sequences=True, activation="relu"))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(LSTM(2, input_shape=(X_train.shape[1:]), return_sequences=True, activation="softmax"))

	opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

	model.compile(loss='sparse_categorical_crossentropy',
				  optimizer=opt,
				  metrics=['accuracy'])

	tensorboard = TensorBoard(log_dir=BASE_DIR/'logs'/model_name)

	filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
	checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True,
														  mode='max'))  # saves only the best ones

	history = model.fit(
		X_train, y_train,
		batch_size=batch_size,
		epochs=epochs,
		validation_data=(X_test, y_test),
		callbacks=[tensorboard, checkpoint]
	)


mms_data = pd.read_csv(BASE_DIR/'export'/'csv'/'mms_with_selections.csv', index_col=0, infer_datetime_format=True,
						   parse_dates=[0])
X_train, X_test, y_train, y_test = preprocess(mms_data)
lstm(EPOCHS, BATCH_SIZE, X_train, X_test, y_train, y_test, LAYER_SIZE)
