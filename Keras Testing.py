from pathlib import Path
from sklearn import preprocessing
from collections import deque
import pandas as pd
import numpy as np
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

def preprocess():
	"""
	Preprocess the mms_data
	"""
	
	# Load data
	mms_data = pd.read_csv(BASE_DIR/'export'/'csv'/'mms_with_selections.csv', index_col=0, infer_datetime_format=True,
						   parse_dates=[0])
	mms_data.dropna(inplace=True)
	scaler = preprocessing.StandardScaler()
	sitl_windows = pd.read_csv('https://lasp.colorado.edu/mms/sdc/public/service/latis/mms_events_view.csv?source=BDM&event_type=sitl_window',
							   infer_datetime_format=True, parse_dates=[0,1])
	
	# Scale data
	index = mms_data.index
	selections = mms_data.pop("selected")
	column_names = mms_data.columns
	mms_data = scaler.fit_transform(mms_data)
	mms_data = pd.DataFrame(mms_data, index, column_names)
	mms_data = mms_data.join(selections)

	# Normalize data
	# TODO

	# Break data up into orbits
	orbits = []
	for start, end in zip(sitl_windows.iloc[:, 0], sitl_windows.iloc[:, 1]):
		orbit = mms_data[start:end]
		if not orbit.empty:
			orbits.append(orbit)

	# Split orbits into sequences
	sequences = []
	for orbit in orbits:
		sequence = deque(maxlen=SEQ_LEN)
		for i in orbit.values:
			sequence.append(i)
			if len(sequence) == SEQ_LEN:
				sequences.append(sequence)

	# Split into training days and test days
	# TODO: rewrite to split sequences list

	#test_y = [train_day.pop("selected") for train_day in train_x]
	#train_y = [test_day.pop("selected") for test_day in train_x]
	return X, y


preprocess()
