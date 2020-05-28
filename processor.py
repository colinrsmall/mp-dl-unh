""" Generate SITL selections from NASA MMS1 spacecraft data.

"""

print("\n-----------------------------------------------------------------------------------------")

import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disables Tensorflow debugging information
import joblib
import sys
import numpy as np
import pandas as pd
import requests
requests.adapters.DEFAULT_RETRIES = 5
import argparse
import tempfile
import shutil
import glob

import mp_dl_unh_data
import pymms
from pymms.sdc import selections as sel

from keras import backend as K
from pathlib import Path
from sklearn.metrics import f1_score
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, TimeDistributed, CuDNNLSTM
from tensorflow.keras.models import Sequential

__author__ = "Colin Small"
__copyright__ = "Copyright 2019"
__credits__ = ["Colin Small", "Matthew Argall", "Marek Petrik"]
__version__ = "1.0"
__email__ = "crs1031@wildcats.unh.edu"
__status__ = "Production"

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

model_dir = os.fspath(BASE_DIR / 'model/')

dropbox_dir = pymms.config['dropbox_root'] if pymms.config['dropbox_root'] is not None else ""


def f1(y_true, y_pred):
    """ Helper function for calculating the f1 score needed for importing the TF Keras model.model.

    Args:
        y_true: A tensor with ground truth values.
        y_pred: A tensor with predicted truth values.

    Returns:
        A float with the f1 score of the two tensors.
    """

    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def lstm(num_features=129, layer_size=128):
    """ Helper function to define the LSTM used to make predictions.

    """
    model = Sequential()

    model.add(
        Bidirectional(LSTM(layer_size, return_sequences=True), input_shape=(None, num_features)))

    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    return model


def gpu_lstm(num_features=129, layer_size=128):
    """ Helper function to define the LSTM used to make predictions.

    """
    model = Sequential()

    model.add(
        Bidirectional(CuDNNLSTM(layer_size, return_sequences=True), input_shape=(None, num_features)))

    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    return model


def roundTime(dt=None, dateDelta=datetime.timedelta(minutes=1)):
    """Round a datetime object to a multiple of a timedelta
    dt : datetime.datetime object, default now.
    dateDelta : timedelta object, we round to a multiple of this, default 1 minute.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
            Stijn Nevens 2014 - Changed to use only datetime objects as variables
    """
    roundTo = dateDelta.total_seconds()

    if dt == None : dt = datetime.datetime.now()
    seconds = (dt - dt.min).seconds
    # // is a floor division, not a comment on following line:
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)


def fix_date_intervals(data_index):
    """
    Temporary workaround for 4.5 second intervals between selection dates.
    """
    dates = []
    for index, date in enumerate(data_index):
        if index % 2 == 0:
            dates.append(date + datetime.timedelta(seconds=1))
        else:
            dates.append(date)
    return dates


def process(start_date, end_date, spacecraft, gpu, test=False):
    # # Define MMS CDF directory location
    # Load model.model
    print(f"Loading model.model. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    model = gpu_lstm() if gpu else lstm()
    model.load_weights(model_dir + '/model_weights.h5')

    # Load data
    print(f"Loading data: | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    data = mp_dl_unh_data.get_data(spacecraft, 'sitl', start_date, end_date, False, False)

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.interpolate(method='time', limit_area='inside')

    # Temporary workaround for 4.5 second time cadence of data not working with selections.combine_selections
    data = data.resample("5S").pad()
    data = data.dropna()
    data_index = data.index

    # Scale data
    print(f"Scaling data. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    scaler = joblib.load(open(model_dir + '/scaler.pkl', 'rb'))
    data = scaler.transform(data)

    # Run data through model.model
    print(f"Generating selection predictions. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    predictions_list = model.predict(np.expand_dims(data, axis=0))

    # Filter predictions with threshold
    threshold = 0.95
    filtered_output = [0 if x < threshold else 1 for x in predictions_list.squeeze()]

    # Return predictions if testing
    if test:
        return data_index, filtered_output

    # Create selections from predictions
    selections = pd.DataFrame()
    selections.insert(0, "tstart", data_index)
    selections.insert(1, "tstop", data_index)
    selections.insert(2, "prediction", filtered_output)
    selections['FOM'] = "150.0" # This is a placeholder for the FOM
    selections['description'] = "MP crossing (automatically generated)"
    selections['createtime'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    selections = selections[selections['prediction'] == 1]

    return selections


def chunk_process(start_date, end_date, spacecraft, gpu, chunks, delete_after_chunk, clear_temp):
    for i, (start, end) in enumerate(chunk_date_range(start_date, end_date, chunks)):
        selections = process(start, end, spacecraft, gpu)
        file_name = f'gls_selections_mp-dl-unh_chunk_{i}.csv'

        print(f"Saving selections to CSV: {dropbox_dir + file_name}, chunk {i} of {chunks}| {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")

        if not selections.empty:
            temp_path = Path(tempfile.gettempdir()) / Path(file_name)
            selections.to_csv(temp_path, index=False)
            selections = sel.read_csv(temp_path)
            sel.combine_segments(selections, 5)
            sel.write_csv(dropbox_dir + file_name, selections)

        if delete_after_chunk:
            shutil.rmtree(pymms.config['data_root'])

        if clear_temp:
            files = glob.glob(tempfile.gettempdir() + "/*")
            for f in files:
                os.remove(f)




def chunk_date_range(start, end, interval):
    diff = (end - start) / interval
    s = start
    for i in range(interval):
        yield s, (start + diff * i)
        s = start + diff * i


def test(gpu):
    """
    Test the model through January of 2018.
    """
    validation_data = mp_dl_unh_data.get_data("mms1", 'sitl', "2018-01-01", "2018-01-02", True, True)
    validation_data = validation_data.resample("5s").pad().dropna()
    validation__y = validation_data['selected']
    test_index, test_y = process("2018-01-01", "2018-01-02", "mms1", gpu, True)
    return f1_score(validation__y.astype(int), test_y)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("start",
                         help="Start date of data interval, formatted as either '%%Y-%%m-%%d' or '%%Y-%%m-%%dT%%H:%%M:%%S'. Optionally an integer, interpreted as an orbit number.",
                         type=mp_dl_unh_data.validate_date)
    parser.add_argument("end",
                         help="Start date of data interval, formatted as either '%%Y-%%m-%%d' or '%%Y-%%m-%%dT%%H:%%M:%%S'. Optionally an integer, interpreted as an orbit number.",
                         type=mp_dl_unh_data.validate_date)
    parser.add_argument("sc", help="Spacecraft IDs ('mms1', 'mms2', 'mms3', 'mms4')")
    parser.add_argument("-g", "-gpu", help="Enables use of GPU-accelerated model for faster predictions. Requires CUDA installed.", action="store_true")
    parser.add_argument("-t", "-test", help="Runs a test routine on the model.", action="store_true")
    parser.add_argument("-c", "-chunks", help="Break up the processing of the date interval in n chunks.", type=int)
    parser.add_argument("-temp", help="If running the job in chunks, deletes the contents of the MMS root data folder after each chunk.", action="store_true")

    args = parser.parse_args()

    if pymms.load_config() is None:
        print("Calling this function requires a valid config.ini so that the program knows where to download the SDC CDFs to.")
        exit(-1)

    sc = args.sc
    start = args.start
    end = args.end
    gpu = args.g
    t = args.t
    chunks = args.c
    temp = args.temp

    if t:
        print(f"Model F1 score: {test(gpu)}")

    if sc not in ["mms1", "mms2", "mms3", "mms4"]:
        print("Error: Invalid spacecraft entered.")
        print(f'Expected one of [ "mms1", "mms2", "mms3", "mms4" ], got {sc}.')
        sys.exit(166)

    if chunks:
        chunk_process(start, end, sc, gpu, chunks, temp, True)

    else:
        selections = process(start, end, sc, gpu)

        if not selections.empty:
            current_datetime = datetime.datetime.now()
            selections_filetime = current_datetime.strftime('%Y-%m-%d-%H-%M-%S')
            file_name = f'gls_selections_mp-dl-unh_{selections_filetime}.csv'

            # Output selections
            print(f"Saving selections to CSV: {dropbox_dir + file_name} | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
            # selections.to_csv(dropbox_dir + file_name, header=False)
            temp_path = Path(tempfile.gettempdir()) / Path(file_name)
            selections.to_csv(temp_path, index=False)
            selections = sel.read_csv(temp_path)
            sel.combine_segments(selections, 5)
            sel.write_csv(dropbox_dir + file_name, selections)

    print(f"Done | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    sys.exit(0)


if __name__ == '__main__':
    main()
