""" Generate SITL selections from NASA MMS1 spacecraft data.

"""

import datetime
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.constants
from keras import backend as K
from spacepy import pycdf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, TimeDistributed
from tensorflow.keras.models import Sequential

__author__ = "Colin Small"
__copyright__ = "Copyright 2019"
__credits__ = ["Colin Small", "Matthew Argall", "Marek Petrik"]
__version__ = "1.0"
__email__ = "crs1031@wildcats.unh.edu"
__status__ = "Production"

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def afg_cdf_to_dataframe(afg_cdf_path):
    """ Converts AFG CDF to a Pandas dataframe.

    Convert a CDF containing AFG data at the provided path to a Pandas dataframe. Only copies measurements from
    mms1_afg_srvy_dmpa.

    Args:
        afg_cdf_path: A path-like object pointing to a CDF file.

    Returns:
        A Pandas dataframe containing data from the CDF indexed by the CDF's Epoch.
    """

    # Open afg CDF
    afg_cdf = pycdf.CDF(str(afg_cdf_path))

    # Create afg dataframe with indexed by the CDF's Epoch
    afg_df = pd.DataFrame()
    afg_df['Epoch'] = afg_cdf['Epoch'][...]
    afg_df.set_index('Epoch', inplace=True)

    # Copy afg data from the CDF to the dataframe
    afg_df['mms1_afg_srvy_dmpa_Bx'] = afg_cdf['mms1_afg_srvy_dmpa'][:, 0]
    afg_df['mms1_afg_srvy_dmpa_By'] = afg_cdf['mms1_afg_srvy_dmpa'][:, 1]
    afg_df['mms1_afg_srvy_dmpa_Bz'] = afg_cdf['mms1_afg_srvy_dmpa'][:, 2]
    afg_df['mms1_afg_srvy_dmpa_|B|'] = afg_cdf['mms1_afg_srvy_dmpa'][:, 3]

    # Computer metaproperties
    afg_df['mms1_afg_magnetic_pressure'] = (afg_df['mms1_afg_srvy_dmpa_|B|'] ** 2) / scipy.constants.mu_0
    afg_df['mms1_afg_clock_angle'] = np.arctan2(afg_df['mms1_afg_srvy_dmpa_By'], afg_df['mms1_afg_srvy_dmpa_Bx'])

    return afg_df


def fpi_des_cdf_to_dataframe(fpi_des_cdf_path):
    """ Converts FPI DES CDF to a Pandas dataframe.

    Convert a CDF containing FPI DES data at the provided path to a Pandas dataframe. Only copies measurements from
    the fpi_des_var_list variable below.

    Args:
        fpi_des_cdf_path: A path-like object pointing to a CDF file.

    Returns:
        A Pandas dataframe containing data from the CDF indexed by the CDF's Epoch.
    """

    fpi_des_var_list = ['mms1_des_energyspectr_omni_fast', 'mms1_des_numberdensity_fast', 'mms1_des_bulkv_dbcs_fast',
                        'mms1_des_heatq_dbcs_fast', 'mms1_des_temppara_fast', 'mms1_des_tempperp_fast']
    fpi_des_var_list_3d = ['mms1_des_prestensor_dbcs_fast', 'mms1_des_temptensor_dbcs_fast']

    # Open FPI DES CDF
    fpi_des_cdf = pycdf.CDF(str(fpi_des_cdf_path))

    # Create afg dataframe with indexed by the CDF's Epoch
    fpi_des_df = pd.DataFrame()
    fpi_des_df['Epoch'] = fpi_des_cdf['Epoch'][...]
    fpi_des_df.set_index('Epoch', inplace=True)

    # Copy afg data from the CDF to the dataframe
    for var in fpi_des_var_list:
        if fpi_des_cdf[var][...].ndim > 1:  # Data includes more than one measurement
            for i in range(0, fpi_des_cdf[var].shape[1] - 1):
                fpi_des_df[var + f'_{i}'] = fpi_des_cdf[var][:, i]
        else:
            fpi_des_df[var] = fpi_des_cdf[var][...]

    for var in fpi_des_var_list_3d:
        fpi_des_df[f'{var}_x1_y1'] = fpi_des_cdf[f'{var}'][:, 0, 0]
        fpi_des_df[f'{var}_x2_y1'] = fpi_des_cdf[f'{var}'][:, 1, 0]
        fpi_des_df[f'{var}_x2_y2'] = fpi_des_cdf[f'{var}'][:, 1, 1]
        fpi_des_df[f'{var}_x3_y1'] = fpi_des_cdf[f'{var}'][:, 2, 0]
        fpi_des_df[f'{var}_x3_y2'] = fpi_des_cdf[f'{var}'][:, 2, 1]
        fpi_des_df[f'{var}_x3_y3'] = fpi_des_cdf[f'{var}'][:, 2, 2]

    # Calculate metafeatures

    fpi_des_df['mms1_des_temp_anisotropy'] = (fpi_des_df['mms1_des_temppara_fast'] /
                                              fpi_des_df['mms1_des_tempperp_fast']) - 1
    fpi_des_df['mms1_des_scalar_temperature'] = (fpi_des_df['mms1_des_temppara_fast'] +
                                                 2 * fpi_des_df['mms1_des_tempperp_fast']) / 3

    prestensor_trace = fpi_des_df['mms1_des_prestensor_dbcs_fast_x1_y1'] + \
                       fpi_des_df['mms1_des_prestensor_dbcs_fast_x2_y2'] + \
                       fpi_des_df['mms1_des_prestensor_dbcs_fast_x1_y1']
    fpi_des_df['mms1_des_scalar_pressure'] = prestensor_trace / 3

    return fpi_des_df


def fpi_dis_cdf_to_dataframe(fpi_dis_cdf_path):
    """ Converts FPI DIS CDF to a Pandas dataframe.

    Convert a CDF containing FPI DIS data at the provided path to a Pandas dataframe. Only copies measurements from
    the fpi_dis_var_list variable below.

    Args:
        fpi_dis_cdf_path: A path-like object pointing to a CDF file.

    Returns:
        A Pandas dataframe containing data from the CDF indexed by the CDF's Epoch.
    """

    fpi_dis_var_list = ['mms1_dis_energyspectr_omni_fast', 'mms1_dis_numberdensity_fast', 'mms1_dis_bulkv_dbcs_fast',
                        'mms1_dis_heatq_dbcs_fast', 'mms1_dis_temppara_fast', 'mms1_dis_tempperp_fast']
    fpi_dis_var_list_3d = ['mms1_dis_prestensor_dbcs_fast', 'mms1_dis_temptensor_dbcs_fast']

    # Open FPI DES CDF
    fpi_dis_cdf = pycdf.CDF(str(fpi_dis_cdf_path))

    # Create afg dataframe with indexed by the CDF's Epoch
    fpi_dis_df = pd.DataFrame()
    fpi_dis_df['Epoch'] = fpi_dis_cdf['Epoch'][...]
    fpi_dis_df.set_index('Epoch', inplace=True)

    # Copy afg data from the CDF to the dataframe
    for var in fpi_dis_var_list:
        if fpi_dis_cdf[var][...].ndim > 1:  # Data includes more than one measurement
            for i in range(0, fpi_dis_cdf[var].shape[1] - 1):
                fpi_dis_df[var + f'_{i}'] = fpi_dis_cdf[var][:, i]
        else:
            fpi_dis_df[var] = fpi_dis_cdf[var][...]

    for var in fpi_dis_var_list_3d:
        fpi_dis_df[f'{var}_x1_y1'] = fpi_dis_cdf['mms1_dis_prestensor_dbcs_fast'][:, 0, 0]
        fpi_dis_df[f'{var}_x2_y1'] = fpi_dis_cdf['mms1_dis_prestensor_dbcs_fast'][:, 1, 0]
        fpi_dis_df[f'{var}_x2_y2'] = fpi_dis_cdf['mms1_dis_prestensor_dbcs_fast'][:, 1, 1]
        fpi_dis_df[f'{var}_x3_y1'] = fpi_dis_cdf['mms1_dis_prestensor_dbcs_fast'][:, 2, 0]
        fpi_dis_df[f'{var}_x3_y2'] = fpi_dis_cdf['mms1_dis_prestensor_dbcs_fast'][:, 2, 1]
        fpi_dis_df[f'{var}_x3_y3'] = fpi_dis_cdf['mms1_dis_prestensor_dbcs_fast'][:, 2, 2]

    # Calculate metafeatures
    fpi_dis_df['mms1_dis_temp_anisotropy'] = (fpi_dis_df['mms1_dis_temppara_fast'] /
                                              fpi_dis_df['mms1_dis_tempperp_fast']) - 1
    fpi_dis_df['mms1_dis_scalar_temperature'] = (fpi_dis_df['mms1_dis_temppara_fast'] +
                                                 2 * fpi_dis_df['mms1_dis_tempperp_fast']) / 3

    return fpi_dis_df


def edp_cdf_to_dataframe(edp_cdf_path):
    """ Converts EDP CDF to a Pandas dataframe.

    Convert a CDF containing EDP data at the provided path to a Pandas dataframe. Only copies measurements from
    the edp_var_list variable below.

    SHOULD NOT BE USED BECAUSE THE EDP CDF DOES NOT CONTAIN THE REQUESTED VARIABLE IN edp_var_list

    Args:
        edp_cdf_path: A path-like object pointing to a CDF file.

    Returns:
        A Pandas dataframe containing data from the CDF indexed by the CDF's Epoch.
    """

    edp_cdf = pycdf.CDF(str(edp_cdf_path))

    # Parse edp
    edp_df = pd.DataFrame()
    edp_df['Epoch'] = edp_cdf['mms1_edp_dce_epoch'][...]
    edp_df.set_index('Epoch', inplace=True)

    edp_df['mms1_edp_x'] = edp_cdf['mms1_edp_dce_xyz_dsl'][:, 0]
    edp_df['mms1_edp_y'] = edp_cdf['mms1_edp_dce_xyz_dsl'][:, 1]
    edp_df['mms1_edp_z'] = edp_cdf['mms1_edp_dce_xyz_dsl'][:, 2]

    # Calculate metafeatures
    edp_df['mms1_edp_|E|'] = np.sqrt(edp_df['mms1_edp_x'] ** 2 + edp_df['mms1_edp_y'] ** 2 + edp_df['mms1_edp_z'] ** 2)

    return edp_df


def concatenate_all_cdf(start_date_str, end_date_str, base_directory_path):
    """ Merge all CDFs into a single dataframe.

    1. Converts each CDF in PROJECT_ROOT/mms_api_downloads to a Pandas dataframe
    2. Merges each dataframe for each instrument into a single, larger dataframe
    3. Downsamples each instruments' dataframe to match indices
    4. Merges each instruments' dataframe into a single dataframe

    EDP IS DISABLED, SEE COMMENT IN EDP HELPER FUNCTION

    Args:
        start_date_str: The first date for the date range, a string given in the format %Y-%m-%d-%H:%M:%S.
        end_date_str: The last date for the date range, a string given in the format %Y-%m-%d-%H:%M:%S.
        base_directory_path: A Path object to the parent .../mms1 directory

    Returns:
        A single merged dataframe with MMS data in the given date range.
    """

    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d-%H:%M:%S")
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d-%H:%M:%S")
    fpi_des_df = merge_fpi_des_dataframes(start_date, end_date, base_directory_path)
    fpi_dis_df = merge_fpi_dis_dataframes(start_date, end_date, base_directory_path)
    afg_df = merge_afg_dataframes(start_date, end_date, base_directory_path)
    edp_df = merge_edp_dataframes(start_date, end_date, base_directory_path)

    # Drop duplicate index entries
    fpi_des_df['index'] = fpi_des_df.index
    fpi_dis_df['index'] = fpi_dis_df.index
    afg_df['index'] = afg_df.index
    edp_df['index'] = edp_df.index

    fpi_des_df = fpi_des_df.drop_duplicates(subset='index', keep='first')
    fpi_dis_df = fpi_dis_df.drop_duplicates(subset='index', keep='first')
    afg_df = afg_df.drop_duplicates(subset='index', keep='first')
    edp_df = edp_df.drop_duplicates(subset='index', keep='first')

    fpi_des_df = fpi_des_df.drop('index', axis=1)
    fpi_dis_df = fpi_dis_df.drop('index', axis=1)
    afg_df = afg_df.drop('index', axis=1)
    edp_df = edp_df.drop('index', axis=1)

    # Sort dataframes
    fpi_des_df = fpi_des_df.sort_index()
    fpi_dis_df = fpi_dis_df.sort_index()
    afg_df = afg_df.sort_index()
    edp_df = edp_df.sort_index()

    # Downsamples dataframes down to match fpi_des timescale
    afg_df = afg_df.reindex(fpi_des_df.index, method='nearest')
    edp_df = edp_df.reindex(fpi_des_df.index, method='nearest')

    # Merge dataframes
    merged_df = fpi_des_df
    merged_df.join(fpi_dis_df, how='outer')
    merged_df.join(afg_df, how='outer')
    merged_df.join(edp_df, how='outer')

    # Computer other metaproperties
    merged_df['mms1_temp_ratio'] = fpi_dis_df['mms1_dis_scalar_temperature'] / fpi_dis_df['mms1_dis_scalar_temperature']
    merged_df['mms1_plasma_beta'] = (fpi_dis_df['mms1_dis_scalar_temperature'] + fpi_dis_df[
        'mms1_dis_scalar_temperature']) \
                                    / edp_df['mms1_edp_|E|']

    return merged_df


def merge_edp_dataframes(start_date, end_date, base_directory_path):
    """ Merge EDP CDFs for the given date range into a single dataframe

     Args:
        start_date: The first date for the date range, a string given in the format "%Y-%m-%d"
        end_date: The last date for the date range, a string given in the format "%Y-%m-%d"
        base_directory_path: A Path object to the parent .../mms1 directory

    Returns:
        A single merged dataframe with EDP instrument data in the given date range.
    """
    # Merge edp dataframes
    edp_directory = base_directory_path / 'edp' / 'fast' / 'ql' / 'dce'
    edp_cdf_glob = edp_directory.glob('**/*.cdf')
    edp_cdf_list = []

    for path in list(edp_cdf_glob):
        date_search = re.search('\d{8}', str(path))
        file_date_string = date_search.group(0)
        file_date = datetime.datetime.strptime(file_date_string, "%Y%m%d")
        if start_date <= file_date <= end_date:
            edp_cdf_list.append(path)

    print("\n   Merging EDP dataframes.")

    edp_df = None
    for file_path in edp_cdf_list:
        if edp_df is None:
            edp_df = edp_cdf_to_dataframe(file_path)
        else:
            edp_df = edp_df.append(edp_cdf_to_dataframe(file_path))

    return edp_df


def merge_fpi_des_dataframes(start_date, end_date, base_directory_path):
    """ Merge FPI DES CDFs for the given date range into a single dataframe

     Args:
        start_date: The first date for the date range, a string given in the format "%Y-%m-%d"
        end_date: The last date for the date range, a string given in the format "%Y-%m-%d"
        base_directory_path: A Path object to the parent .../mms1 directory

    Returns:
        A single merged dataframe with fpi_des instrument data in the given date range.
    """
    # Merge fpi des dataframes
    fpi_des_directory = base_directory_path / 'fpi' / 'fast' / 'ql' / 'des'
    fpi_des_cdf_glob = fpi_des_directory.glob('**/*.cdf')
    fpi_des_cdf_list = []

    for path in list(fpi_des_cdf_glob):
        date_search = re.search('\d{8}', str(path))
        file_date_string = date_search.group(0)
        file_date = datetime.datetime.strptime(file_date_string, "%Y%m%d")
        if start_date <= file_date <= end_date:
            fpi_des_cdf_list.append(path)

    print("\n   Merging FPI DES dataframes.")

    fpi_des_df = None
    for file_path in fpi_des_cdf_list:
        if fpi_des_df is None:
            fpi_des_df = fpi_des_cdf_to_dataframe(file_path)
        else:
            fpi_des_df = fpi_des_df.append(fpi_des_cdf_to_dataframe(file_path))

    return fpi_des_df


def merge_afg_dataframes(start_date, end_date, base_directory_path):
    """ Merge AFG CDFs for the given date range into a single dataframe

     Args:
        start_date: The first date for the date range, a string given in the format "%Y-%m-%d"
        end_date: The last date for the date range, a string given in the format "%Y-%m-%d"
        base_directory_path: A Path object to the parent .../mms1 directory

    Returns:
        A single merged dataframe with AFG instrument data in the given date range.
    """
    # Merge afg dataframes
    afg_directory = base_directory_path / 'afg' / 'srvy' / 'ql'
    afg_cdf_glob = afg_directory.glob(
        '**/*.cdf')  # Creates a generator for all .cdf files in all subdirectories of afg_directory
    afg_cdf_list = []

    for path in list(afg_cdf_glob):
        date_search = re.search('\d{8}', str(path))
        file_date_string = date_search.group(0)
        file_date = datetime.datetime.strptime(file_date_string, "%Y%m%d")
        if start_date <= file_date <= end_date:
            afg_cdf_list.append(path)

    print("\n   Merging AFG dataframes.")

    afg_df = None
    for file_path in afg_cdf_list:
        if afg_df is None:
            afg_df = afg_cdf_to_dataframe(file_path)
        else:
            afg_df = afg_df.append(afg_cdf_to_dataframe(file_path))

    return afg_df


def merge_fpi_dis_dataframes(start_date, end_date, base_directory_path):
    """ Merge FPI DIS CDFs for the given date range into a single dataframe

     Args:
        start_date: The first date for the date range, a string given in the format "%Y-%m-%d"
        end_date: The last date for the date range, a string given in the format "%Y-%m-%d"
        base_directory_path: A Path object to the parent .../mms1 directory

    Returns:
        A single merged dataframe with FPI DIS instrument data in the given date range.
    """
    # Merge fpi dis dataframes
    fpi_dis_directory = base_directory_path / 'fpi' / 'fast' / 'ql' / 'dis'
    fpi_dis_cdf_glob = fpi_dis_directory.glob('**/*.cdf')
    fpi_dis_cdf_list = []

    for path in list(fpi_dis_cdf_glob):
        date_search = re.search('\d{8}', str(path))
        file_date_string = date_search.group(0)
        file_date = datetime.datetime.strptime(file_date_string, "%Y%m%d")
        if start_date <= file_date <= end_date:
            fpi_dis_cdf_list.append(path)

    print("\n   Merging FPI DES dataframes.")

    fpi_dis_df = None
    for file_path in fpi_dis_cdf_list:
        if fpi_dis_df is None:
            fpi_dis_df = fpi_dis_cdf_to_dataframe(file_path)
        else:
            fpi_dis_df = fpi_dis_df.append(fpi_dis_cdf_to_dataframe(file_path))

    return fpi_dis_df


def f1(y_true, y_pred):
    """ Helper function for calculating the f1 score needed for importing the TF Keras model.

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
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def lstm(num_features=55, layer_size=250):
    """ Helper function to define the LSTM used to make predictions.

    """
    model = Sequential()

    model.add(Bidirectional(LSTM(layer_size, return_sequences=True, activation='tanh', recurrent_activation='sigmoid'),
                            input_shape=(None, num_features)))

    model.add(Dropout(0.4))

    model.add(Bidirectional(LSTM(layer_size, return_sequences=True, activation='tanh', recurrent_activation='sigmoid')))

    model.add(Dropout(0.4))

    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    return model

def process(start_date, end_date, base_directory_path):
    # # Define MMS CDF directory location

    # Load model
    print("\nLoading model.")
    model = lstm()
    model.load_weights('model_weights.h5')

    # Load data
    print("\nLoading data:")
    data = concatenate_all_cdf(start_date, end_date, base_directory_path)

    # Interpolate interior values, drop outside rows containing 0s
    print("\nInterpolating NaNs.")
    data = data.interpolate(method='time', limit_area='inside')
    data = data.loc[(data != 0).any(axis=1)]

    # Select data within time range
    data = data.loc[start_date:end_date]
    data_index = data.index

    # Scale data
    print("\nScaling data.")
    scaler = pickle.load(open('scaler.sav', 'rb'))
    data = scaler.transform(data)

    # Run data through model
    print("\nGenerating selection predictions.")
    predictions_list = model.predict(np.expand_dims(data, axis=0))

    # Filter predictions with threshold
    threshold = 0.5
    filtered_output = [0 if x < threshold else 1 for x in predictions_list.squeeze()]

    # Create selections from predictions
    predictions_df = pd.DataFrame()
    predictions_df.insert(0, "time", data_index)
    predictions_df.insert(1, "prediction", filtered_output)
    predictions_df['group'] = (predictions_df.prediction != predictions_df.prediction.shift()).cumsum()
    predictions_df = predictions_df.loc[predictions_df['prediction'] == True]
    selections = pd.DataFrame({'BeginDate': predictions_df.groupby('group').time.first(),
                               'EndDate': predictions_df.groupby('group').time.last()})
    selections = selections.set_index('BeginDate')
    selections['score'] = "Selection score not yet implemented"
    selections['score'] = "Selection description not yet implemented - this is an auto generated description"

    # Output selections
    print("Saving selections to CSV.")
    selections.to_csv(f'gl-mp-unh_{start_date}_{end_date}.csv', header=False)


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Error workaround for running on Mac OS, remove for other systems
process("2017-02-03-10:00:00", "2017-02-05-23:59:59",
        Path("/Users/colinrsmall/Documents/GitHub/sitl-downloader/mms_api_downloads/mms1"))
