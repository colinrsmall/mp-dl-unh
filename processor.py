""" Generate SITL selections from NASA MMS1 spacecraft data.

"""

import datetime
import os
import pickle
import sys
import hashlib
import numpy as np
import pandas as pd
import requests
requests.adapters.DEFAULT_RETRIES = 5
import configparser

import scipy.constants
from keras import backend as K
from pathlib import Path
from sklearn.metrics import f1_score
from spacepy import pycdf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, TimeDistributed
from tensorflow.keras.models import Sequential

from tai import utc2tai

__author__ = "Colin Small"
__copyright__ = "Copyright 2019"
__credits__ = ["Colin Small", "Matthew Argall", "Marek Petrik"]
__version__ = "1.0"
__email__ = "crs1031@wildcats.unh.edu"
__status__ = "Production"

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def query_sdc(base_directory_path, start_date, end_date, spacecraft, instrument, data_level, data_rate_mode, username,
              password,
              descriptor=None):
    """ Query the SDC for a list of CDFs matching parameters.

    Args:
        base_directory_path: A Path object pointing to the root of the MMS data folder (just "/' at the SDC)
        start_date: The first date for the date range, a datimetime object in the format %Y-%m-%dT%H:%M:%S.
        end_date: The last date for the date range, a datimetime object in the format %Y-%m-%dT%H:%M:%S.
        spacecraft: Spacecraft string. One of [ "mms1", "mms2", "mms3", "mms4" ].
        instrument: Which instrument's CDFs are requested. A string, one of [ "afg", "fpi", "edp" ].
        data_level: Requested data level.
        data_rate_mode: Requested data rate mode. One of [ "slow", "fast", "srvy", "brst" ].
        descriptor: Optional descriptor, used for requesting FPI and EDP files. One of [ none, "dis", "des", "dce" ].

    Returns:
        A list of Path objects to all CDFs matching the query parameters.
    """

    start_date_string = start_date.strftime("%Y-%m-%d")
    end_date_string = end_date.strftime("%Y-%m-%d")

    request_url_base = "https://lasp.colorado.edu/mms/sdc/sitl/files/api/v1/file_names/science?"
    request_url = '&'.join(
        [request_url_base, f"start_date={start_date_string}", f"end_date={end_date_string}", f"sc_id={spacecraft}",
         f"instrument_id={instrument}", f"data_level={data_level}", f"data_rate_mode={data_rate_mode}"])
    if descriptor is not None:
        request_url += f"&descriptor={descriptor}"

    request = requests.get(request_url, timeout=2, auth=(username, password))
    request.raise_for_status()

    if len(request.text) == 0:
        print(f"Error: No CDFs found for request: {request_url} | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        sys.exit(3)

    paths_string_list = request.text.split(",")

    paths_list = [Path(base_directory_path / Path(path)) for path in paths_string_list]

    return paths_list


def afg_cdf_to_dataframe(afg_cdf_path, spacecraft):
    """ Converts AFG CDF to a Pandas dataframe.

    Convert a CDF containing AFG data at the provided path to a Pandas dataframe. Only copies measurements from
    mms1_afg_srvy_dmpa.

    Args:
        afg_cdf_path: A path-like object pointing to a CDF file.
        spacecraft: Spacecraft string. One of [ "mms1", "mms2", "mms3", "mms4" ]

    Returns:
        A Pandas dataframe containing data from the CDF indexed by the CDF's Epoch.
    """

    # Open afg CDF
    try:
        afg_cdf = pycdf.CDF(str(afg_cdf_path))
    except pycdf.CDFError as e:
        print(f"e | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f"File path: {afg_cdf_path} | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        sys.exit(3)

    # Create afg dataframe with indexed by the CDF's Epoch
    afg_df = pd.DataFrame()
    afg_df['Epoch'] = afg_cdf['Epoch'][...]
    afg_df.set_index('Epoch', inplace=True)

    # Copy afg data from the CDF to the dataframe
    try:
        afg_df[f'{spacecraft}_afg_srvy_dmpa_Bx'] = afg_cdf[f'{spacecraft}_afg_srvy_dmpa'][:, 0]
        afg_df[f'{spacecraft}_afg_srvy_dmpa_By'] = afg_cdf[f'{spacecraft}_afg_srvy_dmpa'][:, 1]
        afg_df[f'{spacecraft}_afg_srvy_dmpa_Bz'] = afg_cdf[f'{spacecraft}_afg_srvy_dmpa'][:, 2]
        afg_df[f'{spacecraft}_afg_srvy_dmpa_|B|'] = afg_cdf[f'{spacecraft}_afg_srvy_dmpa'][:, 3]

        # Compute metaproperties
        afg_df[f'{spacecraft}_afg_magnetic_pressure'] = (afg_df[
                                                             f'{spacecraft}_afg_srvy_dmpa_|B|'] ** 2) / scipy.constants.mu_0
        afg_df[f'{spacecraft}_afg_clock_angle'] = np.arctan2(afg_df[f'{spacecraft}_afg_srvy_dmpa_By'],
                                                             afg_df[f'{spacecraft}_afg_srvy_dmpa_Bx'])

        # Compute Bz quality
        M = 2
        smoothed_data = [afg_df[f'{spacecraft}_afg_srvy_dmpa_Bx'][0]]
        for i, value in enumerate(afg_df[f'{spacecraft}_afg_srvy_dmpa_Bx'][1:]):
            smoothed_data.append((smoothed_data[i - 1] * (2 ** M - 1) + value) / 2 ** M)
        diff = np.subtract(afg_df[f'{spacecraft}_afg_srvy_dmpa_Bx'], smoothed_data)
        afg_df[f'{spacecraft}_afg_Bz_Q'] = np.absolute(diff)

    except KeyError as e:
        print(f"{e} | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f"For {afg_cdf_path}. Skipping file. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        return pd.DataFrame()

    # Compute metaproperties
    afg_df[f'{spacecraft}_afg_magnetic_pressure'] = (afg_df[
                                                         f'{spacecraft}_afg_srvy_dmpa_|B|'] ** 2) / scipy.constants.mu_0
    afg_df[f'{spacecraft}_afg_clock_angle'] = np.arctan2(afg_df[f'{spacecraft}_afg_srvy_dmpa_By'],
                                                         afg_df[f'{spacecraft}_afg_srvy_dmpa_Bx'])

    return afg_df


def fpi_des_cdf_to_dataframe(fpi_des_cdf_path, spacecraft):
    """ Converts FPI DES CDF to a Pandas dataframe.

    Convert a CDF containing FPI DES data at the provided path to a Pandas dataframe. Only copies measurements from
    the fpi_des_var_list variable below.

    Args:
        fpi_des_cdf_path: A path-like object pointing to a CDF file.
        spacecraft: Spacecraft string. One of [ "mms1", "mms2", "mms3", "mms4" ]

    Returns:
        A Pandas dataframe containing data from the CDF indexed by the CDF's Epoch.
    """

    fpi_des_var_list = [f'{spacecraft}_des_energyspectr_omni_fast', f'{spacecraft}_des_numberdensity_fast',
                        f'{spacecraft}_des_bulkv_dbcs_fast',
                        f'{spacecraft}_des_heatq_dbcs_fast', f'{spacecraft}_des_temppara_fast',
                        f'{spacecraft}_des_tempperp_fast']
    fpi_des_var_list_3d = [f'{spacecraft}_des_prestensor_dbcs_fast', f'{spacecraft}_des_temptensor_dbcs_fast']

    # Open FPI DES CDF
    try:
        fpi_des_cdf = pycdf.CDF(str(fpi_des_cdf_path))
    except pycdf.CDFError as e:
        print(f"{e} | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f"File path: {fpi_des_cdf_path} | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        sys.exit(3)

    # Create afg dataframe with indexed by the CDF's Epoch
    fpi_des_df = pd.DataFrame()
    fpi_des_df['Epoch'] = fpi_des_cdf['Epoch'][...]
    fpi_des_df.set_index('Epoch', inplace=True)

    # Copy afg data from the CDF to the dataframe
    try:
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

        # Compute metafeatures
        fpi_des_df[f'{spacecraft}_des_temp_anisotropy'] = (fpi_des_df[f'{spacecraft}_des_temppara_fast'] /
                                                           fpi_des_df[f'{spacecraft}_des_tempperp_fast']) - 1
        fpi_des_df[f'{spacecraft}_des_scalar_temperature'] = (fpi_des_df[f'{spacecraft}_des_temppara_fast'] +
                                                              2 * fpi_des_df[f'{spacecraft}_des_tempperp_fast']) / 3

        prestensor_trace = fpi_des_df[f'{spacecraft}_des_prestensor_dbcs_fast_x1_y1'] + \
                           fpi_des_df[f'{spacecraft}_des_prestensor_dbcs_fast_x2_y2'] + \
                           fpi_des_df[f'{spacecraft}_des_prestensor_dbcs_fast_x1_y1']
        fpi_des_df[f'{spacecraft}_des_scalar_pressure'] = prestensor_trace / 3

        M = 2
        # Compute N quality
        smoothed_data = [fpi_des_df[f'{spacecraft}_des_numberdensity_fast'][0]]
        for i, value in enumerate(fpi_des_df[f'{spacecraft}_des_numberdensity_fast'][1:]):
            smoothed_data.append((smoothed_data[i-1]*(2**M-1)+value)/2**M)
        diff = np.subtract(fpi_des_df[f'{spacecraft}_des_numberdensity_fast'], smoothed_data)
        fpi_des_df[f'{spacecraft}_des_N_Q'] = np.absolute(diff)

        # Compute Vz quality
        Vz = fpi_des_cdf[f'{spacecraft}_des_bulkv_dbcs_fast'][:,2]
        smoothed_data = [Vz[0]]
        for i, value in enumerate(Vz[1:]):
            smoothed_data.append((smoothed_data[i-1]*(2**M-1)+value)/2**M)
        diff = np.subtract(Vz, smoothed_data)
        fpi_des_df[f'{spacecraft}_des_Vz_Q'] = np.absolute(diff)

    except KeyError as e:
        print(f"{e} | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f"For {fpi_des_cdf_path}. Skipping file. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        return pd.DataFrame()

    return fpi_des_df


def fpi_dis_cdf_to_dataframe(fpi_dis_cdf_path, spacecraft):
    """ Converts FPI DIS CDF to a Pandas dataframe.

    Convert a CDF containing FPI DIS data at the provided path to a Pandas dataframe. Only copies measurements from
    the fpi_dis_var_list variable below.

    Args:
        fpi_dis_cdf_path: A path-like object pointing to a CDF file.
        spacecraft: Spacecraft string. One of [ "mms1", "mms2", "mms3", "mms4" ]

    Returns:
        A Pandas dataframe containing data from the CDF indexed by the CDF's Epoch.
    """

    fpi_dis_var_list = [f'{spacecraft}_dis_energyspectr_omni_fast', f'{spacecraft}_dis_numberdensity_fast',
                        f'{spacecraft}_dis_bulkv_dbcs_fast',
                        f'{spacecraft}_dis_heatq_dbcs_fast', f'{spacecraft}_dis_temppara_fast',
                        f'{spacecraft}_dis_tempperp_fast']
    fpi_dis_var_list_3d = [f'{spacecraft}_dis_prestensor_dbcs_fast', f'{spacecraft}_dis_temptensor_dbcs_fast']

    # Open FPI DES CDF
    try:
        fpi_dis_cdf = pycdf.CDF(str(fpi_dis_cdf_path))
    except pycdf.CDFError as e:
        print(f"{e} | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f"File path: {fpi_dis_cdf_path} | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        sys.exit(3)

    # Create afg dataframe with indexed by the CDF's Epoch
    fpi_dis_df = pd.DataFrame()
    fpi_dis_df['Epoch'] = fpi_dis_cdf['Epoch'][...]
    fpi_dis_df.set_index('Epoch', inplace=True)

    # Copy afg data from the CDF to the dataframe
    try:
        for var in fpi_dis_var_list:
            if fpi_dis_cdf[var][...].ndim > 1:  # Data includes more than one measurement
                for i in range(0, fpi_dis_cdf[var].shape[1] - 1):
                    fpi_dis_df[var + f'_{i}'] = fpi_dis_cdf[var][:, i]
            else:
                fpi_dis_df[var] = fpi_dis_cdf[var][...]

        for var in fpi_dis_var_list_3d:
            fpi_dis_df[f'{var}_x1_y1'] = fpi_dis_cdf[f'{spacecraft}_dis_prestensor_dbcs_fast'][:, 0, 0]
            fpi_dis_df[f'{var}_x2_y1'] = fpi_dis_cdf[f'{spacecraft}_dis_prestensor_dbcs_fast'][:, 1, 0]
            fpi_dis_df[f'{var}_x2_y2'] = fpi_dis_cdf[f'{spacecraft}_dis_prestensor_dbcs_fast'][:, 1, 1]
            fpi_dis_df[f'{var}_x3_y1'] = fpi_dis_cdf[f'{spacecraft}_dis_prestensor_dbcs_fast'][:, 2, 0]
            fpi_dis_df[f'{var}_x3_y2'] = fpi_dis_cdf[f'{spacecraft}_dis_prestensor_dbcs_fast'][:, 2, 1]
            fpi_dis_df[f'{var}_x3_y3'] = fpi_dis_cdf[f'{spacecraft}_dis_prestensor_dbcs_fast'][:, 2, 2]

        # Compute metafeatures
        fpi_dis_df[f'{spacecraft}_dis_temp_anisotropy'] = (fpi_dis_df[f'{spacecraft}_dis_temppara_fast'] /
                                                           fpi_dis_df[f'{spacecraft}_dis_tempperp_fast']) - 1
        fpi_dis_df[f'{spacecraft}_dis_scalar_temperature'] = (fpi_dis_df[f'{spacecraft}_dis_temppara_fast'] +
                                                              2 * fpi_dis_df[f'{spacecraft}_dis_tempperp_fast']) / 3

        # Compute N quality
        M = 2
        smoothed_data = [fpi_dis_df[f'{spacecraft}_dis_numberdensity_fast'][0]]
        for i, value in enumerate(fpi_dis_df[f'{spacecraft}_dis_numberdensity_fast'][1:]):
            smoothed_data.append((smoothed_data[i-1]*(2**M-1)+value)/2**M)
        diff = np.subtract(fpi_dis_df[f'{spacecraft}_dis_numberdensity_fast'], smoothed_data)
        fpi_dis_df[f'{spacecraft}_dis_N_Q'] = np.absolute(diff)

        # Compute Vz quality
        Vz = fpi_dis_cdf[f'{spacecraft}_dis_bulkv_dbcs_fast'][:,2]
        smoothed_data = [Vz[0]]
        for i, value in enumerate(Vz[1:]):
            smoothed_data.append((smoothed_data[i-1]*(2**M-1)+value)/2**M)
        diff = np.subtract(Vz, smoothed_data)
        fpi_dis_df[f'{spacecraft}_dis_Vz_Q'] = np.absolute(diff)

        # Compute n|V| quality
        V = np.sqrt([ m[0]**2 + m[1]**2 + m[2]**2 for m in fpi_dis_cdf['mms1_dis_bulkv_dbcs_fast'][...]])
        nV = np.multiply(fpi_dis_cdf['mms1_dis_numberdensity_fast'], V)
        smoothed_data = [Vz[0]]
        for i, value in enumerate(nV[1:]):
            smoothed_data.append((smoothed_data[i-1]*(2**M-1)+value)/2**M)
        diff = np.subtract(nV, smoothed_data)
        fpi_dis_df[f'{spacecraft}_dis_nV_Q'] = np.absolute(diff)

    except KeyError as e:
        print(f"{e} | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f"For {fpi_dis_cdf_path}. Skipping file. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        return pd.DataFrame()

    return fpi_dis_df


def edp_cdf_to_dataframe(edp_cdf_path, spacecraft):
    """ Converts EDP CDF to a Pandas dataframe.

    Convert a CDF containing EDP data at the provided path to a Pandas dataframe. Only copies measurements from
    the edp_var_list variable below.

    Args:
        edp_cdf_path: A path-like object pointing to a CDF file.
        spacecraft: Spacecraft string. One of [ "mms1", "mms2", "mms3", "mms4" ]

    Returns:
        A Pandas dataframe containing data from the CDF indexed by the CDF's Epoch.
    """

    try:
        edp_cdf = pycdf.CDF(str(edp_cdf_path))
    except pycdf.CDFError as e:
        print(f"{e}  | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f"File path: {edp_cdf_path} | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        sys.exit(3)
    # Parse edp
    try:
        edp_df = pd.DataFrame()
        edp_df['Epoch'] = edp_cdf[f'{spacecraft}_edp_dce_epoch'][...]
        edp_df.set_index('Epoch', inplace=True)

        edp_df[f'{spacecraft}_edp_x'] = edp_cdf[f'{spacecraft}_edp_dce_xyz_dsl'][:, 0]
        edp_df[f'{spacecraft}_edp_y'] = edp_cdf[f'{spacecraft}_edp_dce_xyz_dsl'][:, 1]
        edp_df[f'{spacecraft}_edp_z'] = edp_cdf[f'{spacecraft}_edp_dce_xyz_dsl'][:, 2]

        # Compute metafeatures
        edp_df[f'{spacecraft}_edp_|E|'] = np.sqrt(
            edp_df[f'{spacecraft}_edp_x'] ** 2 + edp_df[f'{spacecraft}_edp_y'] ** 2 + edp_df[
                f'{spacecraft}_edp_z'] ** 2)

    except KeyError as e:
        print(f"{e} | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f"For {edp_cdf_path}. Skipping file. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        return pd.DataFrame()

    return edp_df


def concatenate_all_cdf(start_date, end_date, base_directory_path, spacecraft, username, password):
    """ Merge all CDFs into a single dataframe.

    1. Converts each CDF in PROJECT_ROOT/mms_api_downloads to a Pandas dataframe
    2. Merges each dataframe for each instrument into a single, larger dataframe
    3. Downsamples each instruments' dataframe to match indices
    4. Merges each instruments' dataframe into a single dataframe

    Args:
        start_date: The first date for the date range, a datimetime object in the format %Y-%m-%d-%H:%M:%S.
        end_date: The last date for the date range, a datimetime object in the format %Y-%m-%d-%H:%M:%S.
        base_directory_path: A Path object to the parent .../mms1 directory
        spacecraft: Spacecraft string. One of [ "mms1", "mms2", "mms3", "mms4" ]

    Returns:
        A single merged dataframe with MMS data in the given date range.
    """

    fpi_des_df = merge_fpi_des_dataframes(start_date, end_date, base_directory_path, spacecraft, username, password)
    fpi_dis_df = merge_fpi_dis_dataframes(start_date, end_date, base_directory_path, spacecraft, username, password)
    afg_df = merge_afg_dataframes(start_date, end_date, base_directory_path, spacecraft, username, password)
    edp_df = merge_edp_dataframes(start_date, end_date, base_directory_path, spacecraft, username, password)

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
    if len(fpi_dis_df) != len(fpi_des_df):
        fpi_dis_df = fpi_dis_df.reindex(fpi_des_df.index, method='nearest')

    # Merge dataframes
    merged_df = fpi_des_df
    merged_df = merged_df.join(fpi_dis_df, how='outer')
    merged_df = merged_df.join(afg_df, how='outer')
    merged_df = merged_df.join(edp_df, how='outer')

    # Computer other metaproperties
    merged_df[f'{spacecraft}_temp_ratio'] = fpi_dis_df[f'{spacecraft}_dis_scalar_temperature'] / fpi_dis_df[
        f'{spacecraft}_dis_scalar_temperature']
    merged_df[f'{spacecraft}_plasma_beta'] = (fpi_dis_df[f'{spacecraft}_dis_scalar_temperature'] + fpi_dis_df[
        f'{spacecraft}_dis_scalar_temperature']) / edp_df[f'{spacecraft}_edp_|E|']

    return merged_df


def merge_edp_dataframes(start_date, end_date, base_directory_path, spacecraft, username, password):
    """ Merge EDP CDFs for the given date range into a single dataframe

     Args:
        start_date: The first date for the date range, a string given in the format "%Y-%m-%d"
        end_date: The last date for the date range, a string given in the format "%Y-%m-%d"
        base_directory_path: A Path object to the parent .../[spacecraft] directory
        spacecraft: Spacecraft string. One of [ "mms1", "mms2", "mms3", "mms4" ]

    Returns:
        A single merged dataframe with EDP instrument data in the given date range.
    """
    # Query SDC web service for list of CDFs

    edp_cdf_list = query_sdc(base_directory_path, start_date, end_date, spacecraft, "edp", "ql", "fast",
                             username, password, "dce")

    if len(edp_cdf_list) == 0:
        print(f"Error: No CDFs found for EDP in given range. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f'For date range: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} | {datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}')
        sys.exit(3)

    print(f"   Merging EDP dataframes. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")

    edp_df = None
    for file_path in edp_cdf_list:
        if edp_df is None:
            edp_df = edp_cdf_to_dataframe(file_path, spacecraft)
        else:
            edp_df = edp_df.append(edp_cdf_to_dataframe(file_path, spacecraft))

    if len(edp_df) == 0:
        print(f"Error: No valid EDP CDFs could be read. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f'For date range: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} | {datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}')
        sys.exit(3)

    return edp_df


def merge_fpi_des_dataframes(start_date, end_date, base_directory_path, spacecraft, username, password):
    """ Merge FPI DES CDFs for the given date range into a single dataframe

     Args:
        start_date: The first date for the date range, a string given in the format "%Y-%m-%d"
        end_date: The last date for the date range, a string given in the format "%Y-%m-%d"
        base_directory_path: A Path object to the parent .../mms1 directory
        spacecraft: Spacecraft string. One of [ "mms1", "mms2", "mms3", "mms4" ]

    Returns:
        A single merged dataframe with fpi_des instrument data in the given date range.
    """
    # Query SDC for FPI DES CDFs
    fpi_des_cdf_list = query_sdc(base_directory_path, start_date, end_date, spacecraft, "fpi", "ql", "fast",
                                 username, password, "des")

    if len(fpi_des_cdf_list) == 0:
        print(f"Error: No CDFs found for FPI DES in given range. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f'For date range: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%dT")} | {datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}')
        sys.exit(3)

    print(f"   Merging FPI DES dataframes. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")

    fpi_des_df = None
    for file_path in fpi_des_cdf_list:
        if fpi_des_df is None:
            fpi_des_df = fpi_des_cdf_to_dataframe(file_path, spacecraft)
        else:
            fpi_des_df = fpi_des_df.append(fpi_des_cdf_to_dataframe(file_path, spacecraft))

    if len(fpi_des_df) == 0:
        print(f"Error: No valid FPI DES CDFs could be read. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f'For date range: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} | {datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}')
        sys.exit(3)

    return fpi_des_df


def merge_afg_dataframes(start_date, end_date, base_directory_path, spacecraft, username, password):
    """ Merge AFG CDFs for the given date range into a single dataframe

     Args:
        start_date: The first date for the date range, a string given in the format "%Y-%m-%d"
        end_date: The last date for the date range, a string given in the format "%Y-%m-%d"
        base_directory_path: A Path object to the parent .../mms1 directory
        spacecraft: Spacecraft string. One of [ "mms1", "mms2", "mms3", "mms4" ]

    Returns:
        A single merged dataframe with AFG instrument data in the given date range.
    """
    # Query SDC for AFG CDFs
    afg_cdf_list = query_sdc(base_directory_path, start_date, end_date, spacecraft, "afg", "ql", "srvy", username,
                             password)

    if len(afg_cdf_list) == 0:
        print(f"Error: No CDFs found for AFG in given range. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f'For date range: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} | {datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}')
        sys.exit(3)

    print(f"   Merging AFG dataframes. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")

    afg_df = None
    for file_path in afg_cdf_list:
        if afg_df is None:
            afg_df = afg_cdf_to_dataframe(file_path, spacecraft)
        else:
            afg_df = afg_df.append(afg_cdf_to_dataframe(file_path, spacecraft))

    if len(afg_df) == 0:
        print(f"Error: No valid AFG CDFs could be read. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f'For date range: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} | {datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}')
        sys.exit(3)

    return afg_df


def merge_fpi_dis_dataframes(start_date, end_date, base_directory_path, spacecraft, username, password):
    """ Merge FPI DIS CDFs for the given date range into a single dataframe

     Args:
        start_date: The first date for the date range, a string given in the format "%Y-%m-%d"
        end_date: The last date for the date range, a string given in the format "%Y-%m-%d"
        base_directory_path: A Path object to the parent .../mms1 directory
        spacecraft: Spacecraft string. One of [ "mms1", "mms2", "mms3", "mms4" ]

    Returns:
        A single merged dataframe with FPI DIS instrument data in the given date range.
    """
    # Query SDC for FPI DIS CDFs
    fpi_dis_cdf_list = query_sdc(base_directory_path, start_date, end_date, spacecraft, "fpi", "ql", "fast",
                                 username, password, "dis")

    if len(fpi_dis_cdf_list) == 0:
        print(f"Error: No CDFs found for FPI DIS in given range. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f'For date range: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} | {datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}')
        sys.exit(3)

    print(f"   Merging FPI DIS dataframes. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")

    fpi_dis_df = None
    for file_path in fpi_dis_cdf_list:
        if fpi_dis_df is None:
            fpi_dis_df = fpi_dis_cdf_to_dataframe(file_path, spacecraft)
        else:
            fpi_dis_df = fpi_dis_df.append(fpi_dis_cdf_to_dataframe(file_path, spacecraft))

    if len(fpi_dis_df) == 0:
        print(f"Error: No valid FPI DES CDFs could be read. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f'For date range: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} | {datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}')
        sys.exit(3)

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


def lstm(num_features=123, layer_size=300):
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

def roundTime(dt=None, date_delta=datetime.timedelta(minutes=1), to='average'):
    """
    Round a datetime object to a multiple of a timedelta
    dt : datetime.datetime object, default now.
    dateDelta : timedelta object, we round to a multiple of this, default 1 minute.
    from:  http://stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object-python
    """
    round_to = date_delta.total_seconds()
    if dt is None:
        dt = datetime.now()
    seconds = (dt - dt.replace(hour=0, minute=0, second=0)).seconds

    if seconds % round_to == 0:
        rounding = (seconds + round_to / 2) // round_to * round_to
    else:
        if to == 'up':
            # // is a floor division, not a comment on following line (like in javascript):
            rounding = (seconds + round_to) // round_to * round_to
        elif to == 'down':
            rounding = seconds // round_to * round_to
        else:
            rounding = (seconds + round_to / 2) // round_to * round_to

    return dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)

def process(start_date, end_date, base_directory_path, spacecraft, username, password, test=False):
    # # Define MMS CDF directory location
    # Load model
    print(f"Loading model. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    model = lstm()
    model.load_weights('model/model_weights.h5')

    # Load data
    print(f"Loading data: | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    data = concatenate_all_cdf(start_date, end_date, base_directory_path, spacecraft, username, password)

    # Interpolate interior values, drop outside rows containing 0s
    print(f"Interpolating NaNs. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.interpolate(method='time', limit_area='inside')
    data = data.loc[(data != 0).any(axis=1)]

    # Select data within time range
    data = data.loc[start_date:end_date]
    data_index = data.index

    # Scale data
    print(f"Scaling data. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    scaler = pickle.load(open('model/scaler.sav', 'rb'))
    data = scaler.transform(data)

    # Run data through model
    print(f"Generating selection predictions. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    predictions_list = model.predict(np.expand_dims(data, axis=0))

    # Filter predictions with threshold
    threshold = 0.5
    filtered_output = [0 if x < threshold else 1 for x in predictions_list.squeeze()]

    # Return predictions if testing
    if test:
        return filtered_output

    # Create selections from predictions
    predictions_df = pd.DataFrame()
    predictions_df.insert(0, "time", data_index)
    predictions_df.insert(1, "prediction", filtered_output)
    predictions_df['group'] = (predictions_df.prediction != predictions_df.prediction.shift()).cumsum()
    predictions_df = predictions_df.loc[predictions_df['prediction'] == 1]
    selections = pd.DataFrame({'BeginDate': predictions_df.groupby('group').time.first().map(lambda x: roundTime(utc2tai(x), datetime.timedelta(seconds=10))),
                               'EndDate': predictions_df.groupby('group').time.last().map(lambda x: roundTime(utc2tai(x), datetime.timedelta(seconds=10)))})
    selections = selections.set_index('BeginDate')
    selections['score'] = "150.0" # This is a placeholder for the FOM
    selections['description'] = "MP crossing (automatically generated)"

    current_datetime = datetime.datetime.now()
    selections_filetime = current_datetime.strftime('%Y-%m-%d-%H-%M-%S')
    manifest_filetime = current_datetime.strftime('%Y%m%d%H%M%S')
    file_name = f'gls_selections_mp-dl-unh_{selections_filetime}.csv'

    # Output selections
    print(f"Saving selections to CSV: {file_name} | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")

    if sys.platform == 'darwin':  # Processor is run locally on Colin Small's laptop
        file_path = f''
        selections.to_csv(
            file_path+file_name, header=False)
    else:  # Assume the processor is being run at the SDC
        file_path = f'~/dropbox/'
        selections.to_csv(
            file_path+file_name, header=False)

    absolute_file_path = Path(file_path+file_name).expanduser().absolute()
    md5_hash = get_md5(absolute_file_path)
    manifest_file_name = f'mp-dl-unh_sdc_delivery_{manifest_filetime}.txt'

    # Create manifest file
    with open(Path(file_path+manifest_file_name).expanduser().absolute(), 'w') as manifest_file:
        manifest_file.write(f'{md5_hash}  {absolute_file_path}')
    manifest_file.close()

def get_md5(file_path):
    """
    Confirms file_path is a valid file and then computes and returns the md5 hash of the file.
    Author: Kim Kokkonen
    """
    assert os.path.isfile(file_path), '%s is not a file' % file_path

    # handles a file of any size
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def test(test_output):
    """
    Test the model

    mms_with_selections_2017-02-01 00:00:00_to_2017-02-28 23:59:59.csv
    """
    test_y = pickle.load(open('test/test_y.sav', 'rb'))
    return f1_score(test_y.astype(int), test_output)

def main():

    if sys.platform == 'darwin':  # Processor is run locally on Colin Small's laptop
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Error workaround for running on Mac OS
        base_directory_path = Path('/Users/colinrsmall/Documents/GitHub/sitl-downloader/mms_api_downloads')
    else:  # Assume processor is being run at SDC
        base_directory_path = Path('/')

    if len(sys.argv) == 1 or sys.argv[1] in ['-h', '-help', '--h', '--help']:
        print("Usage: processor.py start_date end_date spacecraft SDC_username SDC_password")
        print("or")
        print("Usage: processoy.py test SDC_username SDC_password")
        sys.exit(166)

    config = configparser.ConfigParser()
    config.read("config.ini")
    username = config["LOGIN CREDENTIALS"]["username"]
    password = config["LOGIN CREDENTIALS"]["password"]

    if sys.argv[1] == "test":
        start_date = datetime.datetime.strptime("2017-02-01T00:00:00", "%Y-%m-%dT%H:%M:%S")
        end_date = datetime.datetime.strptime("2017-02-10T23:59:59", "%Y-%m-%dT%H:%M:%S")
        spacecraft = "mms1"
        username = str(sys.argv[2])
        password = str(sys.argv[3])
        test_output = process(start_date, end_date, base_directory_path, spacecraft, username, password, test=True)
        print(f"F1 score: {test(test_output)}")
        sys.exit(0)

    try:
        start_date = datetime.datetime.strptime(str(sys.argv[1]), "%Y-%m-%dT%H:%M:%S")
        end_date = datetime.datetime.strptime(str(sys.argv[2]), "%Y-%m-%dT%H:%M:%S")
        spacecraft = str(sys.argv[3])

    except ValueError as e:
        print("Error: Input datetime not in correct format.")
        print(f"{e}")
        sys.exit(166)
    except IndexError:
        print(f"Not enough command line arguments. Expected 4, got {len(sys.argv)}")
        print("Usage: processor.py start_date end_date spacecraft")
        sys.exit(166)

    # Error handling
    if len(sys.argv) > 4:
        print(f"Soft Error: Too many command line arguments entered. Expected 4, got {len(sys.argv)}.")
        print("Usage: processor.py start_date end_date spacecraft")
        print("Continuing.")

    if spacecraft not in ["mms1", "mms2", "mms3", "mms4"]:
        print("Error: Invalid spacecraft entered.")
        print(f'Expected one of [ "mms1", "mms2", "mms3", "mms4" ], got {spacecraft}.')

    process(start_date, end_date, base_directory_path, spacecraft, username, password)

    print(f"Done | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    sys.exit(0)

main()
