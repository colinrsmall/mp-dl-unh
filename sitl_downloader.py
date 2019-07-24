""" Helper file for downloading SITL selections."""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from spacepy import pycdf
import re
import os
import scipy.constants
import sys
import datetime
import pickle
import gc
from tqdm import tqdm


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

    request = requests.get(request_url, timeout=10, auth=(username, password))
    request.raise_for_status()

    paths_string_list = request.text.split(",")

    paths_list = [Path(base_directory_path / Path(path)) for path in paths_string_list]

    return paths_list


def download_all_selections():
    """
    Download all SITL selection files from the SDC.
    """

    print("Downloading SITL selection files from SDC:")

    files_url = 'http://mmssitl.sr.unh.edu/sitl/'
    files_request = requests.get(files_url)

    # Check request status, quit if bad
    if files_request.status_code != requests.codes.ok:
        print("Unable to connect to SDC: code {}".format(files_request.status_code))
        return

    files_html = BeautifulSoup(files_request.content, features='lxml')
    files = [link['href'] for link in files_html.findAll('a', href=re.compile('sitl_\d+.csv'))]

    folder_path = Path(BASE_DIR / 'downloads')
    folder_path.mkdir(parents=True, exist_ok=True)

    for file_url in files:
        file_name = file_url.split('/')[-1]
        file_local_path = Path(BASE_DIR / 'downloads' / file_name)

        if not file_local_path.is_file():
            file = requests.get(files_url + file_url)
            with file_local_path.open('w') as f:
                f.write(file.text)
                file.close()


def concatenate_selection_files(files_path=Path(BASE_DIR / 'downloads')):
    """
    Concatenate all downloaded SITL selections into one CSV.

    Args:
        files_path: Path to the download location of SITL selections

    Returns:
        DataFrame: Concatenated list of all selections
    """

    print("Concatenating selections.")

    files = [x for x in files_path.iterdir() if x.is_file()]

    return pd.concat((pd.read_csv(f, infer_datetime_format=True, parse_dates=[0, 1], header=0) for f in files),
                     ignore_index=True, sort=False).sort_values(by='start_t')


def merge_selections(mms_data_path=BASE_DIR / 'mms_data' / 'mms_data.csv',
                     selections_path=BASE_DIR / 'export' / 'all_selections.csv',
                     mms_dataframe=None):
    """
    Merges the MMS data and the SITL selections into a single DataFrame.

    Args:
        mms_data_path: File path to MMS data csv
        selections_path: File path to SITL selections csv
        mms_dataframe: Optional argument to pass in a dataframe instead of using a saved .csv.

    Returns:
        Returns a DataFrame equivalent to the MMS data with an additional column denoting whether or not an observation
        was selected by the SITL.
    """

    print("Merging selections.")

    if mms_dataframe is None:
        mms_data = pd.read_csv(mms_data_path, infer_datetime_format=True, parse_dates=[0], index_col=0)
    else:
        mms_data = mms_dataframe

    selections = pd.read_csv(selections_path, infer_datetime_format=True, parse_dates=[0, 1])
    selections.dropna()
    selections = selections[selections['comments'].str.contains("MP", na=False)]

    # Create column to denote whether an observation is selected by SITLs
    mms_data['selected'] = False

    # Set selected to be True if the observation is in a date range of a selection
    date_col = mms_data.index
    cond_series = mms_data['selected']
    for start, end in zip(selections['start_t'], selections['end_t']):
        cond_series |= (start <= date_col) & (date_col <= end)
    mms_data.loc[cond_series, 'selected'] = True

    return mms_data

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
    afg_cdf = pycdf.CDF(str(afg_cdf_path))

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
        print(f"\n{e}")
        print(f"For {afg_cdf_path}. Skipping file.\n")
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
    fpi_des_cdf = pycdf.CDF(str(fpi_des_cdf_path))

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
        print(f"\n{e}")
        print(f"For {fpi_des_cdf_path}. Skipping file.\n")
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
    fpi_dis_cdf = pycdf.CDF(str(fpi_dis_cdf_path))

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
        print(f"\n{e}")
        print(f"For {fpi_dis_cdf_path}. Skipping file.\n")
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

    edp_cdf = pycdf.CDF(str(edp_cdf_path))

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
        print(f"\n{e}")
        print(f"For {edp_cdf_path}. Skipping file.\n")
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
        print("Error: No CDFs found for EDP in given range.")
        print(f'For date range: {start_date.strftime("%Y-%m-%dT%H:%M:%S")} to {end_date.strftime("%Y-%m-%dT%H:%M:%S")}')
        sys.exit(-1)

    print("\n   Merging EDP dataframes.")

    edp_df = None
    for file_path in tqdm(edp_cdf_list):
        if edp_df is None:
            edp_df = edp_cdf_to_dataframe(file_path, spacecraft)
        else:
            edp_df = edp_df.append(edp_cdf_to_dataframe(file_path, spacecraft))

    if len(edp_df) == 0:
        print("Error: No valid EDP CDFs could be read.")
        print(f'For date range: {start_date.strftime("%Y-%m-%dT%H:%M:%S")} to {end_date.strftime("%Y-%m-%dT%H:%M:%S")}')
        sys.exit(-1)

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
        print("Error: No CDFs found for FPI DES in given range.")
        print(f'For date range: {start_date.strftime("%Y-%m-%dT%H:%M:%S")} to {end_date.strftime("%Y-%m-%dT%H:%M:%S")}')
        sys.exit(-1)

    print("\n   Merging FPI DES dataframes.")

    fpi_des_df = None
    for file_path in tqdm(fpi_des_cdf_list):
        if fpi_des_df is None:
            fpi_des_df = fpi_des_cdf_to_dataframe(file_path, spacecraft)
        else:
            fpi_des_df = fpi_des_df.append(fpi_des_cdf_to_dataframe(file_path, spacecraft))

    if len(fpi_des_df) == 0:
        print("Error: No valid FPI DES CDFs could be read.")
        print(f'For date range: {start_date.strftime("%Y-%m-%dT%H:%M:%S")} to {end_date.strftime("%Y-%m-%dT%H:%M:%S")}')
        sys.exit(-1)

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
        print("Error: No CDFs found for AFG in given range.")
        print(f'For date range: {start_date.strftime("%Y-%m-%dT%H:%M:%S")} to {end_date.strftime("%Y-%m-%dT%H:%M:%S")}')
        sys.exit(-1)

    print("\n   Merging AFG dataframes.")

    afg_df = None
    for file_path in tqdm(afg_cdf_list):
        if afg_df is None:
            afg_df = afg_cdf_to_dataframe(file_path, spacecraft)
        else:
            afg_df = afg_df.append(afg_cdf_to_dataframe(file_path, spacecraft))

    if len(afg_df) == 0:
        print("Error: No valid AFG CDFs could be read.")
        print(f'For date range: {start_date.strftime("%Y-%m-%dT%H:%M:%S")} to {end_date.strftime("%Y-%m-%dT%H:%M:%S")}')
        sys.exit(-1)

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
        print("Error: No CDFs found for FPI DIS in given range.")
        print(f'For date range: {start_date.strftime("%Y-%m-%dT%H:%M:%S")} to {end_date.strftime("%Y-%m-%dT%H:%M:%S")}')
        sys.exit(-1)

    print("\n   Merging FPI DIS dataframes.")

    fpi_dis_df = None
    for file_path in tqdm(fpi_dis_cdf_list):
        if fpi_dis_df is None:
            fpi_dis_df = fpi_dis_cdf_to_dataframe(file_path, spacecraft)
        else:
            fpi_dis_df = fpi_dis_df.append(fpi_dis_cdf_to_dataframe(file_path, spacecraft))

    if len(fpi_dis_df) == 0:
        print("Error: No valid FPI DES CDFs could be read.")
        print(f'For date range: {start_date.strftime("%Y-%m-%dT%H:%M:%S")} to {end_date.strftime("%Y-%m-%dT%H:%M:%S")}')
        sys.exit(-1)

    return fpi_dis_df


def download(BASE_DIR, start_date, end_date, spacecraft):
    """
    Used only for mp-dl-unh
    """
    download_all_selections()
    df = concatenate_all_cdf(start_date, end_date, BASE_DIR/'mms_api_downloads', spacecraft=spacecraft,
                             username=sys.argv[4], password=sys.argv[5])
    df = merge_selections(mms_dataframe=df)
    return df

