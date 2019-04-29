""" Helper file for downloading SITL selections."""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from pymms import mrmms_sdc_api as mms_api
import re
import os
import pycdf_utils

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def download_all():
    """
    Download all SITL selection files from the SDC.
    """

    files_url = 'http://mmssitl.sr.unh.edu/sitl/'
    files_request = requests.get(files_url)

    # Check request status, quit if bad
    if files_request.status_code != requests.codes.ok:
        print("Unable to connect to SDC: code {}".format(files_request.status_code))
        return

    files_html = BeautifulSoup(files_request.content, features='lxml')
    files = [link['href'] for link in files_html.findAll('a', href=re.compile('sitl_\d+.csv'))]

    for file_url in files:
        file_name = file_url.split('/')[-1]
        file_local_path = Path(BASE_DIR / 'downloads' / file_name)

        if not file_local_path.is_file():
            file = requests.get(files_url + file_url)
            with file_local_path.open('w') as f:
                f.write(file.text)
                file.close()


def concatenate_files(files_path=Path(BASE_DIR / 'downloads')):
    """
    Concatenate all downloaded SITl selections into one CSV.

    Args:
        files_path: Path to the download location of SITL selections

    Returns:
        DataFrame: Concatenated list of all selections
    """

    files = [x for x in files_path.iterdir() if x.is_file()]

    return pd.concat((pd.read_csv(f, infer_datetime_format=True, parse_dates=[0, 1], header=0) for f in files),
                     ignore_index=True, sort=False).sort_values(by='start_t')


def merge_selections(mms_data_path=BASE_DIR / 'mms_data' / 'data.csv',
                     selections_path=BASE_DIR / 'mms_data' / 'all_selections.csv'):
    """
    Merges the MMS data and the SITL selections into a single DataFrame.

    Args:
        mms_data_path: This is the first param.
        selections_path: This is a second param.

    Returns:
        Returns a DataFrame equivalent to the MMS data with an additional column denoting whether or not an observation
        was selected by the SITL.
    """

    mms_data = pd.read_csv(mms_data_path, infer_datetime_format=True, parse_dates=[0], index_col=0)
    selections = pd.read_csv(selections_path, infer_datetime_format=True, parse_dates=[0, 1])
    selections.dropna()
    #selections = selections[selections['comments'].str.contains("MP")]

    # Create column to denote whether an observation is selected by SITLs
    mms_data['selected'] = False

    # Set selected to be True if the observation is in a date range of a selection
    date_col = mms_data.index
    cond_series = mms_data['selected']
    for start, end in zip(selections['start_t'], selections['end_t']):
        cond_series |= (start <= date_col) & (date_col <= end)
    mms_data.loc[cond_series, 'selected'] = True

    return mms_data


def download_cdf(start_date, end_date):
    """
    Download needed CDF files for the given times and convert to an dataframe. Uses MrMMS_SDC_API and MMS_Utils.

    Params:
        start_date (str):    Start date of data interval, formatted as either %Y-%m-%d or
                             %Y-%m-%dT%H:%M:%S.
        end_date (str):      End date of data interval, formatted as either %Y-%m-%d or
                             %Y-%m-%dT%H:%M:%S.
    """

    # Depend0 = name of the time variable in the CDF
    # 1. Read depend0
    # 2. Look back in CDF for the name of the variable
    # Fillval = says what the NaN is
    # ONly use DATA var_type
    # Catdesc = variable description

    # mms  # _afg_ql_srvy_
    # mms  # _fpi_sitl_fast_
    # mms  # _dsp_l1a_epsd|bpsd # Use neither or both, but use neither for now
    # mms  # _edp_sitl_fast_dce_
    # mms  # _hpca_l1b_srvy_(ion|moments)_ # Definitely use moments, not ions right now
    # mms  # _feeps_sitl_srvy_(ion|electron)_ # Use both
    # mms  # _epd-eis_l1b_srvy_extof_ # Don't use it for now

    download_dir = BASE_DIR/'mms_api_downloads'

    afg_gl_srvy_api = mms_api.MrMMS_SDC_API(sc='mms1', instr='afg', level='ql', mode='srvy', data_root=str(download_dir),
                                            start_date=start_date,
                                            end_date=end_date)
    print("afg downloaded")

    afg_gl_srvy_api.Download()

    fpi_sitl_fast_api = mms_api.MrMMS_SDC_API(sc='mms1', instr='fpi', level='ql', mode='fast', data_root=str(download_dir),
                                              optdesc='des',
                                              site='team',
                                              start_date=start_date,
                                              end_date=end_date)
    fpi_sitl_fast_api.Download()
    print("fpi des downloaded")

    fpi_sitl_fast_api = mms_api.MrMMS_SDC_API(sc='mms1', instr='fpi', level='ql', mode='fast', data_root=str(download_dir),
                                              optdesc='dis',
                                              start_date=start_date,
                                              end_date=end_date)
    fpi_sitl_fast_api.Download()
    print("fpi dis downloaded")

    fpi_sitl_fast_api = mms_api.MrMMS_SDC_API(sc='mms1', instr='edp', level='ql', mode='fast', data_root=str(download_dir),
                                              optdesc='dce',
                                              start_date=start_date,
                                              end_date=end_date)
    fpi_sitl_fast_api.Download()
    print("edp downloaded")


download_cdf('2017-01-01', '2017-01-31T23:59:59')

# download_all()
# df = concatenate_files()
# df.to_csv(Path(BASE_DIR / 'export' / 'csv' / 'all_selections.csv'), index=False, columns=df.columns[0:5])
# merge_selections().to_csv(BASE_DIR / 'export' / 'csv' / 'mms_with_selections.csv', index_label='Time')
# print("Done")
