""" Helper file for downloading SITL selections."""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import re
import os

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
                     selections_path=BASE_DIR / 'all_selections' / 'all_selections.csv'):
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

    # Create column to denote whether an observation is selected by SITLs
    mms_data['selected'] = False

    # Set selected to be True if the observation is in a date range of a selection
    date_col = mms_data.index
    cond_series = mms_data['selected']
    for start, end in zip(selections['start_t'], selections['end_t']):
        cond_series |= (start <= date_col) & (date_col <= end)
    mms_data.loc[cond_series, 'selected'] = True

    return mms_data


# download_all()
df = concatenate_files()
df.to_csv(Path(BASE_DIR / 'export' / 'csv' / 'all_selections.csv'), index=False, columns=df.columns[0:5])
#merge_selections().to_csv(BASE_DIR / 'export' / 'csv' / 'mms_with_selections.csv', index_label='Time')
print("Done")
