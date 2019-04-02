import os
from pathlib import Path
import numpy as np
import pandas as pd
import sqlite3
import sys
import datetime

"""Preprocess the training and test data"""

# 1. Open MMS data from sql
# 2. Grab relevant dates of selections
# 3. Vectorize selections to match timestamps of MMS data

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

print("Done!")
exit