import pandas as pd
import sys

from Loading import loading
from Dashboard import dashboard

import constants as cst

sample_df = loading.read_csv_from_path(cst.PREPROCESSED_TRAIN_FILE_PATH)
app = dashboard.DashboardApp(sample_df)
app.configure_page()
app.create_main_pages()

