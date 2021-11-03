from src.dashboard import dashboard
import src.loading as loading
import src.constants as cst

#####  Set Logger  #####
from src.utils.loggers import MainLogger
logger = MainLogger.getLogger(__name__)

sample_df = loading.read_csv_from_path(cst.PREPROCESSED_TRAIN_FILE_PATH)
app = dashboard.DashboardApp(sample_df)
app.configure_page()
app.create_main_pages()