import logging
import pandas as pd

from zenml import step
from typing import Union

logger = logging.getLogger(__name__)


@step(enable_cache=True)
def ingest_data(DATA_SOURCE: str) -> Union[pd.DataFrame, None]:
    """
    Ingests data from a given path.

    Args:
        data_source: The path to the data.

    Returns:
        The data as a DataFrame.
    """
    try:
        logger.info(f"Reading data from {DATA_SOURCE}")
        data = pd.read_csv(
            DATA_SOURCE,  encoding="unicode_escape", low_memory=False)
        logger.info(f"Data read from {DATA_SOURCE}")
        return data
    except Exception as e:
        logger.error(f"Error reading data from {DATA_SOURCE}: {e}")
        return None

if __name__ == '__main__':
    data = ingest_data(
        DATA_SOURCE='C:/Users/SRA/Desktop/backup/TowfiqVai/BankCustomer/data/final.csv')
    print(data.head())
    print(data.shape)
    """Now this data come from local system but when you deploy your model that moment
    you have to add the realtime data source link here.
    """