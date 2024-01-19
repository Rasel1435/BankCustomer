import logging
from zenml import step
import pandas as pd
from typing import Union


@step(enable_cache=True)
def ingest_data(data_source: str) -> Union[pd.DataFrame, None]:
    """
    Ingests data from a given path.

    Args:
        data_source: The path to the data.

    Returns:
        The data as a DataFrame.
    """
    try:
        logging.info(f"Reading data from {data_source}")
        data = pd.read_csv(
            data_source,  encoding="unicode_escape", low_memory=False)
        logging.info(f"Data read from {data_source}")
        return data
    except Exception as e:
        logging.error(f"Error reading data from {data_source}: {e}")
        return None
