import logging
import pandas as pd

from zenml import step
from typing import Union

logger = logging.getLogger(__name__)

@step(enable_cache=True)
def clean_data(data: pd.DataFrame) -> Union[pd.DataFrame, None]:
    """
    Clean the data by removing null values, duplicate rows, and formating column names .
    """
    logger.info(f"==> Processing clean_data()")
    try:
        data.dropna(axis=0, inplace=True)
        data.drop_duplicates(inplace=True)
        data.drop(columns=['month'], inplace=True)
        if data.empty:
            logging.info("No data to clean.")
            return None
        # reformat columns
        data.columns = [col.lower().strip().replace(' ', '_')
                        for col in data.columns]

        # # optimize for memory
        # new_types = {
        #     'age': 'int32', 'balance': 'int32',
        #     'yearly_income': 'float32', 'number_of_children': 'int32',
        #     'duration': 'int32', 'day': 'int32', 'campaign': 'int32', 'pdays': 'int32',
        #     'previous': 'int32'
        # }

        # # TypeCasting to optimize for memory
        # for col, typp in new_types.items():
        #     data[col] = data[col].astype(typp)

        logger.info(f"==> Successfully processed clean_data()")
        return data
    except Exception as e:
        logger.error(f"Error while cleaning data: {e}")
        return None


# if __name__ == "__main__":
#     # Example usage
#     data = pd.read_csv("data/bank.csv")
#     cleaned_data = clean_data(data)

