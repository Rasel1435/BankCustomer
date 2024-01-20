import logging
import pandas as pd

from zenml import step
from typing import Union
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

@step(enable_cache=True)
def Spliting(data: pd.DataFrame) -> Union[pd.DataFrame, None]:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        Author: Sheikh Rasel Ahmed <shekhrasel59@gmail.com>

    Returns:
        Union[pd.DataFrame, None]: _description_
    """
    logger.info(f'==> Processing Spliting()')
    try:
        splitter = KFold(n_splits=5, shuffle=True, random_state=33)
        folds = list(splitter.split(data))
        
        # save folds in a dictionary with key as fold_<fold_number>_train and fold_<fold_number>_test
        d_folds = {}
        fold = 0
        for train_index, test_index in folds:
            d_folds[f'fold_{fold}_train'] = train_index
            d_folds[f'fold_{fold}_test'] = test_index
            fold += 1
        logger.info(f"==> Successfully processed Spliting()")
        return data
    except Exception as e:
        logger.error(f'in Spliting(): {e}')
        return None
    
