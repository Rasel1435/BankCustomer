import logging
import pandas as pd

from zenml import step
from typing import Union
from feature_engine.creation import RelativeFeatures

logger = logging.getLogger(__name__)

@step(enable_cache=True)
def FeatureEngineering(data: pd.DataFrame) -> Union[pd.DataFrame, None]:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        Author: Sheikh Rasel Ahmed <shekhrasel59@gmail.com>

    Returns:
        Union[pd.DataFrame, None]: _description_
    """
    logger.info(f'==> Processing the FeatureEngineering()')
    try:
        data['number_of_children'] = data['number_of_children'] + 1
        relativeFeatures = RelativeFeatures(
            variables=['yearly_income', 'balance'], reference=['number_of_children'], func=['div'], 
            fill_value=None, missing_values='ignore', drop_original=False
        )
        features = relativeFeatures.fit_transform(data)
        data['number_of_children'] = data['number_of_children'] - 1
        
        ## add new features
        for feature in list(features.columns)[18:]:
            data[feature] = features[feature]
        
        
        ## add new feature - yearly_income * age
        data['income_mul_age'] = data['yearly_income'] * data['age']
        # return True
        logger.info(f'Successfully processed FeatureEngineering()')
        return data
    except Exception as e:
        logger.error(f'in the FeatureEngineering()')
        return None