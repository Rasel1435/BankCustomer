import os
import config
import logging
import pandas as pd

from zenml import step
from typing import Union
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

logger = logging.getLogger(__name__)

@step(enable_cache=True)
def encode_features(data: pd.DataFrame, nominal_features: Union[list, None] = None, ordinal_features: Union[list, None] = None) -> pd.DataFrame:
    """
    Encode categorical features.
    """
    logger.info(f"==> Processing encode_features()")
    try:
        if nominal_features is None and ordinal_features is None:
            logger.info("No categorical features to encode.")
            return data
        encoders = {}
        for column in nominal_features:
            encoder = LabelEncoder().fit(data[column])
            data[column] = encoder.transform(data[column])
            encoders[column] = encoder
            logger.info(f"Encoded nominal feature: {column}")

        for column in ordinal_features:
            encoder = OrdinalEncoder(handle_unknown="error").fit(data[[column,]])
            data[column] = encoder.transform(data[[column,]])
            encoders[column] = encoder
            logger.info(f"Encoded Ordinal feature: {column}")
        logger.info("Categorical features encoded successfully.")
        dump(encoders, os.path.join(config.ENCODERS_PATH, 'encoders.joblib'))
        logger.info(f'==> Successfully processed encode_features()')
        return data
    except Exception as e:
        logger.error(f"Error encoding categorical features: {e}")
        raise None
