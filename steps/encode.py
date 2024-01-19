import logging
from zenml import step
import pandas as pd
from typing import Union
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from joblib import dump, load
import os
import config


@step(enable_cache=True)
def encode_features(data: pd.DataFrame, nominal_features: Union[list, None] = None, ordinal_features: Union[list, None] = None) -> pd.DataFrame:
    """
    Encode categorical features.
    """
    logging.info("Encoding categorical features")
    try:
        if nominal_features is None and ordinal_features is None:
            logging.info("No categorical features to encode.")
            return data
        encoders = {}
        for column in nominal_features:
            encoder = LabelEncoder().fit(data[column])
            data[column] = encoder.transform(data[column])
            encoders[column] = encoder
            logging.info(f"Encoded nominal feature: {column}")

        for column in ordinal_features:
            encoder = OrdinalEncoder(handle_unknown="error").fit(data[[column,]])
            data[column] = encoder.transform(data[[column,]])
            encoders[column] = encoder
            logging.info(f"Encoded Ordinal feature: {column}")
        logging.info("Categorical features encoded successfully.")
        dump(encoders, os.path.join(config.ENCODERS_PATH, 'encoders.joblib'))
        logging.info("Encoders saved successfully.")
        return data
    except Exception as e:
        logging.error(f"Error encoding categorical features: {e}")
        raise e
