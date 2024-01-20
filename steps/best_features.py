import logging
import pandas as pd

from zenml import step
from typing import Union
from sklearn.linear_model import LogisticRegression
from feature_engine.selection import SmartCorrelatedSelection, RecursiveFeatureElimination

logger = logging.getLogger(__name__)

@step(enable_cache=True)
def SelectBestFeatures(data: pd.DataFrame) -> Union[pd.DataFrame, None]:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        Author: Sheikh Rasel Ahmed <shekhrasel59@gmail.com>

    Returns:
        Union[pd.DataFrame, None]: _description_
    """
    logger.info(f'==> Processing The SelectBestFeatures()')
    try:
        X = data.drop(columns=['deposit'])
        y = data['deposit']
        
        scs = SmartCorrelatedSelection(
            variables=None, method='pearson', 
            threshold=0.4, missing_values='ignore', 
            selection_method='variance', 
            estimator=None, scoring='roc_auc', 
            cv=3, confirm_variables=False
        )
        scs_features = set(scs.fit_transform(X, y).columns)
        
        
        model = LogisticRegression(max_iter=1000)
        rfe = RecursiveFeatureElimination(
            model, scoring='roc_auc', cv=3, 
            threshold=0.01, variables=None, 
            confirm_variables=False
        )
        rfe_features = set(rfe.fit_transform(X, y).columns)
        
        rfe_features.union(scs_features)
        logger.info(f'Successfully Processed The SelectBestFeatures()')
        return data
    except Exception as e:
        logger.error(f'in the SelectBestFeatures()')
        return None