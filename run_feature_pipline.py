import config
import logging

from zenml import pipeline
from steps.ingest import ingest_data
from steps.clean import clean_data
from steps.encode import encode_features
from steps.spliting import Spliting
from steps.featureEngineering import FeatureEngineering
from steps.best_features import SelectBestFeatures



@pipeline(enable_cache=True)
def run_pipeline():
    """_summary_
        Author: Sheikh Rasel Ahmed <shekhrasel59@gmail.com>
    """ 
    try:
        logging.info(f'==> Processing run_pipeline()')
        data = ingest_data(DATA_SOURCE=config.DATA_SOURCE)
        data = clean_data(data)
        data = encode_features(data)
        data = Spliting(data)
        data = FeatureEngineering(data)
        data = SelectBestFeatures(data)
        logging.info(f'==> Successfully processed run_pipeline()')
    except Exception as e:
        logging.error(logging.error(f'==> Error in run_pipeline(): {e}'))
        
        
        
if __name__ == "__main__":
    run = run_pipeline()