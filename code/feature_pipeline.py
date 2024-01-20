import pandas as pd
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from feature_engine.creation import RelativeFeatures
from sklearn.linear_model import LogisticRegression
from feature_engine.selection import SmartCorrelatedSelection, RecursiveFeatureElimination



### Logging Configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt= '%d-%b(%m)-%Y %I:%M:%S',
)
logger = logging.getLogger(__name__)



# Lodaing Data 
def get_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.error(f'in get_data(): {e}')

# cleaning Data
def clean_data() -> None:
    global df
    try:
        df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
    
        ### optimize for memory
        new_types = {
            'age': 'int32', 'balance':'int32', 
            'yearly_income':'float32', 'number_of_children':'int32',
            'duration': 'int32', 'day':'int32', 'campaign':'int32', 'pdays':'int32',
            'previous': 'int32'
        }
        for col, typp in new_types.items():
            df[col] = df[col].astype(typp)
    except Exception as e:
        logger.error(f'in clean_data(): {e}')
        

# LebelEncoding 
def encode_categorical_features() -> None:
    global df
    try:
        ### nominal encoding -
        nominal_cols = ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'poutcome', 'deposit']
        for col in nominal_cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
        
        ### to_int32
        df['job'] = df['job'].astype('int32')
        
        ### hash-based ordinal encoding
        hash_edu = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
        df['education'] = df['education'].map(hash_edu).astype('int32')
        
        ### drop month
        df.drop(columns=['month'], inplace=True) 
    except Exception as e:
        logger.error(f'in encode_data(): {e}')
       
        
# Spliting
def split_data() -> None:
    global df
    try:
        splitter = KFold(n_splits=5, shuffle=True, random_state=33)
        folds = list(splitter.split(df))
        
        # save folds in a dictionary with key as fold_<fold_number>_train and fold_<fold_number>_test
        d_folds = {}
        fold = 0
        for train_index, test_index in folds:
            d_folds[f'fold_{fold}_train'] = train_index
            d_folds[f'fold_{fold}_test'] = test_index
            fold += 1
    except Exception as e:
        logger.error(f'in split_data(): {e}')


# featureEngineering 
def feature_engineering() -> None:
    global df
    try:
        df['number_of_children'] = df['number_of_children'] + 1
        relativeFeatures = RelativeFeatures(
            variables=['yearly_income', 'balance'], reference=['number_of_children'], func=['div'], 
            fill_value=None, missing_values='ignore', drop_original=False
        )
        features = relativeFeatures.fit_transform(df)
        df['number_of_children'] = df['number_of_children'] - 1
        
        ## add new features
        for feature in list(features.columns)[18:]:
            df[feature] = features[feature]
        
        
        ## add new feature - yearly_income * age
        df['income_mul_age'] = df['yearly_income'] * df['age']
        return True
    except Exception as e:
        logger.error(f'in featureEngineering(): {e}')
        

# featureSelection
def select_best_features() -> None:
    global df
    try:
        X = df.drop(columns=['deposit'])
        y = df['deposit']
        
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
        
        return rfe_features.union(scs_features) 
    except Exception as e:
        logger.error(f'in select_best_features(): {e}')


# Now Time to call the all function and save it 
def preprocessFeatures():
    global df
    try:
        clean_data()
        encode_categorical_features()
        split_data()
        feature_engineering()
        select_best_features()
    except Exception as e:
        logger.error(f'in preprocessFeatures(): {e}')
    

if __name__ == '__main__':
    df = get_data('C:/Users/SRA/Desktop/backup/TowfiqVai/BankCustomer/data/bank.csv')
    
    preprocessFeatures()
    if df is not None:
        # Save the processed data
        output_file_path = r"C:/Users/SRA/Desktop/backup/TowfiqVai/BankCustomer/data/final.csv"
        df.to_csv(output_file_path, index=False)
        print(df.head())
        logger.info(f"data has been saved successfully!")
    else:
        logger.error(f"data is not available!")
        
# Author: Sheikh Rasel Ahmed <shekhrasel59@gmail.com>