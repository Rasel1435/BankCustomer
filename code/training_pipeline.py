import pandas as pd
import mlflow.sklearn
import mlflow
import warnings
import logging

from scipy.stats import randint
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
# from sklearn.metrics importsilhouette_score, calinski_harabasz_score, davies_bouldin_score

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%b(%m)-%Y %I:%M:%S',
)
logger = logging.getLogger(__name__)