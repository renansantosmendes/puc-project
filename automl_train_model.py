import os
import git
import mlflow
import logging
import argparse
import pandas as pd
from sklearn import preprocessing
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from tpot import TPOTClassifier

logging.debug(f'Start data processing')
parser = argparse.ArgumentParser(description='Pass data parameters to process')
parser.add_argument('--base_data_url', required=True)
parser.add_argument('--data_file_name', required=True)
parser.add_argument('--tracking_uri', required=True)
parser.add_argument('--tracking_username', required=True)
parser.add_argument('--tracking_password', required=True)
parser.add_argument('--registry_model_name', required=True)
parser.add_argument('--n_estimators', type=int, required=True)
args = parser.parse_args()

MLFLOW_TRACKING_URI = args.tracking_uri
MLFLOW_TRACKING_USERNAME = args.tracking_username
MLFLOW_TRACKING_PASSWORD = args.tracking_password
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
mlflow.sklearn.autolog(log_models=True,
                       log_input_examples=True,
                       log_model_signatures=True)

git.Repo.clone_from(args.base_data_url,
                    'data')

data = pd.read_csv(os.path.join(os.getcwd(),
                                'data',
                                args.data_file_name),
                   sep=',',
                   encoding='utf-8')
features_to_remove = data.columns[7:]
X = data.drop(features_to_remove, axis=1)
y = data["fetal_health"]

columns_names = list(X.columns)
scaler = preprocessing.StandardScaler()
X_df = scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=columns_names)

X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

cv = RepeatedStratifiedKFold(n_splits=2,
                             n_repeats=2,
                             random_state=42)

model = TPOTClassifier(generations=3,
                       population_size=5,
                       cv=cv,
                       scoring='accuracy',
                       verbosity=0,
                       random_state=42,
                       n_jobs=-1)

with mlflow.start_run(run_name='tpot_classifier') as run:
    model.fit(X, y)
    mlflow.sklearn.eval_and_log_metrics(model,
                                        X_test,
                                        y_test,
                                        prefix="test_")

artifact_path = "model"
model_name = args.registry_model_name
model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run.info.run_id,
                                                    artifact_path=artifact_path)
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
