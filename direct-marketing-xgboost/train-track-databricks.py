import mlflow.xgboost
import xgboost as xgb
from load_dataset import load_dataset

if __name__ == '__main__':
    remote_server_uri = "databricks" # set to your server URI
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment('/Users/workshop@weclouddata.com/direct-marketing-experiment')
    with mlflow.start_run(run_name='direct-marketing-xgboost-basic') as run:
        x_train, x_test, y_train, y_test = load_dataset(
            'data/bank-additional-full.csv', ';'
        )
        cls = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc')
        cls.fit(x_train, y_train)
        auc = cls.score(x_test, y_test)
        print("AUC ", auc)
        mlflow.log_metric("auc", auc)

        mlflow.xgboost.log_model(cls, "direct-marketing-xgboost-model")
        mlflow.end_run()
