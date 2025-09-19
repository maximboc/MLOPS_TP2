import mlflow
from mlflow.models import infer_signature
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri(uri="http://mlflow-service:8080")
mlflow.set_experiment("MLflow Quickstart")

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def log_model(params):
    with mlflow.start_run():
        mlflow.log_params(params)
        lr = LogisticRegression(**params)
        lr.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, lr.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)
        signature = infer_signature(X_train, lr.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="iris_model"
        )
        print(f"✅ Logged model with solver={params['solver']}, accuracy={accuracy}")

params_v1 = {"solver": "lbfgs", "max_iter": 1000, "multi_class": "auto", "random_state": 8888}
log_model(params_v1)

params_v2 = {"solver": "liblinear", "max_iter": 500, "multi_class": "auto", "random_state": 8888}
log_model(params_v2)
