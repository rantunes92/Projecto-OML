import mlflow
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from mlflow.tracking import MlflowClient

def carregar_modelo(uri, model_name, model_version):
    mlflow.set_tracking_uri(uri)
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
    return model

def fazer_previsoes(model, data_path, scaler_path, nsamples=10):
    df = pd.read_csv(data_path)
    input_data = df.loc[df['default.payment.next.month'] == 1].sample(nsamples)
    y_true = input_data["default.payment.next.month"]

    X = input_data.drop("default.payment.next.month", axis=1)
    colunas = X.columns
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=colunas)

    y_probs = model.predict_proba(X_scaled)[:, 1]
    y_preds = (y_probs > 0.3).astype(int)

    return y_preds, nsamples

def obter_accuracy(uri, model_name, model_version):
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    model_info = client.get_model_version(model_name, model_version)
    run_id = model_info.run_id
    metrics = client.get_run(run_id).data.metrics
    return metrics.get("accuracy", 0.0)
