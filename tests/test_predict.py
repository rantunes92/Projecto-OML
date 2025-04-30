import os
from src.predict import carregar_modelo, fazer_previsoes, obter_accuracy

def test_model_random_forest_predictions():
    uri = "http://127.0.0.1:5000"
    model_name = "Random_Forest"
    model_version = "11"
    data_path = "data/rumos_bank_test.csv"
    scaler_path = "scaler.pkl"
    nsamples = 10

    # Carrega modelo
    model = carregar_modelo(uri, model_name, model_version)

    # Faz previsões
    y_preds, n = fazer_previsoes(model, data_path, scaler_path, nsamples=nsamples)

    # Verificações básicas
    assert len(y_preds) == nsamples
    assert all([p in [0, 1] for p in y_preds])

    # Valida previsão em relação à accuracy registrada
    accuracy = obter_accuracy(uri, model_name, model_version)
    soma_y_preds = sum(y_preds)

    assert soma_y_preds <= nsamples, "Erro: Previsões positivas excedem o total"
    margem = 0.5 * nsamples
    assert nsamples * accuracy - margem <= soma_y_preds <= nsamples * accuracy + margem, \
        f"Erro: Soma dos y_preds ({soma_y_preds}) fora do intervalo baseado na accuracy ({accuracy})"
