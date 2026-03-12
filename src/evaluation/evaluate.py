import json
import logging
import os
import yaml
import numpy as np
import mlflow
import dagshub

from pathlib import Path
from dotenv import load_dotenv
from PIL import ImageFile

import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocessing.prepare import test_generator

ImageFile.LOAD_TRUNCATED_IMAGES = True
load_dotenv()

logger = logging.getLogger("src.evaluation.evaluate")


def load_params() -> dict:
    """
    Carrega as configurações globais do arquivo params.yaml.

    Returns:
        dict: Dicionário completo de parâmetros.
    """
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def save_confusion_matrix(cm: np.ndarray, class_names: list) -> None:
    """Gera e salva a matriz de confusão como PNG.

    Args:
        cm (np.ndarray): Matriz de confusão calculada pelo sklearn.
        class_names (list): Nomes das classes.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Matriz de confusão - CNN Face Shape")
    plt.ylabel("Real")
    plt.xlabel("Previsto")
    plt.tight_layout()

    Path("metrics").mkdir(parents=True, exist_ok=True)
    plt.savefig("metrics/confusion_matrix.png")
    plt.close()
    print("Matriz de confusão salva em: metrics/confusion_matrix.png")


def evaluate_model(params: dict) -> dict:
    """
    Carrega o melhor modelo salvo e avalia no conjunto de teste.
    Salva métricas em metrics/evaluation.json e loga no MLFlow/DagsHub.

    Args:
        params (dict): Parâmetros do params.yaml.

    Returns:
        dict: Métricas finais de avaliação.
    """

    model_path = "models/best_model.keras"
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}. Execute o treino primeiro."
        )

    print(f"Carregando modelo: {model_path}")
    model = keras.models.load_model(model_path)

    gen_teste = test_generator()
    class_names = list(gen_teste.class_indices.keys())

    # Avaliação
    print("Avaliando o conjunto de teste...")
    test_loss, test_accuracy = model.evaluate(gen_teste, verbose=0)

    # Predições
    y_pred_proba = model.predict(gen_teste, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = gen_teste.classes

    report = classification_report(
        y_true=y_true, y_pred=y_pred, target_names=class_names, output_dict=True
    )

    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=class_names))

    # Matriz de confusão
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    save_confusion_matrix(cm=cm, class_names=class_names)

    # Estrutura de métricas
    evaluation_metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "per_class": {
            name: {
                "precision": round(report[name]["precision"], 4),
                "recall": round(report[name]["recall"], 4),
                "f1-score": round(report[name]["f1-score"], 4),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
        "macro_avg": {
            "precision": round(report["macro avg"]["precision"], 4),
            "recall": round(report["macro avg"]["recall"], 4),
            "f1-score": round(report["macro avg"]["f1-score"], 4),
        },
    }

    Path("metrics").mkdir(parents=True, exist_ok=True)
    with open("metrics/evaluation.json", "w") as f:
        json.dump(evaluation_metrics, f, indent=4)
    print("Métricas salvas em metrics/evaluation.json")

    return evaluation_metrics


def setup_mlflow(params: dict) -> None:
    """Configura DagsHub + MlFlow e loga métricas e artefatos da avaliação.

    Args:
        params (dict): Parâmetros do params.yaml.
    """
    dagshub.init(
        repo_owner=os.getenv("MLFLOW_TRACKING_USERNAME"), repo_name="cnn-face-shape"
    )
    mlflow.set_experiment(params["mlflow"]["experiment_name"])


def log_to_mlflow(params: dict, metrics: dict) -> None:
    """Loga métricas e artefatos da avaliação no MLFlow/DagsHub.

    Args:
        params (dict): Parâmetros do params.yaml.
        metrics (dict): Métricas retornadas por evaluate_model.
    """
    setup_mlflow(params=params)

    with mlflow.start_run(run_name=f"{params['mlflow']['run_name']}_evaluation"):
        mlflow.log_metric("test_loss", metrics["test_loss"])
        mlflow.log_metric("test_accuracy", metrics["test_accuracy"])

        for name, metric in metrics["per_class"].items():
            mlflow.log_metric(f"f1_{name}", metric["f1-score"])
            mlflow.log_metric(f"precision_{name}", metric["precision"])
            mlflow.log_metric(f"recall_{name}", metric["recall"])

        mlflow.log_artifact("metrics/evaluation.json")
        mlflow.log_artifact("metrics/confusion_matrix.png")

    print("Métricas logadas no MLFlow e DagsHub")


def main() -> None:
    """
    Ponto de entrada do step de avaliação.
    """
    print("=" * 60)
    print("AVALIAÇÃO DO MODELO")
    print("=" * 60)

    params = load_params()
    metrics = evaluate_model(params=params)
    log_to_mlflow(params=params, metrics=metrics)

    print("\n" + "=" * 60)
    print("AVALIAÇÃO CONCLUÍDA!!")
    print("=" * 60)

    print(f"Test Loss:{metrics['test_loss']:.4f}")
    print(f"Test Accuracy:{metrics['test_accuracy']:.4f}")
    print(f"\n Por classe:")
    for name, metric in metrics["per_class"].items():
        print(
            f"{name:<10} precision={metric['precision']:.2f} recall={metric['recall']:.2f} f1={metric['f1-score']:.2f}"
        )


if __name__ == "__main__":
    main()
