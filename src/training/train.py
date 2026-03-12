import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List

import yaml
import mlflow
import mlflow.keras
import dagshub
from dotenv import load_dotenv

import keras
import pandas as pd

from src.preprocessing.prepare import train_generator, test_generator
from src.training.model_training import build_model

load_dotenv()
logger = logging.getLogger("src.training.train")


def load_params() -> Dict[str, Any]:
    """Carrega as configurações globais do arquivo params.yaml.

    Returns:
        Dict[str, Any]: Dicionário completo de parâmetros do projeto.
    """
    params_path = Path("params.yaml")
    if not params_path.exists():
        raise FileNotFoundError(f"Arquivo params.yaml não encontrado: {params_path}")

    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def get_callbacks(params: Dict[str, Any]) -> List[keras.callbacks.Callback]:
    """Configura callbacks padrão para treinamento.

    Callbacks:
        - EarlyStopping: monitora val_loss com restauração dos melhores pesos
        - ModelCheckpoint: salva melhor modelo em models/best_model.keras
        - ReduceLROnPlateau: reduz learning rate em plateau

    Args:
        params (Dict[str, Any]): Configurações carregadas de params.yaml.

    Returns:
        List[keras.callbacks.Callback]: Lista de callbacks configurados.
    """
    training_config = params["training"]

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=training_config["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            "models/best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=training_config.get("lr_factor", 0.5),
            patience=training_config.get("lr_patience", 3),
            min_lr=training_config["min_lr"],
            verbose=1,
        ),
    ]

    logger.info("Callbacks de treinamento configurados")
    return callbacks


def setup_mlflow_experiment(params: Dict[str, Any]) -> Dict[str, Any]:
    """Configura experimento MLflow com suporte a DVC nested runs e DagsHub.

    Args:
        params (Dict[str, Any]): Configurações do params.yaml.

    Returns:
        Dict[str, Any]: Argumentos extras para mlflow.start_run().
    """
    # Inicializar DagsHub
    dagshub.init(
        repo_owner=os.getenv("MLFLOW_TRACKING_USERNAME", ""),
        repo_name="cnn-face-shape",
    )

    # Configurar experimento
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    mlflow.keras.autolog()

    # Suporte a experimentos DVC aninhados
    is_dvc_experiment = os.getenv("DVC_EXP_NAME") is not None
    extra_args: Dict[str, Any] = {}

    if is_dvc_experiment:
        experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
        if experiment_id:
            runs = mlflow.search_runs(
                experiment_ids=[experiment_id],
                filter_string="tags.dvc_exp = 'True'",
                order_by=["start_time DESC"],
            )

            if runs.empty:
                with mlflow.start_run() as parent_run:
                    mlflow.set_tag("dvc_exp", "True")
                    parent_run_id = parent_run.info.run_id
            else:
                parent_run_id = runs.iloc[0].run_id

            extra_args = {
                "parent_run_id": parent_run_id,
                "run_name": os.getenv("DVC_EXP_NAME"),
                "nested": True,
            }

    logger.info(f"MLflow experiment configurado: {params['mlflow']['experiment_name']}")
    return extra_args


def save_training_artifacts(history: keras.callbacks.History) -> None:
    """Salva histórico de treinamento em metrics/training.json.

    Args:
        history (keras.callbacks.History): Objeto retornado por model.fit().
    """
    Path("metrics").mkdir(parents=True, exist_ok=True)

    metrics = {
        "train_loss": [float(x) for x in history.history["loss"]],
        "train_accuracy": [float(x) for x in history.history["accuracy"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        "epochs_trained": len(history.history["loss"]),
    }

    with open("metrics/training.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Métricas de treinamento salvas em metrics/training.json")


def train_model() -> tuple[keras.Model, keras.callbacks.History]:
    """Executa pipeline completo de treinamento.

    Returns:
        tuple[keras.Model, keras.callbacks.History]: Modelo treinado e histórico.
    """
    params = load_params()

    # Geradores de dados
    train_gen = train_generator()
    test_gen = test_generator()
    num_classes = len(train_gen.class_indices)

    steps_per_epoch = train_gen.samples // params["data"]["batch_size"]
    validation_steps = test_gen.samples // params["data"]["batch_size"]

    logger.info(
        f"Dataset carregado - Treino: {train_gen.samples:,} imgs, "
        f"Teste: {test_gen.samples:,} imgs, Classes: {num_classes}"
    )

    # Configurar MLflow
    extra_args = setup_mlflow_experiment(params)

    with mlflow.start_run(**extra_args):
        # Log de parâmetros
        mlflow.log_params(
            {k: v for k, v in params.items() if k in ["model", "training", "data"]}
        )
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("steps_per_epoch", steps_per_epoch)

        # Construir e compilar modelo
        logger.info("Construindo modelo EfficientNetB2")
        model = build_model(num_classes=num_classes)

        # Configurar callbacks
        callbacks = get_callbacks(params)

        # Treinamento
        logger.info("Iniciando treinamento")
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=params["training"]["epochs"],
            validation_data=test_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
        )

        # Métricas finais
        best_val_acc = max(history.history["val_accuracy"])
        mlflow.log_metric("best_val_accuracy", best_val_acc)

        # Salvar artefatos
        Path("models").mkdir(parents=True, exist_ok=True)
        model.save("models/best_model.keras")
        save_training_artifacts(history)

        # Log no MLflow
        mlflow.log_artifact("models/best_model.keras")
        mlflow.log_artifact("metrics/training.json")

    return model, history


def main() -> None:
    """Orquestra o pipeline completo de treinamento."""
    print("=" * 70)
    print("TREINAMENTO DO MODELO CNN - FORMATO DE ROSTO")
    print("=" * 70)

    try:
        model, history = train_model()
        best_val_acc = max(history.history["val_accuracy"])

        print("\n" + "=" * 70)
        print("TREINAMENTO CONCLUÍDO COM SUCESSO")
        print("=" * 70)
        print(f"\n Melhor precisão de validação: {best_val_acc:.2%}")
        print(f"\n Modelo salvo: models/best_model.keras")
        print(f"\n Métricas salvas: metrics/training.json")
        print("\n Artefatos logados no MLflow")

    except Exception as e:
        logger.error(f"Erro no treinamento: {e}")
        raise


if __name__ == "__main__":
    main()
