import logging
from typing import Dict, Tuple

import yaml
import tensorflow as tf
import keras
from keras.applications import EfficientNetB2

logger = logging.getLogger("src.training.model_training")


def load_params() -> Dict[str, any]:
    """Carrega as configurações globais do arquivo params.yaml.

    Returns:
        Dict[str, any]: Dicionário completo de parâmetros do projeto.
    """
    params_path = "params.yaml"
    
    try:
        with open(params_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Arquivo de parâmetros não encontrado: {params_path}. "
            "Verifique se params.yaml existe na raiz do projeto."
        )


def build_model(num_classes: int) -> keras.Model:
    """Constrói modelo EfficientNetB2 com transfer learning para classificação.

    Arquitetura:
        - Base EfficientNetB2 pré-treinada ImageNet (não congelada)
        - GlobalAveragePooling2D + BatchNorm + Dense + Dropout + Softmax

    Args:
        num_classes (int): Número de classes do dataset (ex: 5 formatos de rosto).

    Raises:
        ValueError: Se num_classes <= 0 ou configurações inválidas.

    Returns:
        keras.Model: Modelo compilado e pronto para treinamento.
    """
    if num_classes <= 0:
        raise ValueError("num_classes deve ser maior que zero")

    params = load_params()
    model_config = params["model"]
    data_config = params["data"]
    training_config = params["training"]

    # Input shape: (H, W, 3)
    input_shape = tuple(data_config["image_size"]) + (3,)

    # Carregar base pré-treinada
    logger.info(f"Carregando EfficientNetB2 com weights='{model_config['weights']}'")
    base_model = EfficientNetB2(
        weights=model_config["weights"],
        include_top=model_config["include_top"],
        input_tensor=keras.Input(shape=input_shape),
    )
    base_model.trainable = True  # Fine-tuning completo

    # Head customizado
    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(base_model.output)
    x = keras.layers.BatchNormalization(name="batch_norm")(x)
    x = keras.layers.Dense(
        model_config["dense_units"], 
        activation="relu", 
        name="dense"
    )(x)
    x = keras.layers.Dropout(model_config["dropout"], name="dropout")(x)
    outputs = keras.layers.Dense(
        num_classes, 
        activation="softmax", 
        name="predictions"
    )(x)

    # Modelo final
    model = keras.Model(
        inputs=base_model.input,
        outputs=outputs,
        name="FaceShapeClassifier"
    )

    # Compilação
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=training_config["learning_rate"]
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=model_config["label_smoothing"]
        ),
        metrics=["accuracy"],
    )

    logger.info(
        f"\n Modelo construído: EfficientNetB2 + {num_classes} classes, "
        f"input_shape={input_shape}, lr={training_config['learning_rate']}"
    )
    return model


def main() -> None:
    """Testa a construção do modelo com 5 classes (número padrão do dataset)."""
    print("=" * 60)
    print("CONSTRUÇÃO DO MODELO")
    print("=" * 60)

    try:
        model = build_model(num_classes=5)
        model.summary()
        print("\nModelo construído com sucesso.")
        print(f"\n Parâmetros treináveis: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    except Exception as e:
        logger.error(f"Erro na construção do modelo: {e}")
        raise


if __name__ == "__main__":
    main()
