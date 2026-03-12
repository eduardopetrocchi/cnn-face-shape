import logging
from typing import Dict

import yaml
import numpy as np
from pathlib import Path
from PIL import ImageFile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.efficientnet import preprocess_input

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger("src.data.prepare")


def load_params() -> Dict[str, any]:
    """Carrega as configurações globais do arquivo params.yaml.

    Returns:
        Dict[str, any]: Dicionário completo de parâmetros do projeto.
    """
    params_path = Path("params.yaml")
    if not params_path.exists():
        raise FileNotFoundError(f"Arquivo de parâmetros não encontrado: {params_path}")

    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def train_datagen() -> ImageDataGenerator:
    """Cria ImageDataGenerator para treino com data augmentation.

    Configuração carregada de params.yaml["augmentation"].
    Inclui rotações, shifts, zoom, flip horizontal e variação de brilho.

    Returns:
        ImageDataGenerator: Gerador configurado para treinamento.
    """
    params = load_params()
    aug_config = params["augmentation"]

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=aug_config["rotation_range"],
        width_shift_range=aug_config["width_shift_range"],
        height_shift_range=aug_config["height_shift_range"],
        shear_range=aug_config["shear_range"],
        zoom_range=aug_config["zoom_range"],
        horizontal_flip=aug_config["horizontal_flip"],
        brightness_range=aug_config["brightness_range"],
        fill_mode=aug_config["fill_mode"],
    )

    logger.info("Gerador de treino com augmentation criado")
    return datagen


def test_datagen() -> ImageDataGenerator:
    """Cria ImageDataGenerator para validação/teste (sem augmentation).

    Aplica apenas preprocess_input do EfficientNet (normalização ImageNet).

    Returns:
        ImageDataGenerator: Gerador para avaliação sem modificações.
    """
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    logger.info("Gerador de teste/validação criado (sem augmentation)")
    return datagen


def train_generator() -> ImageDataGenerator:
    """Cria generator de treino a partir do diretório data/training_set.

    Args:
        N/A (configuração via params.yaml)

    Returns:
        ImageDataGenerator: Fluxo de dados para model.fit_generator ou model.fit.
    """
    params = load_params()
    data_config = params["data"]

    train_gen = train_datagen().flow_from_directory(
        data_config["train_dir"],
        target_size=tuple(data_config["image_size"]),
        batch_size=data_config["batch_size"],
        class_mode="categorical",
        shuffle=True,
        seed=data_config["seed"],
    )

    logger.info(
        f"Gerador de treino criado: {train_gen.samples} imagens, "
        f"{len(train_gen.class_indices)} classes"
    )
    return train_gen


def test_generator() -> ImageDataGenerator:
    """Cria generator de teste/validação a partir do diretório data/testing_set.

    Shuffle=False para manter ordem consistente na avaliação.

    Returns:
        ImageDataGenerator: Fluxo de dados para validação (shuffle=False).
    """
    params = load_params()
    data_config = params["data"]

    test_gen = test_datagen().flow_from_directory(
        data_config["test_dir"],
        target_size=tuple(data_config["image_size"]),
        batch_size=data_config["batch_size"],
        class_mode="categorical",
        shuffle=False,  # Mantém ordem para avaliação determinística
    )

    logger.info(
        f"Gerador de teste criado: {test_gen.samples} imagens, "
        f"{len(test_gen.class_indices)} classes"
    )
    return test_gen


def main() -> None:
    """Valida a configuração dos geradores e exibe estatísticas dos datasets.

    Testa carregamento dos diretórios e imprime:
    - Número total de imagens por split
    - Mapeamento de classes
    """
    print("=" * 60)
    print("PREPARAÇÃO DOS DADOS")
    print("=" * 60)

    try:
        train_gen = train_generator()
        test_gen = test_generator()

        print(f"\nDataset de treino: {train_gen.samples:,} imagens")
        print(f"Dataset de teste:  {test_gen.samples:,} imagens")
        print(f"Classes detectadas: {len(train_gen.class_indices)}")
        print(f"Mapeamento: {train_gen.class_indices}")

        # Verifica se classes coincidem
        if train_gen.class_indices != test_gen.class_indices:
            logger.warning("Classes diferentes entre treino e teste!")

    except Exception as e:
        logger.error(f"Erro na validação dos geradores: {e}")
        raise


if __name__ == "__main__":
    main()
