import logging
from typing import Dict, List

import yaml
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile

import keras
from keras.applications.efficientnet import preprocess_input

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger("src.prediction.predict")

# Modelo carregado globalmente (singleton pattern)
_model: keras.Model | None = None


def load_params() -> dict:
    """Carrega as configurações globais do arquivo params.yaml.

    Returns:
        dict: Dicionário completo de parâmetros.
    """
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def get_model() -> keras.Model:
    """Carrega o modelo treinado em memória (lazy loading).

    O modelo é carregado apenas na primeira chamada e reutilizado em chamadas subsequentes.

    Raises:
        FileNotFoundError: Se o arquivo do modelo não existir.

    Returns:
        keras.Model: Modelo EfficientNetB2 treinado e compilado para inferência.
    """
    global _model
    if _model is None:
        model_path = Path("models/best_model.keras")
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {model_path}. "
                "Execute o treinamento primeiro com `python -m src.training.train`."
            )
        logger.info(f"Carregando modelo: {model_path}")
        _model = keras.models.load_model(model_path)

    return _model


def preprocess_image(image_path: str) -> np.ndarray:
    """Carrega e pré-processa uma imagem para inferência com EfficientNet.

    Aplica redimensionamento, conversão RGB e preprocess_input (normalização ImageNet).

    Args:
        image_path (str): Caminho para imagem JPG ou PNG.

    Raises:
        FileNotFoundError: Se a imagem não existir.
        ValueError: Se o formato da imagem for inválido.

    Returns:
        np.ndarray: Tensor de shape (1, H, W, 3) normalizado e pronto para model.predict.
    """
    params = load_params()
    target_size = tuple(params["data"]["image_size"])

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(
        img_array, axis=0
    )  # Adiciona batch dimension (1, H, W, 3)

    return preprocess_input(img_array)


def predict(image_path: str) -> Dict[str, str | float | List[str] | Dict[str, float]]:
    """Classifica o formato do rosto na imagem e recomenda modelos de óculos.

    Args:
        image_path (str): Caminho para a imagem de entrada.

    Raises:
        FileNotFoundError: Se imagem ou modelo não existirem.
        RuntimeError: Se ocorrer erro na predição.

    Returns:
        dict: Resultados da predição com:
            - "face_shape": Classe prevista (ex: "Oval")
            - "confidence": Confiança da predição (0-1)
            - "glasses": Lista de óculos recomendados
            - "all_probs": Dict com probabilidades de todas as classes
    """
    params = load_params()
    model = get_model()

    classes = sorted(params["data"]["classes"])
    glasses_map = params["api"]["glasses_map"]

    processed_img = preprocess_image(image_path)
    probabilities = model.predict(processed_img, verbose=0)[0]

    pred_idx = int(np.argmax(probabilities))
    pred_class = classes[pred_idx]
    confidence = float(probabilities[pred_idx])

    result = {
        "face_shape": pred_class,
        "confidence": round(confidence, 4),
        "glasses": glasses_map[pred_class],
        "all_probs": {
            cls: round(float(prob), 4) for cls, prob in zip(classes, probabilities)
        },
    }

    logger.info(f"Predição: {pred_class} ({confidence:.2%})")
    return result


def main() -> None:
    """Executa predição rápida via linha de comando.

    Uso:
        python -m src.prediction.predict caminho/para/imagem.jpg

    Exibe resultados formatados com barras de probabilidade.
    """
    import sys

    if len(sys.argv) < 2:
        print('Uso: python -m src.prediction.predict "caminho/para/imagem.jpg"')
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f" Imagem não encontrada: {image_path}")
        sys.exit(1)

    print("=" * 60)
    print("PREDIÇÃO DE FORMATO DE ROSTO")
    print("=" * 60)

    try:
        result = predict(image_path)

        print(f"\n Formato do rosto: **{result['face_shape']}**")
        print(f" Confiança: **{result['confidence']:.2%}**")

        print(f"\n Óculos recomendados:")
        for glasses in result["glasses"]:
            print(f"   • {glasses}")

        print(f"\n Probabilidades por classe:")
        for cls, prob in result["all_probs"].items():
            bar = "█" * int(prob * 30)
            print(f"   {cls:<10} {prob:.2%} |{bar}|")

    except Exception as e:
        print(f" Erro na predição: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
