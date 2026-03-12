import io
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any

import yaml
from PIL import ImageFile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from src.prediction.predict import predict

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger("app.api")


def load_params() -> Dict[str, Any]:
    """Carrega configurações da API do arquivo params.yaml.

    Returns:
        Dict[str, Any]: Parâmetros da seção 'api' do params.yaml.
    """
    params_path = Path("params.yaml")
    if not params_path.exists():
        raise FileNotFoundError(f"Arquivo params.yaml não encontrado: {params_path}")

    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    api_params = params.get("api", {})
    api_params.setdefault("host", "0.0.0.0")
    api_params.setdefault("port", 8000)
    api_params.setdefault("model_path", "models/best_model.keras")

    return api_params


PARAMS = load_params()
CONFIDENCE_THRESHOLD = 0.40


# Inicializar FastAPI
app = FastAPI(
    title="Face Shape Classifier API",
    description="Classifica formato de rosto em 5 categorias e recomenda óculos ideais.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve página HTML principal com formulário de upload."""
    html_path = Path("app/templates/index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Página inicial não encontrada")

    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Endpoint de health check da API.

    Verifica se modelo existe e API está funcional.

    Returns:
        Dict[str, Any]: Status da API e disponibilidade do modelo.
    """
    model_path = Path(PARAMS["model_path"])
    model_available = model_path.exists()

    return {
        "status": "healthy",
        "model_available": model_available,
        "model_path": str(model_path),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.post("/predict", response_model=Dict[str, Any])
async def predict_endpoint(
    file: UploadFile = File(..., description="Imagem do rosto (JPG/PNG)")
) -> Dict[str, Any]:
    """Realiza predição de formato de rosto e retorna recomendações.

    Args:
        file (UploadFile): Imagem JPG ou PNG contendo rosto frontal.

    Raises:
        HTTPException: Erro 400 para formato inválido, 500 para erro de predição.

    Returns:
        Dict[str, Any]: Resultados com face_shape, confidence, glasses, all_probs e low_confidence.
    """
    # Validação de formato
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(
            status_code=400, detail="Formato inválido. Envie imagem JPG ou PNG."
        )

    # Processar imagem temporariamente
    image_bytes = await file.read()

    with tempfile.NamedTemporaryFile(
        suffix=".jpg", delete=False, dir="/tmp"
    ) as tmp_file:
        tmp_file.write(image_bytes)
        tmp_path = Path(tmp_file.name)

    try:
        logger.info(f"Predição para imagem: {file.filename}")
        result = predict(image_path=str(tmp_path))

        # Adicionar metadados
        result["low_confidence"] = result["confidence"] < CONFIDENCE_THRESHOLD
        if result["low_confidence"]:
            result["warning"] = (
                f"Confiança baixa ({result['confidence']:.0%}). "
                "Tente uma foto com rosto centralizado e boa iluminação."
            )

        logger.info(
            f"Predição concluída: {result['face_shape']} ({result['confidence']:.2%})"
        )
        return result

    except FileNotFoundError as e:
        logger.error(f"Modelo não encontrado: {e}")
        raise HTTPException(
            status_code=500,
            detail="Modelo não encontrado. Execute o treinamento primeiro.",
        )
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(
            status_code=500, detail=f"Erro interno na predição: {str(e)}"
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/params")
async def get_params() -> Dict[str, Any]:
    """Expõe configurações da API para debug."""
    return {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "model_path": PARAMS["model_path"],
        "api_config": {
            k: v
            for k, v in PARAMS.items()
            if k not in ["model_path", "glasses_map"]  # Não expõe dados sensíveis
        },
    }


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Iniciando API em {PARAMS['host']}:{PARAMS['port']}")
    uvicorn.run(
        "app.app:app",
        host=PARAMS["host"],
        port=PARAMS["api"]["port"],
        reload=False,
        log_level="info",
    )
