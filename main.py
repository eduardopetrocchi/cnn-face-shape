import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

logger = None  # Configurado no main()


def run_step(step_name: str, command: List[str]) -> bool:
    """Executa um passo específico do pipeline ML.

    Args:
        step_name (str): Nome descritivo do passo para logging.
        command (List[str]): Lista de argumentos do comando a ser executado.

    Returns:
        bool: True se o passo foi executado com sucesso, False caso contrário.
    """
    print("=" * 60)
    print(f"EXECUTANDO: {step_name}")
    print("=" * 60)

    try:
        result = subprocess.run(
            command, 
            capture_output=False, 
            text=True, 
            check=True
        )
        print(f"\n✅ {step_name} CONCLUÍDO COM SUCESSO")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n ERRO em {step_name}:")
        print(f"  Código: {e.returncode}")
        print(f"  Saída: {e.stdout}")
        print(f"  Erro: {e.stderr}")
        return False


def check_requirements() -> bool:
    """Verifica se todos os arquivos fonte do projeto existem.

    Returns:
        bool: True se todos os arquivos necessários estão presentes.
    """
    required_files = [
        Path("params.yaml"),
        Path("src/data/prepare.py"),
        Path("src/training/model_training.py"),
        Path("src/training/train.py"),
        Path("src/evaluation/evaluate.py"), 
        Path("src/prediction/predict.py"),
        Path("app/app.py"),
    ]

    print("VERIFICANDO REQUISITOS")
    print("-" * 40)

    all_ok = True
    for file_path in required_files:
        status = "OK" if file_path.exists() else "FALTANDO"
        print(f"  [{status}] {file_path}")
        if not file_path.exists():
            all_ok = False

    return all_ok


def check_data() -> bool:
    """Verifica se diretórios de dados existem e contêm imagens.

    Returns:
        bool: True se os datasets estão prontos para treinamento.
    """
    data_dirs = [
        Path("data/training_set"),
        Path("data/testing_set")
    ]

    print("VERIFICANDO DADOS")
    print("-" * 40)

    all_ok = True
    for data_dir in data_dirs:
        if not data_dir.exists():
            print(f" Diretório não encontrado: {data_dir}")
            all_ok = False
            continue

        # Conta imagens JPG e PNG recursivamente
        images = list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.png"))
        img_count = len(images)
        
        status = "OK" if img_count > 0 else "VAZIO"
        print(f"  [{status}] {data_dir} ({img_count:,} imagens)")
        
        if img_count == 0:
            all_ok = False

    return all_ok


def check_artifacts() -> None:
    """Lista status dos artefatos gerados pelo pipeline."""
    print("\n" + "=" * 60)
    print("ARTEFATOS GERADOS")
    print("=" * 60)

    artifacts = [
        ("Modelo treinado", Path("models/best_model.keras")),
        ("Métricas treino", Path("metrics/training.json")),
        ("Métricas avaliação", Path("metrics/evaluation.json")),
        ("Matriz de confusão", Path("metrics/confusion_matrix.png")),
    ]

    for name, artifact_path in artifacts:
        status = "OK" if artifact_path.exists() else "PENDENTE"
        print(f"  [{status}] {name}: {artifact_path}")


def main() -> None:
    """Executa pipeline completo: validação → prepare → train → evaluate → API."""
    print("=" * 70)
    print("PIPELINE ML - CLASSIFICAÇÃO DE FORMATO DE ROSTO")
    print("=" * 70)

    # Configurar logging
    import logging
    logging.basicConfig(level=logging.INFO)
    global logger
    logger = logging.getLogger("pipeline")

    try:
        # 1. Verificar pré-requisitos
        print("\n1. VERIFICANDO REQUISITOS")
        if not check_requirements():
            print("\n REQUISITOS NÃO ATENDIDOS")
            print("Corrija os arquivos faltantes e execute novamente.")
            sys.exit(1)
        print(" Requisitos OK")

        # 2. Verificar dados
        print("\n2. VERIFICANDO DADOS")
        if not check_data():
            print("\n DADOS NÃO ENCONTRADOS")
            print("Coloque imagens em:")
            print("  - data/training_set/")
            print("  - data/testing_set/")
            sys.exit(1)
        print(" Dados OK")

        # 3. Executar pipeline
        pipeline_steps: List[Tuple[str, List[str]]] = [
            ("Preparação dos dados", ["python", "-m", "src.data.prepare"]),
            ("Treinamento do modelo", ["python", "-m", "src.training.train"]),
            ("Avaliação do modelo", ["python", "-m", "src.evaluation.evaluate"]),
        ]

        print("\n3. EXECUTANDO PIPELINE")
        for step_name, command in pipeline_steps:
            if not run_step(step_name, command):
                print(f"\n PIPELINE INTERROMPIDO em: {step_name}")
                sys.exit(1)

        # 4. Verificar resultados
        check_artifacts()

        # 5. Iniciar API
        print("\n" + "=" * 70)
        print("INICIANDO API FASTAPI")
        print("=" * 70)
        print("Acesse:")
        print("  Interface: http://localhost:8000")
        print("  Documentação: http://localhost:8000/docs")
        print("  ReDoc: http://localhost:8000/redoc")
        print("-" * 70)
        print("Pressione CTRL+C para encerrar")
        print("=" * 70)

        subprocess.run(["python", "-m", "app.app"])

    except KeyboardInterrupt:
        print("\n\nPipeline interrompido pelo usuário (CTRL+C)")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erro inesperado no pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
