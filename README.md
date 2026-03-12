# 👓 FrameMatch — Face Shape Classifier

CNN para classificar o formato de rosto e recomendar óculos ideais, estruturada como pipeline MLOps completo com DVC, MLflow e DagsHub.

**Acurácia:** 85% · **Arquitetura:** EfficientNetB2 · **Classes:** 5 formatos de rosto

---

## 🎯 Resultados

| Classe  | Precision | Recall | F1   | Support |
|---------|-----------|--------|------|---------|
| Heart   | 0.84      | 0.87   | 0.86 | 200     |
| Oblong  | 0.89      | 0.92   | 0.90 | 200     |
| Oval    | 0.80      | 0.68   | 0.73 | 200     |
| Round   | 0.80      | 0.87   | 0.83 | 200     |
| Square  | 0.92      | 0.92   | 0.92 | 200     |
| **avg** | **0.85**  | **0.85** | **0.85** | **1000** |

---

## 🗂️ Estrutura do Projeto

```
CNN_FACE_SHAPE/
│
├── app/
│   ├── app.py                        ← FastAPI + interface web
│   └── templates/
│       └── index.html                ← UI tema ótica
│
├── src/
│   ├── preprocessing/
│   │   └── prepare.py                ← geradores de imagem + augmentation
│   ├── training/
│   │   ├── model_training.py         ← build + compile EfficientNetB2
│   │   └── train.py                  ← treino + MLflow/DagsHub
│   └── evaluation/
│   │   └── evaluate.py               ← métricas + confusion matrix
│   └── prediction/                   
│   │   └── predict.py                ← Inferências
│
├── data/
│   ├── training_set/                 ← imagens de treino (versionado DVC)
│   └── testing_set/                  ← imagens de teste (versionado DVC)
│
├── models/                           ← modelo salvo (versionado DVC)
├── metrics/                          ← JSONs e confusion matrix
├── notebooks/
│   └── experimentos_cnn_face_shape.ipynb
│
├── params.yaml                       ← todos os hiperparâmetros
├── dvc.yaml                          ← pipeline DVC
├── main.py                           ← orquestra o pipeline completo
├── docker-compose.yml
└── Dockerfile
```

---

## 🚀 Como usar

### Pré-requisitos

- Docker com suporte a GPU (`nvidia-container-toolkit`)
- Conta no [DagsHub](https://dagshub.com) para MLflow + DVC remoto

### 1. Clonar e configurar credenciais

```bash
git clone https://dagshub.com/eduardopetrocchi/cnn-face-shape.git
cd cnn-face-shape

cp .env.example .env
# edita .env com seu usuario e token do DagsHub
```

### 2. Baixar os dados via DVC

```bash
dvc pull
```

### 3. Treino completo

```bash
# Pipeline completo: prepare → train → evaluate → API
docker-compose run train

# Ou via DVC (rastreia experimentos)
dvc repro
```

### 4. Subir só a API (modelo já treinado)

```bash
docker-compose up api
# acessa http://localhost:8000
```

### 5. MLflow UI

```bash
docker-compose up mlflow
# acessa http://localhost:5000
```

---

## ⚙️ Hiperparâmetros

Tudo em `params.yaml` — sem tocar no código:

```yaml
model:
  base:           "EfficientNetB2"
  dense_units:    128
  dropout:        0.50
  label_smoothing: 0.10

training:
  learning_rate:  0.0001
  epochs:         100
  patience:       7
```

Para rodar um experimento com parâmetro diferente:

```bash
dvc exp run --set-param training.learning_rate=0.00005
dvc exp show   # compara resultados
dvc exp push origin nome_do_experimento
```

---

## 🔌 API

### `POST /predict`

Recebe uma imagem e retorna o formato do rosto e óculos recomendados.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@foto.jpg"
```

```json
{
  "face_shape": "Oval",
  "confidence": 0.87,
  "glasses": ["Qualquer estilo", "Aviador", "Wayfarer", "Redondo"],
  "all_probs": {
    "Heart": 0.04,
    "Oblong": 0.03,
    "Oval": 0.87,
    "Round": 0.03,
    "Square": 0.03
  },
  "low_confidence": false
}
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

---

## 👓 Óculos por formato

| Formato | Recomendações                              |
|---------|--------------------------------------------|
| Heart   | Aviador, Redondo, Oval, Leve/sem aro       |
| Oblong  | Redondo, Quadrado grande, Wayfarer         |
| Oval    | Qualquer estilo — Aviador, Wayfarer        |
| Round   | Quadrado, Retangular, Wayfarer, Angular    |
| Square  | Redondo, Oval, Aviador, Cat-eye            |

---

## 🧪 Evolução dos experimentos

| Configuração                        | Acurácia |
|-------------------------------------|----------|
| CNN do zero                         | ~25%     |
| MobileNetV2 (base congelada)        | ~50%     |
| EfficientNetB2 (base congelada)     | ~53%     |
| EfficientNetB2 + dados Roboflow     | ~61%     |
| EfficientNetB2 (base descongelada)  | **85%**  |

---

## 📦 Stack

- **Deep Learning:** TensorFlow · Keras · EfficientNetB2
- **MLOps:** MLflow · DVC · DagsHub
- **API:** FastAPI · Uvicorn
- **Container:** Docker · docker-compose
- **Dataset:** Kaggle Face Shape Dataset


## ✍️ Autor

**Eduardo Petrocchi**  
[linkedin.com/in/eduardopetrocchi](https://www.linkedin.com/in/eduardopetrocchi/)

---

<sub>Template HTML da interface web, README e formatação final assistidos por [Claude](https://claude.ai) (Anthropic).</sub>
