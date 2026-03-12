FROM tensorflow/tensorflow:latest-gpu-jupyter

WORKDIR /app

# Dependências
COPY pyproject.toml .

RUN apt-get update && apt-get remove -y python3-blinker

RUN pip install --upgrade pip
RUN pip install .

# Código
COPY . .

# Porta da API
EXPOSE 8000

CMD [ "python", "-m", "app.app" ]