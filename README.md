# Risk-Aware-Models

## How to build docker container

`docker build -t cvae_app:latest -f Dockerfile .`

## How to run streamlit after building

`docker run -p 8501:8501 cvae_app:latest`
