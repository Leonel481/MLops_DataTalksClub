# Comandos para ejecutar la imagen

- Build

```bash
docker build -t ride-duration-prediction-service:1 .
```

- Run

```bash
docker run -it --rm -p 9696:9696 ride-duration-prediction-service:1
```