#!/bin/bash
set -e

echo "Inicializando la base de datos (si es necesario)..."
airflow db migrate

echo "Iniciando Airflow en modo standalone..."
airflow standalone &

