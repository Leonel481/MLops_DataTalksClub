FROM apache/airflow:3.0.1


ENV AIRFLOW_HOME=/opt/airflow

USER root
RUN apt-get update -qq && apt-get install vim -qqq

COPY requirements.txt .

SHELL ["/bin/bash", "-o", "pipefail", "-e", "-u", "-x", "-c"]

# COPY scripts scripts
# RUN chmod +x scripts/entrypoint.sh

USER $AIRFLOW_UID
WORKDIR $AIRFLOW_HOME
RUN pip install --no-cache-dir -r requirements.txt