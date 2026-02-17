# MLOps Best Project: Intelligent Credit Risk Scoring System

## Overview

This repository contains a comprehensive MLOps project that implements an intelligent credit‑risk scoring system. The goal is to demonstrate end‑to‑end machine‑learning operations using modern tools that ensure reproducibility, automation, scalability, monitoring and governance.

The system predicts the probability of loan default using historical loan application and repayment data. It is built to run in production with a fully automated pipeline covering data ingestion, validation, feature engineering, training, deployment, monitoring and automatic retraining.

### Key Features

- **Data Ingestion & Validation**: Raw data is ingested via Apache Airflow and validated using Great Expectations.
- **Data Versioning**: All datasets are versioned with DVC to guarantee reproducibility.
- **Feature Store**: Feast manages online and offline features to avoid training/serving skew.
- **Model Training**: Training pipelines support traditional models (XGBoost) and neural architectures (PyTorch). Experiments are tracked with MLflow.
- **Model Registry**: Models are stored and promoted through stages (Staging, Production, Archived) using the MLflow Model Registry.
- **Continuous Integration/Continuous Deployment (CI/CD)**: GitHub Actions orchestrate unit tests, data checks, Docker builds and Kubernetes deployments.
- **Containerisation & Orchestration**: The application is containerised with Docker and deployed to Kubernetes for scalability and resilience.
- **Serving Layer**: A FastAPI microservice exposes predictions via a REST API.
- **Monitoring & Drift Detection**: Prometheus and Grafana collect operational metrics while Evidently AI monitors data and model drift. Alerts trigger automatic retraining.
- **Retraining Pipeline**: Airflow jobs retrain the model on a schedule or upon drift detection, comparing performance and promoting the best model.

## Directory Structure

```
mlops_best_project/
├── airflow/                 # Airflow DAG definitions
│   └── dags/
│       └── pipeline.py
├── api/                     # FastAPI application for serving predictions
│   └── app.py
├── data/                    # Raw data and DVC configuration
│   └── README.md
├── docker/                  # Docker images
│   └── Dockerfile
├── dvc.yaml                 # DVC pipeline definition
├── feature_store/           # Feast feature store configuration
│   └── feast_feature_repo/
│       ├── feature_store.yaml
│       └── driver_features.py
├── k8s/                     # Kubernetes manifests
│   ├── deployment.yaml
│   └── service.yaml
├── monitoring/              # Monitoring and drift detection
│   ├── monitoring.py
│   ├── grafana_dashboard.json
│   └── prometheus_config.yaml
├── .github/
│   └── workflows/
│       └── ci_cd.yml        # GitHub Actions workflow
├── requirements.txt         # Python dependencies
├── tests/                   # Unit and integration tests
│   ├── test_training.py
│   └── test_api.py
├── training/                # Training code and configs
│   ├── train.py
│   └── config/
│       └── training_config.yaml
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Kubernetes cluster (e.g. Minikube)
- Git and DVC
- Optional: MLflow Tracking server, Prometheus, Grafana

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-fork-url>
   cd mlops_best_project
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up DVC remote** (e.g. AWS S3, Azure Blob, GCS) and pull data:
   ```bash
   dvc remote add -d myremote s3://my-bucket/path
   dvc pull
   ```

4. **Initialise Feast feature store**:
   ```bash
   cd feature_store/feast_feature_repo
   feast apply
   cd ../../
   ```

5. **Run Airflow locally** (optional) or deploy to a scheduler.

6. **Build and run the API locally**:
   ```bash
   docker build -t credit-risk-api:latest -f docker/Dockerfile .
   docker run -p 8000:80 credit-risk-api:latest
   ```

7. **Deploy to Kubernetes**:
   ```bash
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   ```

### Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
