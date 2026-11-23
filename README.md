<div align="center">

# 🛡️ NetworkSecurity MLOps

### Phishing Website Classification · Automated Training · Model Tracking · Cloud Delivery

**An end-to-end, feature-based machine learning project built around reproducible training, API inference, and a container delivery workflow.**

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Inference-009688?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-S3_%7C_ECR_%7C_EC2-232F3E?logo=amazonwebservices&logoColor=white)

[Explore the architecture](#-system-architecture) · [Run locally](#-quick-start) · [Try the API](#-inference-api) · [Understand deployment](#-cicd-and-cloud-deployment) · [Publishing checklist](#-publish-to-github-safely)

</div>

---

> [!IMPORTANT]
> This classifier accepts a **CSV of engineered website features**. It does **not** accept a raw URL and extract all features automatically. Cloud delivery is **configured as a GitHub Actions workflow**, but a successful live deployment has not been independently verified. Do not treat it as a production security service without further testing.

## 🎯 Problem statement

Phishing websites imitate legitimate websites to deceive visitors. Instead of checking every site manually, this project uses **30 numerical/categorical website indicators**—such as URL characteristics, SSL-related signals, and domain attributes—to predict a binary class. The system also addresses the engineering work around the model: storing data, validating datasets, training and comparing classifiers, versioning artifacts, serving predictions, and preparing a cloud delivery path.

## ✨ What this repository implements

| Layer | Implementation | Where to look |
|:--|:--|:--|
| Data store | MongoDB collection used as the training data source | `push_data.py`, `data_ingestion.py` |
| Ingestion | MongoDB → Pandas → train/test split (80/20) | `networksecurity/components/data_ingestion.py` |
| Validation | Schema column-count checks and Kolmogorov–Smirnov drift report | `networksecurity/components/data_validation.py` |
| Transformation | Missing-value imputation using `KNNImputer`; target `-1` mapped to `0` | `networksecurity/components/data_transformation.py` |
| Model training | Comparison of five classifiers with `GridSearchCV` | `networksecurity/components/model_trainer.py` |
| Experiment tracking | F1, precision, recall and model logging with MLflow; optional DagsHub tracking | `model_trainer.py` |
| Artifacts | Local timestamped pipeline artifacts and final-model pickles; optional S3 sync | `training_pipeline.py` |
| Serving | FastAPI CSV-upload prediction endpoint and HTML results table | `app.py`, `templates/table.html` |
| Delivery | Dockerfile + GitHub Actions configuration for an ECR image and self-hosted runner | `Dockerfile`, `.github/workflows/main.yml` |

### Data at a glance

| Property | Value found in the source archive |
|:--|:--|
| Dataset size | **11,055 records** |
| Input feature count | **30 features** |
| Target | `Result` |
| Original target values | `-1` and `1` |
| Training target mapping | `-1 → 0`, `1 → 1` |
| Train/test split | 80% / 20% (configured) |

**Model choices:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting and AdaBoost. The cleaned publishing copy uses **F1-based model selection** (the uploaded source used regression-style R² scoring for classifier comparison, which was corrected here).

## 🧠 System architecture

```mermaid
flowchart TD
  A[Website-feature CSV] --> B[Seed training records into MongoDB]
  B --> C[(MongoDB)]
  C --> D[Data ingestion]
  D --> E[80 / 20 train-test split]
  E --> F[Schema checks + K-S drift report]
  F --> G[KNN imputation + target mapping]
  G --> H[GridSearchCV: 5 classifiers]
  H --> I[Selected scikit-learn model]
  H --> J[MLflow metrics + model artifacts]
  I --> K[(final_model/model.pkl)]
  G --> L[(final_model/preprocessor.pkl)]
  K --> M[FastAPI /predict]
  L --> M
  N[CSV with 30 input features] --> M
  M --> O[HTML predictions table + output.csv]
  D -. Training pipeline artifacts .-> P[(Amazon S3 - optional)]
  I -. Final model artifacts .-> P

  classDef source fill:#e7f2ff,stroke:#3478c0,color:#102b44;
  classDef processing fill:#e7f6ed,stroke:#32865c,color:#153e25;
  classDef storage fill:#fff1df,stroke:#ce8b28,color:#5a3611;
  class A,N source;
  class D,E,F,G,H,M processing;
  class C,K,L,P storage;
```

<details>
<summary><strong>🔍 Expand: What happens in every ML pipeline stage?</strong></summary>

1. **Data ingestion:** Fetches documents from MongoDB, removes MongoDB's `_id` column, writes a local feature-store CSV, and splits records into training and testing sets.
2. **Validation:** Checks the expected number of columns against `data_schema/schema.yaml` and writes a per-column K-S drift analysis to YAML. Drift is reported, not used as a deployment gate.
3. **Data transformation:** Fits `KNNImputer(n_neighbors=3)` on the training data, transforms both datasets, and converts training targets from `-1/1` to `0/1`.
4. **Training:** Runs grid search across five candidate classifiers and selects the best candidate using binary F1 in this publishing copy.
5. **Experiment tracking:** Logs F1, precision and recall to MLflow, storing model artifacts locally or in an explicitly configured tracking server.
6. **Model output:** Writes the model and preprocessor to `final_model/`. These runtime artifacts are ignored by Git.
7. **Prediction:** A CSV upload is transformed by the saved preprocessor and passed to the saved model, with results rendered as an HTML table.

</details>

## 🧰 Technology stack

| Focus | Tools |
|:--|:--|
| Language and data | Python, Pandas, NumPy |
| Machine learning | scikit-learn, GridSearchCV, KNNImputer |
| Validation | SciPy (`ks_2samp`), YAML schema |
| Experiment tracking | MLflow, optional DagsHub |
| Database | MongoDB, PyMongo |
| Backend | FastAPI, Uvicorn, Jinja2 |
| MLOps and delivery | Docker, GitHub Actions, AWS ECR, EC2 self-hosted runner |
| Artifact storage | Local files; optional AWS S3 |

## 🚀 Quick start

### 0. Prerequisites

- Python **3.10** and `pip` installed.
- A reachable **MongoDB** database, with a user authorized to insert and read records.
- A dataset you are entitled to use, saved as **`Network_Data/phisingData.csv`**. This cleaned package intentionally excludes the original bulk dataset and generated model files pending redistribution verification.
- AWS is **not required** for the basic local training and prediction flow.

### 1. Clone and set up your environment

```bash
# Replace YOUR_USERNAME and REPOSITORY_NAME with your actual GitHub values.
git clone https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git
cd REPOSITORY_NAME

python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

<details>
<summary><strong>🪟 Windows PowerShell setup</strong></summary>

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

If activation is blocked by PowerShell execution policy, use `\.venv\Scripts\python.exe` directly rather than globally relaxing security settings.
</details>

### 2. Create `.env` and configure MongoDB

```bash
cp .env.example .env  # On Windows PowerShell: Copy-Item .env.example .env
```

Edit **`.env`** and set your own `MONGO_DB_URL`. Keep credentials in this file, not in source code. `.env` is excluded by `.gitignore` and `.dockerignore`.

```dotenv
MONGO_DB_URL=mongodb+srv://USERNAME:PASSWORD@YOUR_CLUSTER.example.mongodb.net/?retryWrites=true&w=majority
MLFLOW_TRACKING_URI=file:./mlruns
```

The seeded database and collection are configured in `networksecurity/constant/training_pipeline/__init__.py` as `KRISHAI` / `NetworkData`. Use the same database/collection when importing data.

### 3. Prepare data and seed MongoDB

```bash
mkdir -p Network_Data
# Put the dataset CSV at Network_Data/phisingData.csv
python test_mongodb.py   # Connectivity test
python push_data.py      # Inserts records into MongoDB
```

> [!NOTE]
> `push_data.py` inserts the entire CSV using `insert_many`. **Do not run it repeatedly** against the same collection unless duplicate rows are acceptable or you've cleared the previous import. MongoDB credentials and network allowlisting must be configured correctly.

### 4. Train the model locally

```bash
python main.py
```

This entry point runs ingestion → validation → transformation → training, and writes local model artifacts. It does **not** run the S3 sync stage. An optional training API route exists but is disabled unless `TRAINING_API_KEY` is configured.

Expected output locations after successful training:

```text
Artifacts/<timestamp>/       # timestamped stage outputs
final_model/model.pkl         # selected classifier
final_model/preprocessor.pkl  # fitted preprocessing pipeline
mlruns/                      # local MLflow runs by default
logs/                        # runtime logs
```

No specific model accuracy or F1 score is claimed here because no reproducible completed run was executed during this repository review.

### 5. Start the API

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive Swagger UI:** `http://127.0.0.1:8000/docs`
- **Health check:** `http://127.0.0.1:8000/health`
- **CSV prediction endpoint:** `POST http://127.0.0.1:8000/predict`

## 🔮 Inference API

`/predict` accepts a CSV containing **the same 30 input feature columns** used during training. It does not accept raw website URLs.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@valid_data/test.csv" \
  -o prediction_result.html
```

Open `prediction_result.html` in a browser. The service also writes the latest output to `prediction_output/output.csv` (ignored by Git).

<details>
<summary><strong>📑 Expand: What does the response look like?</strong></summary>

The HTML page renders the input feature columns alongside **`predicted_column`**, the model's predicted class. The prediction labels reflect the training target mapping (`-1 → 0`, `1 → 1`). Verify the semantic meaning of each original label against the authoritative dataset documentation before presenting user-facing “phishing” or “legitimate” badges.

</details>

## 🐳 Run inference with Docker

Train locally first so the two model files exist. They are deliberately **not baked into the image**.

```bash
docker build -t networksecurity-mlops:local .

docker run --rm -p 8000:8000 \
  -v "$(pwd)/final_model:/app/final_model:ro" \
  --name networksecurity-mlops \
  networksecurity-mlops:local
```

Open `http://localhost:8000/docs`. The model-volume mount is required because the Docker build excludes generated model artifacts. On Windows PowerShell, replace the volume argument with `-v "${PWD}/final_model:/app/final_model:ro"`.

## ☁️ CI/CD and cloud deployment

The repository includes **a configured deployment workflow**, not a claim of a verified public production service.

```mermaid
flowchart LR
  A[Push to main] --> B[GitHub Actions]
  B --> C[Python syntax check]
  C --> D{ENABLE_CLOUD_DEPLOYMENT = true?}
  D -- No --> E[Cloud jobs skipped]
  D -- Yes --> F[Docker build]
  F --> G[(Amazon ECR)]
  G --> H[Self-hosted runner / EC2]
  H --> I[Pull image + mount model files]
  I --> J[FastAPI container :8000]
  J --> K[Host port :8080]

  classDef cloud fill:#fff2e5,stroke:#cd8a36,color:#58370c;
  classDef code fill:#e8efff,stroke:#4f7dbc,color:#193451;
  class F,G,H cloud;
  class A,B,C,D code;
```

### Workflow responsibilities

| Job | What it does | Verification |
|:--|:--|:--|
| Integration | Checks out source and checks Python syntax | **Static check only**, not a full test suite |
| Image delivery | Builds a Docker image and pushes it to AWS ECR | Requires AWS credentials, an ECR repository and opt-in variable |
| Deployment | Runs on a self-hosted GitHub runner, pulls the image, mounts saved models, maps host `8080` to container `8000` | Requires a configured runner and prepositioned models; **not tested end to end** |

**To enable cloud jobs**, configure `ENABLE_CLOUD_DEPLOYMENT=true` as a GitHub repository variable and provide your own repository secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ECR_LOGIN_URI`, `ECR_REPOSITORY_NAME`). Prepare the ECR repository, EC2 self-hosted runner, firewall rules, and model files under the runner's `$HOME/networksecurity_models/`. Never paste secret values into this README or workflow source.

**Optional S3 artifact sync:** The API's protected `POST /train` route calls the training orchestrator; setting `TRAINING_BUCKET_NAME` enables S3 synchronization. The standalone `python main.py` flow trains locally without invoking that synchronization.

> [!WARNING]
> Public deployment needs more than a successful Docker build: add rate limiting and authentication for sensitive endpoints, restrict CORS, create meaningful tests, use least-privilege IAM, pin dependencies, and introduce model version management. A syntax-only CI check is **not** evidence of model or API correctness.

## 📂 Repository structure

```text
.
├── .github/workflows/main.yml       # Optional GitHub Actions delivery
├── .env.example                     # Safe config template
├── .gitignore                       # Local secrets/artifacts excluded
├── .dockerignore                    # Files excluded from image builds
├── app.py                           # FastAPI routes /health, /predict, /train
├── main.py                          # Standalone local training pipeline
├── push_data.py                     # Seed feature CSV into MongoDB
├── test_mongodb.py                  # MongoDB connectivity test
├── Dockerfile                       # Inference image
├── data_schema/schema.yaml         # Expected dataset columns
├── valid_data/test.csv             # Sample feature CSV for /predict
├── networksecurity/
│   ├── components/                  # Ingestion, validation, transform, train
│   ├── pipeline/training_pipeline.py
│   ├── entity/                      # Dataclass artifacts and configs
│   ├── constant/                    # Names, paths and parameters
│   ├── cloud/s3_syncer.py           # Optional S3 sync helper
│   ├── utils/                       # Serialization, metrics, evaluation
│   ├── exception/                   # Exception formatting
│   └── logging/                     # Runtime logs
├── templates/table.html            # Rendered prediction response
├── requirements.txt
└── setup.py
```

## 🔐 Publish to GitHub safely

**Do not upload the original unmodified ZIP.** It contains a hardcoded MLflow tracking credential. The cleaned publishing copy removes that value from source, but the exposed credential must still be **revoked/rotated** wherever it was issued. Never commit the original code or credentials to any Git history.

This package excludes `.env`, virtual environments, Python bytecode, timestamped training artifacts, model `.pkl` files, logs, output CSVs, and the full training CSV. Use `git status --short` and `git check-ignore -v` before your first commit.

```bash
git init
git branch -M main
git status --short
git check-ignore -v .env final_model/model.pkl Artifacts/example/model.pkl

git add .
git diff --cached --name-only    # REVIEW THIS LIST BEFORE COMMITTING
# git commit -m "Add phishing detection MLOps pipeline"
# git remote add origin https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git
# git push -u origin main
```

<details>
<summary><strong>🧹 Expand: Which files should be committed?</strong></summary>

| Commit | Keep out of Git |
|:--|:--|
| Python source, workflow YAML, schema, Dockerfile, dependency lists, templates | `.env`, API tokens, private keys |
| README, `.gitignore`, `.dockerignore`, `.env.example` | `.venv/`, `venv/`, `__pycache__/`, `.pyc` |
| Small authorized examples and meaningful tests | `Artifacts/`, `logs/`, `mlruns/`, `prediction_output/` |
| Your own code and configurations | `final_model/`, `*.pkl`, `*.npy`, full training CSV unless redistribution is permitted |

**Important Git rule:** `.gitignore` prevents *untracked* matching files from being added automatically. It does **not** remove files already committed. If sensitive data entered Git history, rotate credentials and clean history using an appropriate process.

</details>

## ⚠️ Known limitations and next improvements

- **No end-to-end integration test was run** during the ZIP review. MongoDB, AWS and any external DagsHub workspace were not contacted.
- **No full unit-test suite** currently exists; the workflow's CI job performs Python syntax checking only.
- **Model artifacts are private runtime outputs.** Supply trained models to the API/Docker deployment separately.
- **The feature-extraction step from raw URLs is not implemented.** The inference API expects precomputed feature columns.
- **Validation is basic.** Column count is checked and a K-S drift report is generated; more robust dtype, missing-feature and threshold handling would be needed for production.
- **Artifacts use Python pickle.** Never deserialize `.pkl` files from untrusted sources.
- **Training and metrics are not reproducibly benchmarked here.** Add a fixed random seed, pinned package versions, and a held-out evaluation report.
- **Training data and project provenance:** The supplied `setup.py` names **Krish Naik** as author, suggesting course/template lineage. Attribute source material accurately and verify code/dataset licensing before making the repository public.

## 🗺️ Suggested roadmap

- [x] MongoDB ingestion and feature dataset splitting
- [x] Schema validation and drift-report generation
- [x] Preprocessing, candidate model search and MLflow logging
- [x] FastAPI CSV prediction and HTML response
- [x] Docker and AWS ECR/EC2 workflow configuration
- [ ] Automated tests with meaningful assertions
- [ ] Verified cloud deployment and health checks
- [ ] Automated raw-URL feature extraction
- [ ] Model/version registry with promotion gates
- [ ] Monitoring, retraining triggers, and alerting

---

<div align="center">

**Built to demonstrate practical MLOps building blocks — with implementation status documented transparently.**

[⬆ Back to top](#️-networksecurity-mlops)

</div>
