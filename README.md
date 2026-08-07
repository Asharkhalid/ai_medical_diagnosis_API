# AI Medical Diagnosis API (Backend)

This is the backend for the AI-Based Medical Diagnosis System. It leverages **FastAPI**, **PostgreSQL**, **Scikit-Learn (Machine Learning)**, and **NLTK (Natural Language Processing)** to predict potential diseases from user-submitted symptoms.

## Features Implemented
- **NLP Engine**: Tokenization, Stop-word removal, Lemmatization, and Synonym Mapping (e.g., mapping "temperature" to "fever").
- **Feature Engineering**: Binary Encoding (symptom presence = 1 or 0), Severity Weighting (Low=1, High=3), Duration Encoding.
- **Machine Learning**: A robust Random Forest classifier trained on 5,000 synthetic clinical records covering 15 diseases.
- **Explainable AI (XAI)**: The model extracts feature importances to show the user exactly *why* a disease was predicted and which symptoms influenced the prediction.

## Directory Structure
- `main.py`: The FastAPI application and core routing.
- `models.py`: SQLAlchemy database models representing the Database ERD.
- `schemas.py`: Pydantic data validation schemas for requests and responses.
- `database.py`: PostgreSQL connection and SQLAlchemy setup.
- `ml_engine.py`: Contains the `MedicalNLPEngine` and `DiagnosticModel` singletons that handle all NLP processing and ML inference.
- `dataset_generator.py`: Script to auto-generate the `symptoms_dataset.csv` with randomized diseases, mapped symptoms, severity, and duration.
- `train_model.py`: Script to train the `RandomForestClassifier` on the generated dataset and save it to the `models/` directory using joblib.

## Prerequisites
- Python 3.12+
- PostgreSQL (Running locally on port 5432 with default `postgres:postgres` credentials, and an empty database named `ai_diagnosis`)

## Setup & Running

1. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic[email] python-multipart passlib[bcrypt] pyjwt scikit-learn nltk joblib numpy pandas
   ```

3. **Generate Dataset & Train Model:**
   ```bash
   python dataset_generator.py
   python train_model.py
   ```
   *Note: This will generate `symptoms_dataset.csv` and a `models/` folder containing `random_forest_v1.pkl` and `feature_columns.pkl`.*

4. **Start the FastAPI Server:**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

5. **Test the API:**
   Navigate to `http://localhost:8000/docs` to view the interactive Swagger documentation and test the `/api/diagnose` endpoint.
