from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import models, schemas
from database import engine, get_db
from ml_engine import nlp_engine, diagnostic_model

# Create DB tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Medical Diagnosis API")

# Configure CORS for React UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to AI Medical Diagnosis API"}

@app.post("/api/auth/register", response_model=schemas.UserResponse)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Note: Mocking auth for now, use bcrypt in production
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    new_user = models.User(
        full_name=user.full_name,
        email=user.email,
        password_hash=user.password, # Plaintext for mock, will hash later
        role=user.role
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/api/diagnose")
def predict_diagnosis(req: schemas.DiagnosisRequest, db: Session = Depends(get_db)):
    # Process natural language input via NLP pipeline
    extracted_symptoms = nlp_engine.extract_symptoms(req.input_text)
    feature_vector = nlp_engine.vectorize_features(extracted_symptoms, req.severity, req.duration)
    
    # ML Prediction + Explainable AI Generation
    mock_prediction = diagnostic_model.predict(feature_vector, extracted_symptoms)
    
    # Store in History
    # Note: Hardcoding a user_id just for testing if auth isn't fully integrated
    user = db.query(models.User).first()
    if not user:
        raise HTTPException(status_code=404, detail="No users exist to bind this diagnosis to. Please register a user.")

    diagnosis = models.DiagnosisHistory(
        user_id=user.user_id,
        input_text=req.input_text,
        extracted_symptoms=extracted_symptoms,
        severity=req.severity,
        duration=req.duration,
        predictions=mock_prediction,
        model_name="RandomForestClassifier",
        model_version="v1.2",
        validation_status=0 # Pending
    )
    
    db.add(diagnosis)
    db.commit()
    db.refresh(diagnosis)
    
    return diagnosis
