from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
import subprocess
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
import models, schemas
from database import engine, get_db
from ml_engine import nlp_engine, diagnostic_model

# Config
SECRET_KEY = "my_super_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Medical Diagnosis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired. Please log in again.")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

@app.post("/api/auth/register", response_model=schemas.UserResponse)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    new_user = models.User(
        full_name=user.full_name,
        email=user.email,
        password_hash=get_password_hash(user.password),
        role=user.role
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/api/auth/login", response_model=schemas.Token)
def login_user(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": str(db_user.user_id), "role": db_user.role})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/diagnose")
def predict_diagnosis(req: schemas.DiagnosisRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    extracted_symptoms, spell_corrections = nlp_engine.extract_symptoms(req.input_text)
    
    # NLP Engine Text-based extraction overrides UI dropdowns if present
    text_severity = nlp_engine.extract_severity(req.input_text)
    text_duration = nlp_engine.extract_duration(req.input_text)
    
    final_severity = text_severity if text_severity else req.severity
    final_duration = text_duration if text_duration else req.duration
    
    feature_vector = nlp_engine.vectorize_features(extracted_symptoms, final_severity, final_duration)
    
    mock_prediction = diagnostic_model.predict(feature_vector, extracted_symptoms)
    
    diagnosis = models.DiagnosisHistory(
        user_id=current_user.user_id,
        input_text=req.input_text,
        extracted_symptoms=extracted_symptoms,
        severity=final_severity,
        duration=final_duration,
        predictions=mock_prediction,
        model_name="RandomForestClassifier",
        model_version="v1.2",
        validation_status=0
    )
    
    db.add(diagnosis)
    db.commit()
    db.refresh(diagnosis)
    
    # Return as dict to include spell corrections
    response = {column.name: getattr(diagnosis, column.name) for column in diagnosis.__table__.columns}
    response["spell_corrections"] = spell_corrections
    
    return response

@app.get("/api/history")
def get_diagnosis_history(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    history = db.query(models.DiagnosisHistory).filter(models.DiagnosisHistory.user_id == current_user.user_id).order_by(models.DiagnosisHistory.prediction_timestamp.desc()).all()
    results = []
    for diag in history:
        feedback = db.query(models.DiagnosisFeedback).filter(models.DiagnosisFeedback.diagnosis_id == diag.diagnosis_id).order_by(models.DiagnosisFeedback.created_at.desc()).first()
        item = {column.name: getattr(diag, column.name) for column in diag.__table__.columns}
        item["feedback"] = feedback.comments if feedback else ""
        results.append(item)
    return results

@app.get("/api/users/me")
def get_user_profile(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    diagnosis_count = db.query(models.DiagnosisHistory).filter(models.DiagnosisHistory.user_id == current_user.user_id).count()
    return {
        "user_id": current_user.user_id,
        "full_name": current_user.full_name,
        "email": current_user.email,
        "role": current_user.role,
        "created_at": current_user.created_at,
        "diagnosis_count": diagnosis_count
    }

@app.get("/api/admin/synonyms")
def get_admin_synonyms(current_user: models.User = Depends(get_current_user)):
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized. Admins only.")
    return nlp_engine.synonym_map

@app.post("/api/admin/synonyms")
def add_admin_synonym(req: schemas.SynonymRequest, current_user: models.User = Depends(get_current_user)):
    if current_user.role != 3: # 3 = Admin
        raise HTTPException(status_code=403, detail="Not authorized. Admins only.")
    nlp_engine.add_synonym(req.phrase, req.target)
    return {"message": f"Synonym '{req.phrase}' added and mapped to '{req.target}' successfully."}

@app.delete("/api/admin/synonyms/{phrase}")
def delete_admin_synonym(phrase: str, current_user: models.User = Depends(get_current_user)):
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized. Admins only.")
    
    success = nlp_engine.remove_synonym(phrase)
    if not success:
        raise HTTPException(status_code=404, detail=f"Synonym '{phrase}' not found.")
    
    return {"message": f"Synonym '{phrase}' removed successfully."}

@app.get("/api/admin/datasets")
def get_datasets(current_user: models.User = Depends(get_current_user)):
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized.")
    
    dataset_path = 'symptoms_dataset.csv'
    if not os.path.exists(dataset_path):
        return {"datasets": []}
        
    stats = os.stat(dataset_path)
    # count lines manually for records
    with open(dataset_path, 'r', encoding='utf-8') as f:
        records = sum(1 for _ in f) - 1 # minus header
        
    return {
        "datasets": [{
            "name": "symptoms_dataset.csv",
            "records": records,
            "last_updated": datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Active"
        }]
    }

@app.post("/api/admin/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), current_user: models.User = Depends(get_current_user)):
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized.")
        
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    content = await file.read()
    with open('symptoms_dataset.csv', 'wb') as f:
        f.write(content)
        
    return {"message": "Dataset uploaded successfully."}

@app.post("/api/admin/datasets/train")
def train_model(current_user: models.User = Depends(get_current_user)):
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized.")
        
    try:
        # Run training script
        process = subprocess.run(['python', 'train_model.py'], capture_output=True, text=True, check=True)
        # Reload model into memory
        diagnostic_model.__init__()
        nlp_engine.__init__()
        return {"message": "Model trained and reloaded successfully!", "logs": process.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e.stderr}")

from pydantic import BaseModel
class RoleUpdateRequest(BaseModel):
    role: int

@app.get("/api/admin/users")
def get_all_users(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized.")
        
    users = db.query(models.User).order_by(models.User.created_at.desc()).all()
    return [{"user_id": u.user_id, "full_name": u.full_name, "email": u.email, "role": u.role, "created_at": u.created_at} for u in users]

@app.put("/api/admin/users/{target_user_id}/role")
def update_user_role(target_user_id: str, req: RoleUpdateRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized.")
        
    user = db.query(models.User).filter(models.User.user_id == target_user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    user.role = req.role
    db.commit()
    
    return {"message": f"User {user.email} role updated successfully"}

@app.get("/api/doctor/validations")
def get_all_validations(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role not in [2, 3]:
        raise HTTPException(status_code=403, detail="Not authorized.")
        
    records = db.query(models.DiagnosisHistory, models.User.full_name).join(
        models.User, models.DiagnosisHistory.user_id == models.User.user_id
    ).order_by(models.DiagnosisHistory.prediction_timestamp.desc()).all()
    
    results = []
    for diag, patient_name in records:
        feedback = db.query(models.DiagnosisFeedback).filter(models.DiagnosisFeedback.diagnosis_id == diag.diagnosis_id).order_by(models.DiagnosisFeedback.created_at.desc()).first()
        results.append({
            "diagnosis_id": str(diag.diagnosis_id),
            "patient": patient_name,
            "symptoms": diag.input_text,
            "prediction": diag.predictions.get('disease') if isinstance(diag.predictions, dict) else diag.predictions,
            "confidence": diag.predictions.get('confidence') if isinstance(diag.predictions, dict) else 0,
            "date": diag.prediction_timestamp.isoformat(),
            "status": diag.validation_status,
            "feedback": feedback.comments if feedback else ""
        })
    return results

@app.put("/api/doctor/validations/{diagnosis_id}")
def update_validation_status(diagnosis_id: str, req: schemas.ValidationRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role not in [2, 3]:
        raise HTTPException(status_code=403, detail="Not authorized.")
        
    diag = db.query(models.DiagnosisHistory).filter(models.DiagnosisHistory.diagnosis_id == diagnosis_id).first()
    if not diag:
        raise HTTPException(status_code=404, detail="Diagnosis not found")
        
    diag.validation_status = req.status
    db.commit()
    return {"message": "Status updated successfully"}

@app.post("/api/doctor/validations/{diagnosis_id}/feedback")
def add_feedback(diagnosis_id: str, req: schemas.FeedbackRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role not in [2, 3]:
        raise HTTPException(status_code=403, detail="Not authorized.")
        
    import uuid
    diag_uuid = uuid.UUID(diagnosis_id)
    feedback = db.query(models.DiagnosisFeedback).filter(models.DiagnosisFeedback.diagnosis_id == diag_uuid).order_by(models.DiagnosisFeedback.created_at.desc()).first()
    
    if feedback:
        feedback.comments = req.comments
        feedback.doctor_id = current_user.user_id
    else:
        feedback = models.DiagnosisFeedback(
            diagnosis_id=diagnosis_id,
            doctor_id=current_user.user_id,
            comments=req.comments
        )
        db.add(feedback)
        
    db.commit()
    return {"message": "Feedback saved successfully"}

@app.get("/api/doctor/rules")
def get_expert_rules(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role not in [2, 3]:
        raise HTTPException(status_code=403, detail="Not authorized.")
    return db.query(models.ExpertRule).order_by(models.ExpertRule.priority.desc()).all()

@app.post("/api/doctor/rules")
def create_expert_rule(req: schemas.ExpertRuleRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role not in [2, 3]:
        raise HTTPException(status_code=403, detail="Not authorized.")
    new_rule = models.ExpertRule(
        condition=req.condition,
        target=req.target,
        action=req.action,
        priority=req.priority,
        created_by=current_user.user_id
    )
    db.add(new_rule)
    db.commit()
    db.refresh(new_rule)
    return new_rule

@app.put("/api/doctor/rules/{rule_id}")
def update_expert_rule(rule_id: str, req: schemas.ExpertRuleRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role not in [2, 3]:
        raise HTTPException(status_code=403, detail="Not authorized.")
    rule = db.query(models.ExpertRule).filter(models.ExpertRule.rule_id == rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    rule.condition = req.condition
    rule.target = req.target
    rule.action = req.action
    rule.priority = req.priority
    db.commit()
    db.refresh(rule)
    return rule

@app.delete("/api/doctor/rules/{rule_id}")
def delete_expert_rule(rule_id: str, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role not in [2, 3]:
        raise HTTPException(status_code=403, detail="Not authorized.")
    rule = db.query(models.ExpertRule).filter(models.ExpertRule.rule_id == rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    db.delete(rule)
    db.commit()
    return {"message": "Rule deleted successfully"}
