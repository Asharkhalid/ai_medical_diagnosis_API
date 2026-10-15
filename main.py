from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Request
from fastapi.responses import JSONResponse, Response
import subprocess
import sys
import os
import io
import uuid
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Date
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
import models, schemas
from database import engine, get_db, SessionLocal
from ml_engine import nlp_engine, diagnostic_model, CONFIDENCE_THRESHOLD, DEFAULT_SYNONYMS
from diseases_seed import DISEASE_INFO
from rule_engine import apply_expert_rules, CRITICAL_ACTION
from report_generator import generate_diagnosis_pdf

# Config
SECRET_KEY = "my_super_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

models.Base.metadata.create_all(bind=engine)

def log_audit_event(db: Session, event_type: str, description: str, user_id=None):
    entry = models.AuditLog(user_id=user_id, event_type=event_type, description=description)
    db.add(entry)
    db.commit()

def seed_diseases():
    db = SessionLocal()
    try:
        for name, info in DISEASE_INFO.items():
            exists = db.query(models.Disease).filter(models.Disease.disease_name == name).first()
            if not exists:
                db.add(models.Disease(
                    disease_name=name,
                    precautions=info["precautions"],
                    remedies=info["remedies"],
                    is_critical=info["is_critical"],
                ))
        db.commit()
    finally:
        db.close()

def seed_symptom_dictionary():
    # FR-05 / Design Doc SymptomDictionary: one-time seed from the hardcoded
    # defaults, grouped by standard term. Runs only if the table is empty so
    # it never overwrites admin-managed synonym data on later restarts.
    db = SessionLocal()
    try:
        if db.query(models.SymptomDictionary).count() == 0:
            by_target = {}
            for phrase, target in DEFAULT_SYNONYMS.items():
                by_target.setdefault(target, []).append(phrase)
            for target, phrases in by_target.items():
                db.add(models.SymptomDictionary(standard_name=target, synonyms=phrases, severity_weight=1))
            db.commit()
    finally:
        db.close()

seed_diseases()
seed_symptom_dictionary()

app = FastAPI(title="AI Medical Diagnosis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    db = SessionLocal()
    try:
        log_audit_event(db, "system_error", f"{request.method} {request.url.path} -> {type(exc).__name__}: {exc}")
    finally:
        db.close()
    return JSONResponse(status_code=500, content={"detail": "Something went wrong on our end. Please try again in a moment."})

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
    if not user.is_active:
        raise HTTPException(status_code=403, detail="This account has been deactivated. Please contact an administrator.")
    return user

@app.post("/api/auth/register", response_model=schemas.UserResponse)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        log_audit_event(db, "register_failed", f"Registration attempted with an email already in use: {user.email}")
        raise HTTPException(status_code=400, detail="Email already registered")

    # Public self-registration always creates a Patient account (UC-01).
    # Doctor/Admin accounts are provisioned separately by an admin via
    # POST /api/admin/users.
    new_user = models.User(
        full_name=user.full_name,
        email=user.email,
        password_hash=get_password_hash(user.password),
        role=1
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    log_audit_event(db, "register", f"New patient account registered: {user.email}", user_id=new_user.user_id)
    return new_user

@app.post("/api/auth/login", response_model=schemas.Token)
def login_user(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password_hash):
        log_audit_event(db, "login_failed", f"Failed login attempt for: {user.email}", user_id=db_user.user_id if db_user else None)
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    if not db_user.is_active:
        log_audit_event(db, "login_rejected_inactive", f"Login rejected for deactivated account: {user.email}", user_id=db_user.user_id)
        raise HTTPException(status_code=403, detail="This account has been deactivated. Please contact an administrator.")

    access_token = create_access_token(data={"sub": str(db_user.user_id), "role": db_user.role})
    log_audit_event(db, "login_success", f"User logged in: {user.email}", user_id=db_user.user_id)
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

    # FR-09: hybrid rule-based processing — let a matching expert rule
    # (highest priority wins) adjust or override the raw ML prediction.
    active_rules = db.query(models.ExpertRule).order_by(models.ExpertRule.priority.desc()).all()
    mock_prediction, applied_rule = apply_expert_rules(extracted_symptoms, mock_prediction, active_rules)
    rule_bypasses_threshold = bool(applied_rule) and applied_rule["action"] == CRITICAL_ACTION

    # FR-08: don't present a disease as a likely diagnosis below the
    # confidence threshold — ask for more symptom detail instead. A matched
    # "Flag Critical" rule bypasses this: a known dangerous symptom pattern
    # should never be suppressed just because the ML model's confidence
    # alone was low.
    if not rule_bypasses_threshold and (not extracted_symptoms or mock_prediction.get("confidence", 0) < CONFIDENCE_THRESHOLD):
        log_audit_event(
            db, "diagnosis_insufficient_confidence",
            f"Diagnosis request had insufficient confidence ({mock_prediction.get('confidence', 0)}%) with {len(extracted_symptoms)} symptom(s) extracted.",
            user_id=current_user.user_id,
        )
        return {
            "insufficient_confidence": True,
            "message": "We couldn't identify a likely condition with enough confidence from that description. Please add more detail about your symptoms, severity, and duration.",
            "extracted_symptoms": extracted_symptoms,
            "spell_corrections": spell_corrections,
            "appliedRule": applied_rule,
        }

    disease_record = db.query(models.Disease).filter(models.Disease.disease_name == mock_prediction.get("disease")).first()
    recommendation = {
        "precautions": disease_record.precautions if disease_record else "Monitor your symptoms and rest as needed.",
        "remedies": disease_record.remedies if disease_record else "Stay hydrated and consult a doctor if symptoms persist or worsen.",
        "is_critical": disease_record.is_critical if disease_record else False,
    }

    diagnosis = models.DiagnosisHistory(
        user_id=current_user.user_id,
        input_text=req.input_text,
        extracted_symptoms=extracted_symptoms,
        severity=final_severity,
        duration=final_duration,
        predictions=mock_prediction,
        model_name=diagnostic_model.model_name,
        model_version=diagnostic_model.model_version,
        validation_status=0
    )

    db.add(diagnosis)
    db.commit()
    db.refresh(diagnosis)

    log_audit_event(
        db, "diagnosis_request",
        f"Diagnosis generated: {mock_prediction.get('disease')} ({mock_prediction.get('confidence')}%)",
        user_id=current_user.user_id,
    )

    if applied_rule:
        log_audit_event(
            db, "expert_rule_applied",
            f"Rule '{applied_rule['condition']}' ({applied_rule['action']}, priority {applied_rule['priority']}) adjusted diagnosis to: {mock_prediction.get('disease')}",
            user_id=current_user.user_id,
        )

    # Return as dict to include spell corrections
    response = {column.name: getattr(diagnosis, column.name) for column in diagnosis.__table__.columns}
    response["spell_corrections"] = spell_corrections
    response["recommendation"] = recommendation
    response["appliedRule"] = applied_rule

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

@app.get("/api/diagnosis/{diagnosis_id}/report")
def download_diagnosis_report(diagnosis_id: str, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    diagnosis = db.query(models.DiagnosisHistory).filter(models.DiagnosisHistory.diagnosis_id == diagnosis_id).first()
    if not diagnosis:
        raise HTTPException(status_code=404, detail="Diagnosis not found")

    # UC-08: the owning patient may download their own report; doctors/admins
    # may pull it too, consistent with their existing access to all
    # diagnoses via /api/doctor/validations.
    if diagnosis.user_id != current_user.user_id and current_user.role not in (2, 3):
        raise HTTPException(status_code=403, detail="Not authorized to access this report.")

    patient = db.query(models.User).filter(models.User.user_id == diagnosis.user_id).first()
    pdf_bytes = generate_diagnosis_pdf(diagnosis, patient_name=patient.full_name if patient else "")

    log_audit_event(db, "report_downloaded", f"Diagnosis report downloaded for diagnosis {diagnosis_id}", user_id=current_user.user_id)

    filename = f"diagnosis-report-{diagnosis_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

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
def add_admin_synonym(req: schemas.SynonymRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != 3: # 3 = Admin
        raise HTTPException(status_code=403, detail="Not authorized. Admins only.")
    nlp_engine.add_synonym(db, req.phrase, req.target)
    return {"message": f"Synonym '{req.phrase}' added and mapped to '{req.target}' successfully."}

@app.delete("/api/admin/synonyms/{phrase}")
def delete_admin_synonym(phrase: str, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized. Admins only.")

    success = nlp_engine.remove_synonym(db, phrase)
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

REQUIRED_DATASET_COLUMNS = {"disease", "severity", "duration"}

@app.post("/api/admin/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized.")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    content = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="The file could not be read as a valid CSV.")

    missing_columns = REQUIRED_DATASET_COLUMNS - set(df.columns)
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Dataset is missing required column(s): {', '.join(sorted(missing_columns))}.")

    symptom_columns = [c for c in df.columns if c not in REQUIRED_DATASET_COLUMNS]
    if not symptom_columns:
        raise HTTPException(status_code=400, detail="Dataset must include at least one symptom column.")

    if df.isnull().any().any():
        raise HTTPException(status_code=400, detail="Dataset contains missing values. Please fill or remove incomplete rows.")

    if df.duplicated().any():
        raise HTTPException(status_code=400, detail="Dataset contains duplicate rows. Please remove duplicates and try again.")

    with open('symptoms_dataset.csv', 'wb') as f:
        f.write(content)

    log_audit_event(db, "dataset_upload", f"Dataset '{file.filename}' uploaded ({len(df)} rows).", user_id=current_user.user_id)

    return {"message": "Dataset uploaded successfully."}

@app.post("/api/admin/datasets/train")
def train_model(current_user: models.User = Depends(get_current_user)):
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized.")
        
    try:
        # Run training script with the same interpreter running this app
        # (not bare 'python' from PATH, which may not have the venv's deps).
        process = subprocess.run([sys.executable, 'train_model.py'], capture_output=True, text=True, check=True)
        # Reload model into memory
        diagnostic_model.__init__()
        nlp_engine.__init__()
        return {"message": "Model trained and reloaded successfully!", "logs": process.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e.stderr}")

@app.get("/api/admin/models")
def get_models(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized.")
    return db.query(models.ModelMetadata).order_by(models.ModelMetadata.model_name).all()

@app.put("/api/admin/models/{model_id}/activate")
def activate_model(model_id: str, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized.")

    target = db.query(models.ModelMetadata).filter(models.ModelMetadata.model_id == model_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Model not found")

    db.query(models.ModelMetadata).update({models.ModelMetadata.is_active: False})
    target.is_active = True
    db.commit()

    # Reload the live diagnostic model to reflect the newly activated one.
    diagnostic_model.__init__()

    log_audit_event(db, "model_activated", f"Model '{target.model_name}' ({target.model_version}) set as active.", user_id=current_user.user_id)

    return {"message": f"{target.model_name} is now the active model."}

from pydantic import BaseModel
class RoleUpdateRequest(BaseModel):
    role: int

@app.get("/api/admin/users")
def get_all_users(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized.")

    users = db.query(models.User).order_by(models.User.created_at.desc()).all()
    return [{"user_id": u.user_id, "full_name": u.full_name, "email": u.email, "role": u.role, "is_active": u.is_active, "created_at": u.created_at} for u in users]

@app.put("/api/admin/users/{target_user_id}")
def edit_user(target_user_id: str, req: schemas.AdminUserEdit, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized.")

    user = db.query(models.User).filter(models.User.user_id == target_user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    existing = db.query(models.User).filter(models.User.email == req.email, models.User.user_id != target_user_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already in use by another account.")

    user.full_name = req.full_name
    user.email = req.email
    db.commit()

    log_audit_event(db, "user_edited", f"Admin updated account details for {user.email}", user_id=current_user.user_id)
    return {"message": f"User {user.email} updated successfully"}

@app.put("/api/admin/users/{target_user_id}/status")
def set_user_status(target_user_id: str, req: schemas.UserStatusRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized.")

    if str(current_user.user_id) == target_user_id and not req.is_active:
        raise HTTPException(status_code=400, detail="You cannot deactivate your own account.")

    user = db.query(models.User).filter(models.User.user_id == target_user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_active = req.is_active
    db.commit()

    log_audit_event(db, "user_status_changed", f"{'Activated' if req.is_active else 'Deactivated'} account: {user.email}", user_id=current_user.user_id)
    return {"message": f"User {user.email} {'activated' if req.is_active else 'deactivated'} successfully"}

@app.post("/api/admin/users", response_model=schemas.UserResponse)
def create_user_as_admin(req: schemas.AdminUserCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized.")

    if req.role not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Role must be 1 (Patient), 2 (Doctor), or 3 (Admin).")

    existing = db.query(models.User).filter(models.User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = models.User(
        full_name=req.full_name,
        email=req.email,
        password_hash=get_password_hash(req.password),
        role=req.role
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    log_audit_event(db, "admin_user_created", f"Admin created a role={req.role} account: {req.email}", user_id=current_user.user_id)
    return new_user

@app.get("/api/admin/audit-logs", response_model=list[schemas.AuditLogResponse])
def get_audit_logs(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized.")

    return db.query(models.AuditLog).order_by(models.AuditLog.created_at.desc()).limit(200).all()

@app.get("/api/admin/stats")
def get_usage_stats(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # FR-14 / UC-16: system usage information and diagnostic activity statistics.
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized.")

    total_diagnoses = db.query(models.DiagnosisHistory).count()
    total_users = db.query(models.User).count()

    role_labels = {1: "Patient", 2: "Doctor", 3: "Admin"}
    users_by_role = {
        role_labels.get(role, str(role)): count
        for role, count in db.query(models.User.role, func.count(models.User.user_id)).group_by(models.User.role).all()
    }

    status_labels = {0: "Pending", 1: "Validated", 2: "Rejected"}
    validation_by_status = {
        status_labels.get(status, str(status)): count
        for status, count in db.query(models.DiagnosisHistory.validation_status, func.count(models.DiagnosisHistory.diagnosis_id))
            .group_by(models.DiagnosisHistory.validation_status).all()
    }

    top_diseases_query = (
        db.query(
            models.DiagnosisHistory.predictions["disease"].astext.label("disease"),
            func.count(models.DiagnosisHistory.diagnosis_id).label("count"),
        )
        .group_by(models.DiagnosisHistory.predictions["disease"].astext)
        .order_by(func.count(models.DiagnosisHistory.diagnosis_id).desc())
        .limit(10)
        .all()
    )
    top_diseases = [{"disease": disease, "count": count} for disease, count in top_diseases_query]

    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    daily_query = (
        db.query(
            cast(models.DiagnosisHistory.prediction_timestamp, Date).label("day"),
            func.count(models.DiagnosisHistory.diagnosis_id).label("count"),
        )
        .filter(models.DiagnosisHistory.prediction_timestamp >= thirty_days_ago)
        .group_by("day")
        .order_by("day")
        .all()
    )
    diagnoses_per_day = [{"date": day.isoformat(), "count": count} for day, count in daily_query]

    return {
        "total_diagnoses": total_diagnoses,
        "total_users": total_users,
        "users_by_role": users_by_role,
        "validation_by_status": validation_by_status,
        "top_diseases": top_diseases,
        "diagnoses_per_day": diagnoses_per_day,
    }

@app.put("/api/admin/users/{target_user_id}/role")
def update_user_role(target_user_id: str, req: RoleUpdateRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != 3:
        raise HTTPException(status_code=403, detail="Not authorized.")

    if req.role not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Role must be 1 (Patient), 2 (Doctor), or 3 (Admin).")

    user = db.query(models.User).filter(models.User.user_id == target_user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.role = req.role
    db.commit()

    log_audit_event(db, "role_updated", f"Role for {user.email} changed to {req.role}", user_id=current_user.user_id)

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
