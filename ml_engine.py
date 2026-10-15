import re
import nltk
import difflib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import joblib
import os

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Minimum prediction confidence (%) required before a disease is presented
# as a likely diagnosis (FR-08) — below this, the caller should ask for
# more symptom detail instead of showing a name and confidence score.
CONFIDENCE_THRESHOLD = 40

# FR-05 / Design Doc SymptomDictionary: default phrase->standard-term
# mapping used to seed the database the first time the app runs (see
# main.py's seed_symptom_dictionary()). Once seeded, the DB is the source
# of truth — this dict is only a bootstrap fallback for a database that
# isn't reachable yet (e.g. before create_all runs on a brand new DB).
DEFAULT_SYNONYMS = {
    "temperature": "fever",
    "stomach pain": "abdominal_pain",
    "belly ache": "abdominal_pain",
    "stomach_pain": "abdominal_pain",
    "head ache": "headache",
    "migraine": "headache",
    "throw up": "nausea",
    "vomit": "nausea",
    "tired": "fatigue",
    "exhausted": "fatigue",
    "breathing issue": "shortness_of_breath",
    "spots": "rash",
    "skin rash": "rash",
    "hives": "rash",
    "yellowing": "jaundice",
    "yellow skin": "jaundice",
    "yellow eyes": "jaundice",
    "joint ache": "joint_pain",
    "achy joints": "joint_pain",
    "stuffy nose": "runny_nose",
    "blocked nose": "runny_nose",
    "lower back pain": "back_pain",
    "backache": "back_pain",
    "loin pain": "back_pain",
    "skin itch": "itching",
    "itchy skin": "itching",
    "vision problem": "blurred_vision",
    "blurry vision": "blurred_vision",
    "swollen glands": "swollen_lymph_nodes",
    "lymph nodes": "swollen_lymph_nodes",
    "no appetite": "loss_of_appetite",
    "not hungry": "loss_of_appetite",
    "dark yellow urine": "dark_urine",
    "brown urine": "dark_urine",
    "painful urination": "burning_urination",
    "pain when urinating": "burning_urination",
    "bloated": "bloating",
    "gas": "bloating",
    "heartburn": "acid_reflux",
    "acidity": "acid_reflux",
    "cramps": "muscle_cramps",
    "muscle ache": "muscle_cramps",
    "heart racing": "palpitations",
    "rapid heartbeat": "palpitations",
}

class MedicalNLPEngine:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # FR-05: symptom ontology/dictionary, persisted in SymptomDictionary
        # (one row per standard term, holding all phrases that map to it).
        # Falls back to the hardcoded defaults if the DB/table isn't ready
        # yet (e.g. the very first import, before main.py's create_all runs).
        self.synonym_map = self._load_synonym_map()

        # Try loading feature columns dynamically from model
        try:
            self.supported_features = joblib.load('models/feature_columns.pkl')
            # Extract just symptoms (removing 'severity' and 'duration')
            self.supported_symptoms = [f for f in self.supported_features if f not in ['severity', 'duration']]
        except:
            # Fallback if model not trained yet
            self.supported_symptoms = [
                "fever", "headache", "cough", "fatigue", "abdominal_pain",
                "nausea", "vomiting", "sore_throat", "shortness_of_breath",
                "body_pain", "chills", "loss_of_taste", "loss_of_smell",
                "diarrhea", "chest_pain", "wheezing", "dizziness",
                "weight_loss", "sweating", "pale_skin", "frequent_urination",
                "rash", "jaundice", "joint_pain", "runny_nose", "back_pain",
                "itching", "blurred_vision", "swollen_lymph_nodes",
                "loss_of_appetite", "dark_urine", "burning_urination",
                "bloating", "acid_reflux", "muscle_cramps", "palpitations"
            ]

        # Multi-word feature names (e.g. "body_pain") only get matched via
        # extract_symptoms' tokenizer when the user types the exact
        # underscored form. Alias the plain-English phrasing ("body pain")
        # to each one so natural descriptions aren't silently dropped.
        for feature in self.supported_symptoms:
            if '_' in feature:
                self.synonym_map.setdefault(feature.replace('_', ' '), feature)

    def _load_synonym_map(self):
        try:
            import models
            from database import SessionLocal
            db = SessionLocal()
            try:
                rows = db.query(models.SymptomDictionary).all()
                if not rows:
                    return dict(DEFAULT_SYNONYMS)
                flat_map = {}
                for row in rows:
                    for phrase in (row.synonyms or []):
                        flat_map[phrase.lower()] = row.standard_name
                return flat_map
            finally:
                db.close()
        except Exception:
            # Table/DB not ready yet (e.g. first import, before create_all
            # has run) — fall back to the hardcoded defaults for this boot.
            return dict(DEFAULT_SYNONYMS)

    def add_synonym(self, db, phrase, target):
        import models
        phrase = phrase.lower()

        # A phrase should map to exactly one standard term — drop it from
        # wherever it currently lives before re-adding it under `target`.
        existing_rows = db.query(models.SymptomDictionary).filter(models.SymptomDictionary.synonyms.contains([phrase])).all()
        for row in existing_rows:
            row.synonyms = [s for s in row.synonyms if s != phrase]

        target_row = db.query(models.SymptomDictionary).filter(models.SymptomDictionary.standard_name == target).first()
        if target_row:
            if phrase not in target_row.synonyms:
                target_row.synonyms = target_row.synonyms + [phrase]
        else:
            db.add(models.SymptomDictionary(standard_name=target, synonyms=[phrase], severity_weight=1))

        db.commit()
        self.synonym_map[phrase] = target
        return True

    def remove_synonym(self, db, phrase):
        import models
        phrase = phrase.lower()

        if phrase not in self.synonym_map:
            return False

        rows = db.query(models.SymptomDictionary).filter(models.SymptomDictionary.synonyms.contains([phrase])).all()
        for row in rows:
            row.synonyms = [s for s in row.synonyms if s != phrase]
        db.commit()

        del self.synonym_map[phrase]
        return True

    def clean_text(self, text):
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def extract_severity(self, text):
        text = text.lower()
        # "high/low fever" and "high/low temperature" describe the fever
        # reading itself, not how severe the overall episode is — strip
        # those phrases first so they don't silently override the severity
        # the user chose in the UI.
        text = re.sub(r'\b(high|low)\s+(fever|temperature)\b', r'\2', text)
        if any(word in text for word in ["severe", "high", "extreme", "bad", "terrible", "intense"]):
            return 3
        if any(word in text for word in ["mild", "low", "slight", "minor"]):
            return 1
        return None
        
    def extract_duration(self, text):
        text = text.lower()
        if "week" in text or "month" in text:
            return 3
        if "day" in text:
            return 2
        if "hour" in text or "minute" in text:
            return 1
        return None

    def extract_symptoms(self, text):
        # Apply 2-gram synonyms first exactly
        text_lower = text.lower()
        for syn, target in self.synonym_map.items():
            text_lower = text_lower.replace(syn, target)
            
        cleaned = self.clean_text(text_lower)
        
        # Tokenization & Stop-word removal
        tokens = word_tokenize(cleaned)
        filtered_tokens = [w for w in tokens if w not in self.stop_words]
        
        # Lemmatization
        lemmatized = [self.lemmatizer.lemmatize(w) for w in filtered_tokens]
        
        extracted = []
        corrections = {}
        
        for word in lemmatized:
            if word in self.supported_symptoms:
                extracted.append(word)
            else:
                # Fuzzy matching / Typo Tolerance against supported symptoms
                matches = difflib.get_close_matches(word, self.supported_symptoms, n=1, cutoff=0.75)
                if matches:
                    extracted.append(matches[0])
                    corrections[word] = matches[0]
                else:
                    # Fuzzy match against synonym map keys
                    syn_matches = difflib.get_close_matches(word, list(self.synonym_map.keys()), n=1, cutoff=0.8)
                    if syn_matches:
                        target = self.synonym_map[syn_matches[0]]
                        extracted.append(target)
                        corrections[word] = target

        return list(set(extracted)), corrections

    def vectorize_features(self, extracted_symptoms, severity=1, duration=1):
        """
        severity: 1=Low, 2=Medium, 3=High
        duration: 1=Hours, 2=Days, 3=Weeks
        """
        # Create dictionary of features
        feature_dict = {sym: 0 for sym in self.supported_symptoms}
        for sym in extracted_symptoms:
            if sym in feature_dict:
                feature_dict[sym] = 1 # Binary encoding (Present)
                
        feature_dict['severity'] = severity
        feature_dict['duration'] = duration
        
        # Convert to ordered list based on feature_columns
        try:
            ordered_features = joblib.load('models/feature_columns.pkl')
            vector = [feature_dict.get(f, 0) for f in ordered_features]
            return np.array([vector])
        except:
            # Fallback
            return None

class DiagnosticModel:
    def __init__(self):
        active = self._get_active_metadata()
        try:
            if active:
                self.model = joblib.load(f'models/{active.algorithm}_v1.pkl')
                self.model_name = active.model_name
                self.model_version = active.model_version
            else:
                # FR-07 backward-compat fallback: no ModelMetadata row yet
                # (e.g. training has never been run via the new pipeline).
                self.model = joblib.load('models/random_forest_v1.pkl')
                self.model_name = "RandomForestClassifier"
                self.model_version = "v1.2"
            self.feature_columns = joblib.load('models/feature_columns.pkl')
            self.model_loaded = True
        except Exception:
            self.model_loaded = False

    def _get_active_metadata(self):
        try:
            import models
            from database import SessionLocal
            db = SessionLocal()
            try:
                return db.query(models.ModelMetadata).filter(models.ModelMetadata.is_active.is_(True)).first()
            finally:
                db.close()
        except Exception:
            # Table/DB not ready yet (e.g. first import, before create_all
            # has run) — fall back to the Random Forest default for this boot.
            return None

    def predict(self, feature_vector, extracted_symptoms):
        """
        ML prediction and basic Explainable AI weights
        """
        if not self.model_loaded or feature_vector is None:
            return {
                "disease": "Model Not Loaded (Run Training)",
                "confidence": 0,
                "influentialFactors": [],
                "rankedPredictions": [],
                "explanationText": ""
            }

        prediction = self.model.predict(feature_vector)[0]

        # Predict probability
        probas = self.model.predict_proba(feature_vector)[0]
        confidence = round(max(probas) * 100, 2)

        # FR-08/FR-15: top-3 ranked diseases with their own probabilities,
        # alongside the top-1 disease/confidence fields kept for the
        # existing UI and PDF report.
        top_indices = np.argsort(probas)[::-1][:3]
        ranked_predictions = [
            {"disease": self.model.classes_[i], "confidence": round(probas[i] * 100, 2)}
            for i in top_indices
        ]

        # Explainable AI (XAI) - Extract feature importances. Not every
        # supported algorithm exposes this (e.g. Naive Bayes has no
        # feature_importances_) — explanation is simply omitted in that case
        # rather than fabricated.
        importances = getattr(self.model, "feature_importances_", None)

        # Match importances with symptoms present in the user
        influential = []
        if importances is not None:
            for sym in extracted_symptoms:
                if sym in self.feature_columns:
                    idx = self.feature_columns.index(sym)
                    # Scale up small importance for UI visualization
                    impact_score = round(min(importances[idx] * 1000, 95), 1)
                    influential.append({"name": sym, "impact": impact_score})

        # Sort by most influential
        influential = sorted(influential, key=lambda x: x['impact'], reverse=True)

        return {
            "disease": prediction,
            "confidence": confidence,
            "influentialFactors": influential,
            "rankedPredictions": ranked_predictions,
            "explanationText": self._build_explanation_text(prediction, confidence, influential)
        }

    @staticmethod
    def _impact_bucket(score):
        if score >= 60:
            return "high"
        if score >= 30:
            return "medium"
        return "low"

    def _build_explanation_text(self, disease, confidence, influential):
        # FR-10 / NFR-11: a human-readable explanation, not just numeric
        # bars, matching the Proposal's own example format ("Prediction:
        # Flu (82%) Reason: - Fever -> high importance ...").
        if not influential:
            return f"Prediction: {disease} ({confidence}%). No individual symptom-level explanation is available for the current model."

        reasons = "; ".join(
            f"{f['name'].replace('_', ' ')} → {self._impact_bucket(f['impact'])} importance"
            for f in influential
        )
        return f"Prediction: {disease} ({confidence}%). Reasoning: {reasons}."

# Singleton instances
nlp_engine = MedicalNLPEngine()
diagnostic_model = DiagnosticModel()
