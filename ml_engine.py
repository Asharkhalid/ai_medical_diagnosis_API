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

class MedicalNLPEngine:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Medical Ontology / Synonym Mapping
        self.synonym_map = {
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
            "breathing issue": "shortness_of_breath"
        }
        
        # Try loading feature columns dynamically from model
        try:
            self.supported_features = joblib.load('models/feature_columns.pkl')
            # Extract just symptoms (removing 'severity' and 'duration')
            self.supported_symptoms = [f for f in self.supported_features if f not in ['severity', 'duration']]
        except:
            # Fallback if model not trained yet
            self.supported_symptoms = [
                "fever", "headache", "cough", "fatigue", "stomach_pain",
                "nausea", "vomiting", "sore_throat", "shortness_of_breath",
                "body_pain", "chills", "loss_of_taste", "loss_of_smell",
                "diarrhea", "chest_pain", "wheezing", "dizziness",
                "weight_loss", "sweating", "pale_skin", "frequent_urination"
            ]

    def add_synonym(self, phrase, target):
        self.synonym_map[phrase.lower()] = target
        return True

    def remove_synonym(self, phrase):
        if phrase.lower() in self.synonym_map:
            del self.synonym_map[phrase.lower()]
            return True
        return False

    def clean_text(self, text):
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def extract_severity(self, text):
        text = text.lower()
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
        try:
            self.model = joblib.load('models/random_forest_v1.pkl')
            self.feature_columns = joblib.load('models/feature_columns.pkl')
            self.model_loaded = True
        except:
            self.model_loaded = False
        
    def predict(self, feature_vector, extracted_symptoms):
        """
        ML prediction and basic Explainable AI weights
        """
        if not self.model_loaded or feature_vector is None:
            return {
                "disease": "Model Not Loaded (Run Training)",
                "confidence": 0,
                "influentialFactors": []
            }
            
        prediction = self.model.predict(feature_vector)[0]
        
        # Predict probability
        probas = self.model.predict_proba(feature_vector)[0]
        confidence = round(max(probas) * 100, 2)
        
        # Explainable AI (XAI) - Extract feature importances
        importances = self.model.feature_importances_
        
        # Match importances with symptoms present in the user
        influential = []
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
            "influentialFactors": influential
        }

# Singleton instances
nlp_engine = MedicalNLPEngine()
diagnostic_model = DiagnosticModel()
