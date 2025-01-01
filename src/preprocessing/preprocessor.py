import re
from typing import List, Dict
import spacy
from transformers import AutoTokenizer

class EHRPreprocessor:
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", spacy_model: str = "en_core_web_sm"):
        """
        Initialize the EHR preprocessor
        
        Args:
            model_name: Name of the pretrained model to use for tokenization
            spacy_model: Name of the spaCy model to use (default: en_core_web_sm)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading spaCy model {spacy_model}...")
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)
        except Exception as e:
            print(f"Error loading spaCy model: {e}")
            print("Falling back to basic tokenization...")
            self.nlp = None
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.lower()
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities using spaCy"""
        if self.nlp is None:
            return {
                'DISEASE': [],
                'DRUG': [],
                'SYMPTOM': [],
                'PROCEDURE': []
            }
        
        doc = self.nlp(text)
        entities = {
            'DISEASE': [],
            'DRUG': [],
            'SYMPTOM': [],
            'PROCEDURE': []
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
                
        return entities
    
    def tokenize(self, text: str) -> Dict:
        """Tokenize text using the pretrained tokenizer"""
        return self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ) 