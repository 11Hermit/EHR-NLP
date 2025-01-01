from preprocessing.preprocessor import EHRPreprocessor
from models.ehr_classifier import EHRClassifier
from utils.trainer import EHRTrainer

def test_preprocessing():
    # Test the preprocessor
    sample_text = """
    Patient presents with severe headache and fever of 102°F. 
    Prescribed acetaminophen 500mg and advised rest. 
    Previous history of hypertension. 
    Follow-up scheduled in 2 weeks.
    """
    
    print("Testing EHR Processing Pipeline...")
    print("-" * 50)
    
    # Initialize preprocessor
    preprocessor = EHRPreprocessor()
    
    # Test text cleaning
    print("\n1. Clean Text:")
    cleaned_text = preprocessor.clean_text(sample_text)
    print(cleaned_text)
    
    # Test entity extraction
    print("\n2. Medical Entities:")
    entities = preprocessor.extract_medical_entities(sample_text)
    for entity_type, items in entities.items():
        if items:
            print(f"{entity_type}: {items}")
    
    # Test tokenization
    print("\n3. Tokenization:")
    tokens = preprocessor.tokenize(sample_text)
    print(f"Input shape: {tokens['input_ids'].shape}")
    
    return preprocessor, sample_text

def test_model():
    # Test the model
    print("\nTesting EHR Classification Model...")
    print("-" * 50)
    
    model = EHRClassifier(num_labels=2)  # Binary classification example
    print("\nModel initialized successfully")
    
    return model

def main():
    # Run preprocessing tests
    preprocessor, sample_text = test_preprocessing()
    
    # Run model tests
    model = test_model()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main() 