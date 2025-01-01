import torch
import transformers
import pandas as pd
import numpy as np
import sklearn
import spacy
import nltk
import matplotlib
import seaborn

def verify_installations():
    """Verify that all required packages are installed correctly"""
    packages = {
        'PyTorch': torch.__version__,
        'Transformers': transformers.__version__,
        'Pandas': pd.__version__,
        'NumPy': np.__version__,
        'Scikit-learn': sklearn.__version__,
        'spaCy': spacy.__version__,
        'NLTK': nltk.__version__,
        'Matplotlib': matplotlib.__version__,
        'Seaborn': seaborn.__version__
    }
    
    print("Package versions:")
    for package, version in packages.items():
        print(f"{package}: {version}")
    
    # Verify CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Verify spaCy model
    try:
        nlp = spacy.load("en_core_sci_md")
        print("\nspaCy medical model loaded successfully")
    except Exception as e:
        print(f"\nError loading spaCy model: {e}")

if __name__ == "__main__":
    verify_installations() 