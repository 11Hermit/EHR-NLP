# Natural Language Processing for Electronic Health Records (EHRs)

This project implements NLP techniques to analyze and extract information from Electronic Health Records (EHRs). It uses transformer-based models and spaCy for medical text processing, entity recognition, and classification.

## Features

- Medical text preprocessing and cleaning
- Medical entity extraction (diseases, drugs, symptoms, procedures)
- Text classification using BioClinicalBERT
- Modular and extensible architecture

## Project Structure

```
ehr_nlp/
├── data/
│   ├── raw/                # Raw EHR data
│   └── processed/          # Preprocessed data
├── src/
│   ├── preprocessing/      # Data preprocessing scripts
│   ├── models/            # Model implementations
│   ├── evaluation/        # Evaluation metrics
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for analysis
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/11Hermit/EHR-NLP.git
cd EHR-NLP
```

2. Create and activate a virtual environment:

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

For Linux/Mac:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. To run the main application:
```bash
python -m src.main
```

2. To run tests and verify the setup:
```bash
python -m src.test_ehr
```

3. To verify all installations:
```bash
python -m src.utils.verify_setup
```

## Model Architecture

The project uses a transformer-based architecture with BioClinicalBERT for processing medical text. The main components are:

- **Preprocessor**: Handles text cleaning, tokenization, and medical entity extraction
- **Classifier**: Implements the neural network for text classification
- **Trainer**: Manages model training and evaluation

## Example Usage

```python
from src.preprocessing.preprocessor import EHRPreprocessor
from src.models.ehr_classifier import EHRClassifier

# Initialize preprocessor
preprocessor = EHRPreprocessor()

# Process medical text
text = """
Patient presents with severe headache and fever of 102°F. 
Prescribed acetaminophen 500mg and advised rest.
"""

# Clean and extract entities
cleaned_text = preprocessor.clean_text(text)
entities = preprocessor.extract_medical_entities(text)

# Initialize and use classifier
model = EHRClassifier(num_labels=2)
# ... train and use the model
```

## Data

- The project expects EHR data in text format
- Sample data is not included due to privacy concerns
- Users should provide their own EHR data in the `data/raw` directory

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BioClinicalBERT team for the pretrained medical language model
- spaCy for NLP tools and models
- The medical NLP community for research and insights

## Contact

Your Name - [@your_twitter](https://twitter.com/your_twitter) - email@example.com

Project Link: [https://github.com/11Hermit/EHR-NLP](https://github.com/11Hermit/EHR-NLP)

## Troubleshooting

### Common Issues

1. **Installation Problems**
   - Make sure you're using Python 3.8 or higher
   - Try installing packages one by one if batch installation fails
   - Check your system's CUDA version if using GPU

2. **Model Loading Issues**
   - Ensure you have enough RAM/GPU memory
   - Check internet connection for downloading pretrained models
   - Verify all dependencies are correctly installed

3. **Data Processing Errors**
   - Verify input data format
   - Check file permissions
   - Ensure sufficient disk space

For more issues, please check the [Issues](https://github.com/11Hermit/EHR-NLP/issues) page.
```

This README.md provides:
- Clear installation instructions
- Project structure overview
- Usage examples
- Troubleshooting guide
- Contributing guidelines
- License information
- Contact details