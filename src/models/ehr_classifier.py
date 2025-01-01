import torch
import torch.nn as nn
from transformers import AutoModel

class EHRClassifier(nn.Module):
    def __init__(
        self, 
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        num_labels: int = 2
    ):
        """
        Initialize the EHR classifier
        
        Args:
            model_name: Name of the pretrained model
            num_labels: Number of classification labels
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        """Forward pass"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits 