# microservices/model_service/model_definition.py

import torch
from transformers import AutoModelForSequenceClassification

def create_sft_model(model_name="distilbert-base-uncased", num_labels=2):
    """
    Loads a pretrained DistilBERT for sequence classification.
    Returns the model ready for fine-tuning
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels = num_labels
    )

    return model