import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer once at container startup for efficiency
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# If GPU is available, use it (will speed up inference)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Maximum length of generated summaries
MAX_LENGTH = 150
MIN_LENGTH = 30

def model_fn(model_dir):
    """
    Load the model for inference
    """
    # Model already loaded globally to save time
    return model

def input_fn(request_body, request_content_type):
    """
    Parse input request body
    """
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Generate text summary using the pre-trained model
    """
    text = input_data.get('text', '')
    
    # Set parameters for generation
    params = {
        'max_length': MAX_LENGTH,
        'min_length': MIN_LENGTH,
        'no_repeat_ngram_size': 3,
        'early_stopping': True,
        'num_beams': 4
    }
    
    # Add optional parameters if provided
    if 'max_length' in input_data:
        params['max_length'] = min(int(input_data['max_length']), 500)  # Limit to 500 tokens
    if 'min_length' in input_data:
        params['min_length'] = int(input_data['min_length'])
    
    # Tokenize input text
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate summary
    summary_ids = model.generate(**inputs, **params)
    
    # Decode summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

def output_fn(prediction, response_content_type):
    """
    Format prediction as JSON response
    """
    if response_content_type == 'application/json':
        return json.dumps({'summary': prediction})
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
