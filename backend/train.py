import argparse
import logging
import os
import numpy as np
import torch
import sys
import subprocess

# Install packages at runtime if needed in the training environment
print("Installing required packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "transformers", "rouge-score"])

from datasets import load_dataset
try:
    from datasets import load_metric
except ImportError:
    # For older versions of datasets
    from datasets import load_metric as load_metrics
    def load_metric(name):
        return load_metrics(name)

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Data and model parameters
    parser.add_argument("--model-name", type=str, default="facebook/bart-base")
    parser.add_argument("--max-input-length", type=int, default=512)  # Reduced for CPU training
    parser.add_argument("--max-target-length", type=int, default=64)   # Reduced for CPU training
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=1)  # Reduced for CPU training
    parser.add_argument("--batch-size", type=int, default=4)  # Reduced for CPU training
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)  # Reduced for CPU training
    
    # SageMaker parameters
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/tmp/output"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/tmp/model"))
    parser.add_argument("--n-gpus", type=int, default=0)  # Default to CPU training
    
    return parser.parse_args()

def preprocess_function(examples, tokenizer, max_input_length, max_target_length):
    inputs = [doc for doc in examples["article"]]
    targets = [summary for summary in examples["highlights"]]
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred, tokenizer, metric):
    predictions, labels = eval_pred
    
    # Replace -100 with the pad token id
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Convert ids to tokens
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(pred.split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.split()) for label in decoded_labels]
    
    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, 
                           use_stemmer=True)
    
    # Extract scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return result

def train(args):
    # Load the CNN/DailyMail dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    
    # Preprocess the dataset
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(
            examples, tokenizer, args.max_input_length, args.max_target_length
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # Load ROUGE metric
    rouge_metric = load_metric("rouge")
    
    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        logging_dir=f"{args.output_data_dir}/logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        gradient_accumulation_steps=2,
        fp16=(torch.cuda.is_available() and args.n_gpus > 0),
    )
    
    # Initialize the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer, rouge_metric),
    )
    
    # Train the model
    logger.info("*** Training the model ***")
    trainer.train()
    
    # Save the model
    logger.info("*** Saving the model ***")
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    
    logger.info("*** Training completed ***")

if __name__ == "__main__":
    args = parse_args()
    train(args)
