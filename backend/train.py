import argparse
import logging
import os
import numpy as np
import torch
import sys
import subprocess
import json
import psutil
import gc

# Install packages at runtime if needed in the training environment
print("Installing required packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "transformers", "rouge-score", "psutil", "evaluate"])

from datasets import load_dataset
from evaluate import load as load_metric

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Data and model parameters
    parser.add_argument("--model-name", type=str, default="facebook/bart-base")
    parser.add_argument("--max-input-length", type=int, default=128)  # Reduced for CPU training
    parser.add_argument("--max-target-length", type=int, default=32)   # Reduced for CPU training
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)  # Minimal batch size
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)  # Reduced for CPU training
    
    # SageMaker parameters
    parser.add_argument("--output-data-dir", type=str, 
                        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/tmp/output"))
    parser.add_argument("--model-dir", type=str, 
                        default=os.environ.get("SM_MODEL_DIR", "/tmp/model"))
    parser.add_argument("--train-dir", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--validation-dir", type=str,
                        default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    
    # Dataset size parameters
    parser.add_argument("--dataset-size", type=float, default=0.01)  # Use only 1% of the data
    parser.add_argument("--max-train-samples", type=int, default=50)  # Cap at 50 samples max
    parser.add_argument("--max-val-samples", type=int, default=10)    # Cap at 10 samples max
    
    # Training control parameters
    parser.add_argument("--max-steps", type=int, default=10)  # Very small number of steps
    parser.add_argument("--use-max-steps", type=bool, default=True)
    
    args, _ = parser.parse_known_args()
    return args

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

def monitor_memory(message=""):
    """Print memory usage information"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Convert to MB for readability
    rss_mb = memory_info.rss / (1024 * 1024)
    vms_mb = memory_info.vms / (1024 * 1024)
    
    # Get disk usage
    disk_usage = psutil.disk_usage('/')
    free_disk_gb = disk_usage.free / (1024 * 1024 * 1024)
    
    # Add a forced garbage collection with each memory check
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    logger.info(f"{message} - Memory usage: RSS {rss_mb:.2f} MB, VMS {vms_mb:.2f} MB, Free disk: {free_disk_gb:.2f} GB")

class MemoryEfficientCallback(TrainerCallback):
    def __init__(self):
        self.step_count = 0
        
    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        if self.step_count % 5 == 0:  # Check memory every 5 steps
            monitor_memory(f"Step {self.step_count}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def train(args):
    try:
        monitor_memory("Starting training")
        
        # Load dataset from CSV files
        train_file = os.path.join(args.train_dir, "train.csv")
        val_file = os.path.join(args.validation_dir, "validation.csv")
        
        logger.info(f"Loading datasets from: {train_file} and {val_file}")
        
        dataset = {
            "train": load_dataset("csv", data_files=train_file, split="train"),
            "validation": load_dataset("csv", data_files=val_file, split="train")
        }
        
        # Limit dataset size
        dataset["train"] = dataset["train"].select(range(min(len(dataset["train"]), args.max_train_samples)))
        dataset["validation"] = dataset["validation"].select(range(min(len(dataset["validation"]), args.max_val_samples)))
        
        monitor_memory("After dataset loading")
        
        # Load model and tokenizer
        logger.info(f"Loading model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        
        # Enable memory optimizations
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        
        # Convert to half precision if possible
        try:
            model = model.half()
            logger.info("Model converted to half precision")
        except Exception as e:
            logger.warning(f"Could not convert to half precision: {e}")
        
        monitor_memory("After model loading")
        
        # Preprocess datasets
        logger.info("Preprocessing datasets")
        tokenized_datasets = {}
        for split in dataset:
            tokenized_datasets[split] = dataset[split].map(
                lambda examples: preprocess_function(
                    examples, tokenizer, args.max_input_length, args.max_target_length
                ),
                batched=True,
                batch_size=1,  # Minimal batch size
                remove_columns=dataset[split].column_names,
            )
        
        # Clean up memory
        del dataset
        gc.collect()
        
        monitor_memory("After preprocessing")
        
        # Set up training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.model_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=1,
            predict_with_generate=False,
            generation_max_length=args.max_target_length,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps if args.use_max_steps else -1,
            warmup_steps=args.warmup_steps,
            evaluation_strategy="steps",
            eval_steps=5,
            save_strategy="steps",
            save_steps=5,
            save_total_limit=2,  # Keep only 2 checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
            greater_is_better=True,
            fp16=False,  # Disable mixed precision training
            gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
            gradient_checkpointing=True,
            optim="adamw_torch",  # Use PyTorch's AdamW implementation
            logging_steps=1,
            logging_dir=os.path.join(args.model_dir, "logs"),
            report_to="none",  # Disable wandb/tensorboard
        )
        
        # Initialize data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        
        # Load ROUGE metric
        metric = load_metric("rouge")
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer, metric),
            callbacks=[MemoryEfficientCallback()]
        )
        
        # Train the model
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        logger.info("Saving model...")
        trainer.save_model(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
        
        # Save training arguments
        with open(os.path.join(args.model_dir, "training_args.json"), "w") as f:
            json.dump(training_args.to_dict(), f, indent=2)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    args = parse_args()
    train(args)
