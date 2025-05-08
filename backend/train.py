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
subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "transformers", "rouge-score", "psutil"])

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
    TrainerCallback
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)  # Disable automatic error on unknown args
    
    # Data and model parameters
    parser.add_argument("--model-name", type=str, default="facebook/bart-base")
    parser.add_argument("--max-input-length", type=int, default=256)  # Reduced further for CPU training
    parser.add_argument("--max-target-length", type=int, default=32)   # Reduced further for CPU training
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=1)  # Reduced for CPU training
    parser.add_argument("--batch-size", type=int, default=2)  # Reduced batch size further
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)  # Reduced for CPU training
    
    # SageMaker parameters - support different environment variable names
    parser.add_argument("--output-data-dir", type=str, 
                        default=os.environ.get("SM_OUTPUT_DATA_DIR", 
                                              os.environ.get("OUTPUT_DATA_DIR", "/tmp/output")))
    parser.add_argument("--model-dir", type=str, 
                        default=os.environ.get("SM_MODEL_DIR", 
                                              os.environ.get("MODEL_DIR", "/tmp/model")))
    parser.add_argument("--train-dir", type=str,  # Fixed type.str to type=str
                        default=os.environ.get("SM_CHANNEL_TRAIN",
                                              os.environ.get("TRAIN_DIR", "/opt/ml/input/data/train")))
    parser.add_argument("--validation-dir", type=str,  # Fixed type.str to type=str
                        default=os.environ.get("SM_CHANNEL_VALIDATION",  # Fixed default.os to default=os
                                              os.environ.get("VALIDATION_DIR", "/opt/ml/input/data/validation")))
    
    # Add smaller dataset size option
    parser.add_argument("--dataset-size", type=float, default=0.005)  # Use only 0.5% of the data
    parser.add_argument("--max-train-samples", type=int, default=100)  # Cap at 100 samples max
    parser.add_argument("--max-val-samples", type=int, default=20)     # Cap at 20 samples max
    
    # Parse known arguments only - ignore any Jupyter/IPython specific arguments
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

def train(args):
    try:
        # Monitor memory at the start
        monitor_memory("Starting training")
        
        try:
            # Try to load from CSV files first (provided by SageMaker channels)
            train_file = os.path.join(args.train_dir, "train.csv")
            val_file = os.path.join(args.validation_dir, "validation.csv")
            
            logger.info(f"Looking for training data at: {train_file}")
            logger.info(f"Looking for validation data at: {val_file}")
            
            if os.path.exists(train_file) and os.path.exists(val_file):
                logger.info("Loading datasets from CSV files")
                dataset = {
                    "train": load_dataset("csv", data_files=train_file, split="train"),
                    "validation": load_dataset("csv", data_files=val_file, split="train")
                }
                logger.info("Successfully loaded datasets from CSV")
                logger.info(f"Train set size: {len(dataset['train'])}")
                logger.info(f"Validation set size: {len(dataset['validation'])}")
            else:
                logger.info("Could not find CSV files, downloading dataset directly")
                # Fall back to downloading the dataset
                dataset = load_dataset("cnn_dailymail", "3.0.0")
                
                # Use a very small subset for CPU training
                train_size = min(int(len(dataset['train']) * args.dataset_size), args.max_train_samples)
                val_size = min(int(len(dataset['validation']) * args.dataset_size), args.max_val_samples)
                
                logger.info(f"Using reduced dataset: {train_size} train samples, {val_size} validation samples")
                dataset = {
                    "train": dataset["train"].select(range(train_size)),
                    "validation": dataset["validation"].select(range(val_size))
                }
        except Exception as e:
            # If loading fails, create a tiny dataset for testing
            logger.error(f"Error loading dataset: {e}")
            logger.info("Creating a minimal test dataset")
            
            # Create a very small sample dataset
            small_articles = ["This is a short test article. It needs to be summarized."] * 10
            small_summaries = ["Short summary."] * 10
            
            from datasets import Dataset
            dataset = {
                "train": Dataset.from_dict({"article": small_articles, "highlights": small_summaries}),
                "validation": Dataset.from_dict({"article": small_articles[:2], "highlights": small_summaries[:2]})
            }
        
        # Monitor memory after dataset loading
        monitor_memory("After dataset loading")
        
        logger.info("Loading tokenizer and model")
        # Try to load a smaller model if specified model is too large
        try:
            # Load the pre-trained model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
            
            # Enable gradient checkpointing to save memory
            model.gradient_checkpointing_enable()
            model.config.use_cache = False  # Disable KV cache to save memory
        except Exception as e:
            logger.error(f"Error loading model {args.model_name}: {e}")
            logger.info("Falling back to t5-small model")
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            
        # Try to reduce model size by half-precision
        try:
            import torch.nn as nn
            logger.info("Converting model to half precision")
            model = model.half()  # Convert to half precision
        except Exception as e:
            logger.warning(f"Could not convert to half precision: {e}")
        
        # Monitor memory after model loading
        monitor_memory("After model loading")
        
        # Preprocess the dataset with very small batches
        logger.info("Preprocessing datasets (small batches)")
        tokenized_datasets = {}
        for split in dataset:
            tokenized_datasets[split] = dataset[split].map(
                lambda examples: preprocess_function(
                    examples, tokenizer, args.max_input_length, args.max_target_length
                ),
                batched=True,
                batch_size=2,  # Very small batch size during preprocessing
                remove_columns=dataset[split].column_names,
            )
        
        # Force garbage collection to free memory
        del dataset
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Monitor memory after preprocessing
        monitor_memory("After preprocessing")
        
        # Set up training arguments with extreme memory optimizations
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.model_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=1,   # Reduced eval batch size to absolute minimum
            predict_with_generate=False,    # Don't use generate during evaluation to save memory
            generation_max_length=args.max_target_length,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            warmup_steps=args.warmup_steps,
            logging_dir=f"{args.output_data_dir}/logs",
            logging_steps=1,  # Log every step
            eval_strategy="no",  # Disable auto evaluation to save memory
            save_strategy="steps",
            save_steps=5,  # Save every 5 steps
            save_total_limit=1,  # Keep only 1 checkpoint to save disk space
            load_best_model_at_end=False,  # Don't try to load best model at end
            gradient_accumulation_steps=8,  # Increased for smaller effective batch size
            fp16=False,  # Disable fp16 for CPU training
            dataloader_num_workers=0,  # Disable multiprocessing
            optim="adamw_torch",  # Use memory-efficient optimizer
            report_to="none",  # Disable wandb or other reporting 
            # Safe dispatch to avoid OOM
            group_by_length=True,  # Group similar length sequences
            length_column_name="length",
            remove_unused_columns=True,
        )
        
        logger.info("Setting up training")
        # Data collator for dynamic padding 
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True,
            return_tensors="pt"
        )
        
        # Custom memory-efficient evaluation function
        def compute_memory_efficient_metrics(eval_pred, tokenizer, metric):
            try:
                # Take only first 10 examples to save memory during evaluation
                predictions = eval_pred.predictions[:10]
                labels = eval_pred.label_ids[:10]
                
                # Just return basic scores to avoid memory issues
                return {"rouge1": 0.0}
            except Exception as e:
                logger.error(f"Error in metrics computation: {e}")
                return {"rouge1": 0.0}
        
        # Create memory monitoring callback
        class SaveMemoryCallback(TrainerCallback):
            def __init__(self, save_path):
                self.save_path = save_path
                
            def on_step_end(self, args, state, control, **kwargs):
                # Monitor memory every step
                if state.global_step % 1 == 0:
                    monitor_memory(f"Training step {state.global_step}")
                    
                # Force garbage collection every step
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Save checkpoint more frequently
                if state.global_step % 5 == 0:
                    # Save a checkpoint
                    output_dir = os.path.join(self.save_path, f"checkpoint-{state.global_step}")
                    kwargs['model'].save_pretrained(output_dir)
                    kwargs['tokenizer'].save_pretrained(output_dir)
                    logger.info(f"Saved checkpoint to {output_dir}")
                
                return control
        
        # Initialize the trainer with minimal evaluation
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=None,  # Don't provide eval dataset to save memory
            tokenizer=tokenizer,
            data_collator=data_collator,
            # No compute_metrics to save memory
        )
        
        # Add memory monitoring callback
        trainer.add_callback(SaveMemoryCallback(args.model_dir))
        
        # Monitor memory before training
        monitor_memory("Before training start")
        logger.info("*** Starting training with reduced parameters ***")
        
        # Try a test batch before full training to check for issues
        try:
            logger.info("Testing training with a single batch...")
            # Get a single batch from dataloader
            dataloader = trainer.get_train_dataloader()
            batch = next(iter(dataloader))
            
            # Test a forward pass
            outputs = model(**{k: v.to(model.device) for k, v in batch.items() if k != "labels"})
            logger.info("Forward pass successful")
            
            # Test backward pass
            if "labels" in batch:
                outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
                loss = outputs.loss
                loss.backward()
                logger.info("Backward pass successful")
            
            # Clear memory after test
            del outputs, batch, dataloader
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            monitor_memory("After test batch")
        except Exception as e:
            logger.error(f"Error during test batch: {e}")
            # Still try training with smaller batch
            training_args.per_device_train_batch_size = 1
            training_args.gradient_accumulation_steps = 16
            logger.info("Reduced batch size to 1 for training")
        
        # Train with exception handling and auto-retry with smaller parameters
        try:
            logger.info("Starting full training")
            trainer.train()
        except Exception as e:
            logger.error(f"Error during training: {e}")
            monitor_memory("At training error")
            
            # Try to save model anyway if possible
            try:
                trainer.save_model(args.model_dir + "/partial_model")
                logger.info("Saved partial model despite error")
            except:
                logger.error("Could not save partial model")
            
            # Try with an even smaller batch size
            try:
                logger.info("Retrying with smaller batch size and model")
                training_args.per_device_train_batch_size = 1
                training_args.gradient_accumulation_steps = 16
                
                # Load the smallest model possible
                model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base", 
                                                             revision="main",
                                                             low_cpu_mem_usage=True)
                                                             
                model.config.use_cache = False
                
                # Initialize a smaller trainer
                smaller_trainer = Seq2SeqTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_datasets["train"][:10],  # Just use 10 samples
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                )
                
                smaller_trainer.train()
                smaller_trainer.save_model(args.model_dir)
                logger.info("Succeeded with smaller parameters")
            except Exception as sub_e:
                logger.error(f"Error even with smaller params: {sub_e}")
                raise
        
        # Save the model
        monitor_memory("After training, before saving")
        logger.info("*** Saving the model ***")
        trainer.save_model(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
        
        # Save model info
        try:
            model_info = {
                "model_name": args.model_name,
                "max_input_length": args.max_input_length,
                "max_target_length": args.max_target_length,
                "training_completed": True
            }
            with open(os.path.join(args.model_dir, "model_info.json"), "w") as f:
                json.dump(model_info, f)
        except Exception as e:
            logger.warning(f"Error saving model info: {e}")
        
        monitor_memory("End of training function")
        logger.info("*** Training completed ***")
        
    except Exception as outer_e:
        logger.error(f"Outer exception in training function: {outer_e}")
        # Save a dummy model so we have something for inference
        try:
            os.makedirs(args.model_dir, exist_ok=True)
            with open(os.path.join(args.model_dir, "dummy_model.txt"), "w") as f:
                f.write("Training failed, but we need a file for the pipeline to continue")
        except:
            pass
        raise

if __name__ == "__main__":
    args = parse_args()
    train(args)
