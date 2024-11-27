import os
import torch
import torch.nn as nn
import pandas as pd
import json
import zipfile
import gc
import time
import wandb  # Import wandb for logging

from transformers import (
    AutoTokenizer,
    AutoConfig,
    Gemma2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import traceback

# ===============================
# Step 1: Define Centralized Configuration
# ===============================

# Define a configuration dictionary
config = {
    "project_name": "suicide_incident_classification",
    "model_name": "google/gemma-2-2b",
    "num_train_epochs": 3,  # Reduced for testing
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "learning_rate": 5e-5,  # Reduced learning rate for initial testing
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "fp16": True,
    "output_dir": "./models",
    "logging_dir": "./logs",
    "num_labels": 2,
    "max_input_length": 256,  # Further reduced for testing
    "max_target_length": 128,
}

# Check if TPU is available
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
    device = xm.xla_device()
    print("TPU detected and will be used for training.")
except ImportError:
    TPU_AVAILABLE = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"TPU not detected. Using device: {device}")

# ===============================
# Step 2: Initialize wandb with Centralized Config
# ===============================

# Initialize wandb with the config dictionary
wandb.init(
    project=config["project_name"],
    config=config,
    name=f"{config['model_name'].split('/')[-1]}_training_run",  # Name your run based on model name
    reinit=True  # Allows multiple runs in a single script
)

# ===============================
# Step 3: Remove Custom F1 Loss Functions
# ===============================

# Removed custom loss functions since we'll use the model's default loss

# ===============================
# Step 4: Define Corresponding Metrics Functions
# ===============================

def compute_metrics_binary(eval_preds):
    """
    Compute F1 score and F1 loss for binary classification.
    """
    logits, labels = eval_preds
    predictions = torch.argmax(torch.sigmoid(torch.tensor(logits)), dim=1).cpu().numpy()

    f1 = f1_score(labels, predictions, average='macro')
    f1_loss = 1 - f1

    return {'f1': f1, 'f1_loss': f1_loss}

def compute_metrics_micro(eval_preds):
    """
    Compute micro-averaged F1 score and F1 loss for multi-class classification.
    """
    logits, labels = eval_preds
    predictions = torch.argmax(torch.sigmoid(torch.tensor(logits)), dim=1).cpu().numpy()

    f1_micro = f1_score(labels, predictions, average='micro')
    f1_micro_loss = 1 - f1_micro

    return {'f1_micro': f1_micro, 'f1_micro_loss': f1_micro_loss}

# ===============================
# Step 5: Define the Custom Trainer Class
# ===============================

class CustomTrainer(Trainer):
    """
    Custom Trainer for logging purposes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # No custom loss function

    # Use the default compute_loss method

# ===============================
# Step 6: Use Pre-defined Model Class
# ===============================

# No changes needed; already using Gemma2ForSequenceClassification

# ===============================
# Step 7: Initialize the Tokenizer and Model
# ===============================

model_name = config["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use AutoConfig to load Gemma2Config
config_gemma = AutoConfig.from_pretrained(model_name)
config_gemma.num_labels = config["num_labels"]  # Ensure num_labels is set correctly

# Initialize the model using Gemma2ForSequenceClassification
model = Gemma2ForSequenceClassification.from_pretrained(
    model_name,
    config=config_gemma
).to(device)

# ===============================
# Step 8: Load JSON and CSV Files
# ===============================

# Load JSON configuration
with open("feature_classification_input.json", "r") as f:
    json_data = json.load(f)

# Load feature and label data
features_df = pd.read_csv('train_features.csv')
labels_df = pd.read_csv('train_labels.csv')

# ===============================
# Step 9: Set 'uid' as Index for Alignment
# ===============================

# Check and set 'uid' as index for features_df
if 'uid' in features_df.columns:
    features_df.set_index('uid', inplace=True)
else:
    print("Warning: 'uid' column not found in features_df. Ensure that labels align correctly.")

# Check and set 'uid' as index for labels_df
if 'uid' in labels_df.columns:
    labels_df.set_index('uid', inplace=True)
else:
    print("Warning: 'uid' column not found in labels_df. Assuming labels align with features_df by order.")
    # Align labels_df index with features_df index based on order
    if len(labels_df) == len(features_df):
        labels_df.index = features_df.index
    else:
        raise ValueError("Labels and features DataFrames do not align by index or order. Please verify your data.")

# Fill any missing values in labels_df
labels_df.fillna(0, inplace=True)  # Assuming labels are numerical; adjust if different

# ===============================
# Step 10: Prepare Data for Each Section
# ===============================

def create_classification_data(section):
    """
    Create input and label pairs for a given section.
    Args:
        section (dict): A section dictionary from the JSON configuration.
    Returns:
        list of dict: List containing input_text and label for each sample.
    """
    qa_data = []
    for variable in section["variables"]:
        variable_id = variable.get("id")
        question = variable.get("question", "No question provided.")
        criteria = "; ".join(variable.get("criteria", []))
        examples = "; ".join(variable.get("examples", []))
        exclusions = "; ".join(variable.get("exclusions", []))
        notes = "; ".join(variable.get("notes", []))

        # Full prompt with context
        full_prompt = (
            f"Question: {question}\n"
            f"Definition: {variable.get('definition', 'No definition provided.')}\n"
            f"Criteria: {criteria}\n"
            f"Examples: {examples}\n"
            f"Exclusions: {exclusions}\n"
            f"Notes: {notes}\n"
            "Answer:"
        )

        # Check if the variable exists in `labels_df.columns`
        if variable_id not in labels_df.columns:
            print(f"Variable ID '{variable_id}' not found in labels_df columns.")
            continue

        # Generate QA samples for each entry in the dataset
        for idx, row in features_df.iterrows():
            context = row.get("NarrativeCME", "No context provided.")
            label = labels_df.loc[idx, variable_id]
            qa_data.append({
                "input_text": f"{full_prompt} Context: {context}",
                "label": label
            })
    return qa_data

# ===============================
# Step 11: Preprocess Function
# ===============================

def preprocess_function(examples):
    """
    Tokenize the input and labels.
    Args:
        examples (dict): A batch of examples from the dataset.
    Returns:
        dict: Tokenized inputs and labels.
    """
    inputs = examples["input_text"]
    labels = examples["label"]

    # Tokenize inputs with padding and truncation
    model_inputs = tokenizer(
        inputs, max_length=config["max_input_length"], truncation=True, padding="max_length"
    )

    # Convert labels to tensors
    # For binary and multi-class classification, labels should be integers
    labels = torch.tensor(labels).long()  # (batch_size,)

    model_inputs["labels"] = labels
    return model_inputs

# ===============================
# Step 12: Define Custom Callback for Logging Time
# ===============================

class LogTimeCallback(TrainerCallback):
    """
    Custom callback to log training and validation time to wandb.
    """
    def __init__(self):
        self.train_start_time = None
        self.eval_start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()

    def on_train_end(self, args, state, control, **kwargs):
        train_time = time.time() - self.train_start_time
        wandb.log({"training_time_seconds": train_time})
        print(f"Training time: {train_time:.2f} seconds")

    def on_evaluate(self, args, state, control, **kwargs):
        if self.eval_start_time is None:
            self.eval_start_time = time.time()
        else:
            eval_time = time.time() - self.eval_start_time
            wandb.log({"validation_time_seconds": eval_time})
            print(f"Validation time: {eval_time:.2f} seconds")
            self.eval_start_time = None

    def on_evaluate_end(self, args, state, control, metrics=None, **kwargs):
        if self.eval_start_time is not None:
            eval_time = time.time() - self.eval_start_time
            wandb.log({"validation_time_seconds": eval_time})
            print(f"Validation time: {eval_time:.2f} seconds")
            self.eval_start_time = None

# ===============================
# Step 13: Instantiate Data Collator
# ===============================

data_collator = DataCollatorWithPadding(tokenizer)

# ===============================
# Step 14: Training Loop for Each Section with Conditional F1 Metrics
# ===============================

for i, section in enumerate(json_data["sections"]):
    section_name = section["name"].replace(" ", "_").lower()

    print(f"\nProcessing section: {section_name}")

    # Create classification data for the current section
    qa_data = create_classification_data(section)
    if not qa_data:
        print(f"No valid QA data found for section: {section_name}")
        continue

    # Create a DataFrame from qa_data
    dataset = pd.DataFrame(qa_data)

    # Split into training and evaluation sets
    train_data, eval_data = train_test_split(dataset, test_size=0.2, random_state=42)

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_data)
    eval_dataset = Dataset.from_pandas(eval_data)

    # Apply preprocessing
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["input_text", "label"]
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["input_text", "label"]
    )

    # Set format for PyTorch
    train_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )
    eval_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )

    # Determine metrics based on section index
    if i < 4:
        # First 4 sections: Binary Classification
        compute_metrics_fn = compute_metrics_binary
        metric_for_best_model = "f1"
    else:
        # Last section: Categorical Classification
        compute_metrics_fn = compute_metrics_micro
        variable_id = section["variables"][0]["id"]  # Assuming single variable per section
        current_num_labels = labels_df[variable_id].nunique()
        metric_for_best_model = "f1_micro"
        print(f"Section '{section_name}' has {current_num_labels} classes.")

    # Update the model's num_labels if it has changed
    if i >= 4 and current_num_labels != config["num_labels"]:
        print(f"Updating num_labels from {config['num_labels']} to {current_num_labels}")
        config_gemma.num_labels = current_num_labels
        model = Gemma2ForSequenceClassification.from_pretrained(
            model_name,
            config=config_gemma
        ).to(device)
        config["num_labels"] = current_num_labels  # Update the config dictionary

    # Define training arguments with wandb integration using centralized config
    training_args = TrainingArguments(
        output_dir=os.path.join(config["output_dir"], f"{section_name}_qa_finetuned"),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        fp16=config["fp16"],
        evaluation_strategy=config["evaluation_strategy"],
        save_strategy=config["save_strategy"],  # Save checkpoint at each epoch
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        logging_dir=os.path.join(config["logging_dir"], f"{section_name}_qa_finetuned"),
        # Removed logging_steps to enable epoch-wise logging
        logging_strategy="epoch",  # Log at the end of each epoch
        report_to=["wandb"],  # Enable logging to wandb
        optim="adamw_torch_xla_fused" if TPU_AVAILABLE else "adamw",
        save_total_limit=2,  # Keep only the last 2 checkpoints
        gradient_checkpointing=True
        # Uncomment below if using a learning rate scheduler
        # warmup_steps=1000,
        # lr_scheduler_type='linear',
    )

    # Instantiate the CustomTrainer with the appropriate metrics function
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[LogTimeCallback()]  # Add custom callback for logging time
    )

    try:
        print(f"\nStarting training for section: {section_name}...")
        trainer.train()
    except Exception as e:
        print(f"\nTraining failed for section: {section_name} with error: {e}")
        traceback.print_exc()

    # Save the best model
    best_model_dir = os.path.join(training_args.output_dir, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    # Optionally, zip the model directory
    zip_file_path = os.path.join(training_args.output_dir, "best_model.zip")
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for root, dirs, files in os.walk(best_model_dir):
            for file in files:
                zipf.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), best_model_dir)
                )

    # Clear memory more aggressively
    del train_dataset, eval_dataset, trainer
    gc.collect()
    if TPU_AVAILABLE:
        torch_xla.core.xla_model.xla_shard()  # Properly clear TPU memory
    else:
        torch.cuda.empty_cache()
    # Move model back to CPU to free up TPU memory
    model.to("cpu")

    # Finish the wandb run for this section
    wandb.finish()