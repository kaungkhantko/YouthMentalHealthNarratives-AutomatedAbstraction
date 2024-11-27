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
    T5Config,
    T5Model,
    PreTrainedModel,
    AutoTokenizer,
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
    "project_name": "suicide_incident_classification",  # Replace with your project name
    "model_name": "t5-base",
    "num_train_epochs": 4,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 16,
    "learning_rate": 8e-3,
    "weight_decay": 0.05,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "fp16": True,
    "output_dir": "./models",
    "logging_dir": "./logs",
    "num_labels": 1,  # For binary classification per variable
    "max_input_length": 512,
    "max_target_length": 128,
    # "logging_steps": 10,  # Removed to enable epoch-wise logging
}
# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
# ===============================
# Step 2: Initialize wandb with Centralized Config
# ===============================

# Initialize wandb with the config dictionary
wandb.init(
    project=config["project_name"],
    config=config,
    name=f"{config['model_name']}_training_run",  # Optional: Name your run
    reinit=True  # Allows multiple runs in a single script
)

# ===============================
# Step 3: Define the Soft F1 Loss Function
# ===============================

def soft_f1_loss(y_pred, y_true):
    """
    Calculate the soft F1 loss between predicted and true labels.
    Args:
        y_pred (Tensor): Predicted logits (batch_size, num_labels).
        y_true (Tensor): True labels (batch_size, num_labels).
    Returns:
        Tensor: Soft F1 loss value.
    """
    y_pred = torch.sigmoid(y_pred)  # Convert logits to probabilities
    y_true = y_true.float()         # Ensure true labels are float

    tp = torch.sum(y_pred * y_true, dim=0)
    fp = torch.sum(y_pred * (1 - y_true), dim=0)
    fn = torch.sum((1 - y_pred) * y_true, dim=0)

    soft_f1 = 2 * tp / (2 * tp + fp + fn + 1e-16)
    loss = 1 - soft_f1  # We subtract because we want to minimize the loss
    return loss.mean()

# ===============================
# Step 4: Define the Custom Model with Classification Head
# ===============================

class T5ForSequenceClassification(PreTrainedModel):
    """
    Custom T5 model with a classification head for sequence classification tasks.
    """
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.t5 = T5Model(config)
        self.classifier = nn.Linear(config.d_model, num_labels)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Get encoder outputs
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Pool the encoder output by taking the mean of hidden states
        sequence_output = encoder_outputs.last_hidden_state  # (batch_size, seq_length, d_model)
        pooled_output = torch.mean(sequence_output, dim=1)   # (batch_size, d_model)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)

        loss = None
        if labels is not None:
            loss = soft_f1_loss(logits, labels)

        return {'loss': loss, 'logits': logits}

# ===============================
# Step 5: Initialize the Tokenizer and Model
# ===============================

model_name = config["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
config_t5 = T5Config.from_pretrained(model_name)

# Initialize the custom model
model = T5ForSequenceClassification(config=config_t5, num_labels=config["num_labels"]).to(device)

# ===============================
# Step 6: Load JSON and CSV Files
# ===============================

# Load JSON configuration
with open("feature_classification_input.json", "r") as f:
    json_data = json.load(f)

# Load feature and label data
features_df = pd.read_csv('train_features.csv')
labels_df = pd.read_csv('train_labels.csv')

# ===============================
# Step 7: Set 'uid' as Index for Alignment
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
# Step 8: Prepare Data for Each Section
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
# Step 9: Preprocess Function
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
    # Assuming binary classification; adjust if multi-class
    labels = torch.tensor(labels).unsqueeze(1).float()  # (batch_size, 1)

    model_inputs["labels"] = labels
    return model_inputs

# ===============================
# Step 10: Define compute_metrics Function
# ===============================

def compute_metrics(eval_preds):
    """
    Compute F1 score based on predictions and true labels.
    Args:
        eval_preds (tuple): Tuple containing predictions and labels.
    Returns:
        dict: Dictionary with F1 score and F1 loss.
    """
    logits, labels = eval_preds
    predictions = (logits > 0).astype(int)  # Threshold at 0 for binary classification
    f1 = f1_score(labels, predictions, average='macro')
    f1_loss = 1 - f1
    return {'f1': f1, 'f1_loss': f1_loss}

# ===============================
# Step 11: Define Custom Callback for Logging Time
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
# Step 12: Instantiate Data Collator
# ===============================

data_collator = DataCollatorWithPadding(tokenizer)

# ===============================
# Step 13: Training Loop for Each Section
# ===============================

for section in json_data["sections"]:
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
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=os.path.join(config["logging_dir"], f"{section_name}_qa_finetuned"),
        # Removed logging_steps to enable epoch-wise logging
        logging_strategy="epoch",  # Log at the end of each epoch
        report_to=["wandb"],  # Enable logging to wandb
    )

    # Initialize the Trainer with custom callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
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

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

    # Finish the wandb run for this section
    wandb.finish()