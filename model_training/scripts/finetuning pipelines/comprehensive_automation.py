# ===============================
# Step 1: Environment Setup (Final)
# ===============================

# Uncomment and run the following line if you're using a notebook
# !pip install --upgrade transformers torch datasets wandb scikit-learn pynvml

# Import necessary libraries
import os
import sys
import torch
from transformers import (
    AutoTokenizer,  # Generic tokenizer
    AutoModelForSequenceClassification,  # For sequence classification
    AutoModelForMaskedLM,  # For fill-mask models
    AutoModelForSeq2SeqLM,  # For text generation models
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from datasets import Dataset
import wandb
import numpy as np
import pandas as pd
import json
import random
import gc
import time
import math
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import traceback
import warnings
import zipfile
import datetime

# ===============================
# Step 2: Set Random Seeds for Reproducibility
# ===============================
# Set PyTorch CUDA allocation configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
seed = 42  # You can choose any integer value

# Set the seed for Python's built-in random module
random.seed(seed)

# Set the seed for NumPy
np.random.seed(seed)

# Set the seed for PyTorch
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Ensure deterministic behavior in PyTorch (may impact performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===============================
# Step 3: Define Helper Functions (Removed Dynamic GPU Memory Allocation)
# ===============================
# All dynamic GPU memory allocation functions have been removed.
# Batch sizes are now fixed as per `base_config`.

# ===============================
# Step 4: Define Centralized Configuration
# ===============================

base_config = {
    "project_name": "suicide_incident_classification",
    "num_train_epochs": 10,  # Adjust as needed
    "train_batch_size": 8,   # Fixed batch size
    "eval_batch_size": 8,    # Fixed eval batch size
    "gradient_accumulation_steps": 4,  # Fixed value
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "output_dir": "./models",    # Base output directory
    "logging_dir": "./logs",
    "max_input_length": 512,    # Adjust based on data analysis
    "save_total_limit": 1,       # Keep only the best model
    "continue_training": False,  # Set to True to resume from checkpoint
}

# ===============================
# Step 5: Load JSON and CSV Files
# ===============================

# Load JSON configuration
with open("feature_classification_input.json", "r") as f:
    json_data = json.load(f)

# Load feature and label data
features_df = pd.read_csv('train_features.csv')
labels_df = pd.read_csv('train_labels.csv')

# ===============================
# Step 6: Prepare Data
# ===============================

# Set 'uid' as index for features_df
if 'uid' in features_df.columns:
    features_df.set_index('uid', inplace=True)
else:
    print("Warning: 'uid' column not found in features_df.")

# Set 'uid' as index for labels_df
if 'uid' in labels_df.columns:
    labels_df.set_index('uid', inplace=True)
else:
    print("Warning: 'uid' column not found in labels_df.")

# Fill any missing values in labels_df
labels_df.fillna(0, inplace=True)

# Ensure labels are integers
labels_df = labels_df.astype(int)

# ===============================
# Step 7: Create Label Mappings (Final)
# ===============================

# Initialize label_mappings dictionary
label_mappings = {}

for section in json_data["sections"]:
    for variable in section["variables"]:
        variable_id = variable["id"]
        var_type = variable["type"]  # "binary" or "categorical"
        response_options = variable["responseOptions"]

        # For binary and categorical, map value to label
        if var_type == "binary":
            # Ensure unique labels for binary classes
            # Assuming binary labels are 0: "No", 1: "Yes"
            label_mappings[variable_id] = {
                0: "No",
                1: "Yes"
            }
        else:
            # For categorical, ensure each value maps to a unique single label
            label_mappings[variable_id] = {
                option["value"]: option["label"] for option in response_options
            }

# Example structure:
# label_mappings = {
#     "DepressedMood": {0: "No", 1: "Yes"},
#     "MentalIllnessTreatmentCurrnt": {0: "No", 1: "Yes"},
#     ...
# }

# ===============================
# Step 8: Define Model List (Final)
# ===============================

# Define a list of models to iterate through (zero-shot classification models removed)
model_list = [
    {
        "model_type": "sequence_classification",
        "model_name": "roberta-base",
    },
    {
        "model_type": "sequence_classification",
        "model_name": "facebook/bart-large-mnli",
    },
    {
        "model_type": "sequence_classification",
        "model_name": "bert-base-uncased",
    },
    {
        "model_type": "sequence_classification",
        "model_name": "distilbert-base-uncased",
    },
    # Add more models as needed, ensuring they align with the specified model_type
]

# ===============================
# Step 9: Create Classification Data Function (Final)
# ===============================

def create_classification_data(
    section,
    features_df,
    labels_df,
    variable,
    label_type="binary",
    model_type="sequence_classification",
    tokenizer=None,
    model_name=None  # Added parameter
):
    """
    Creates classification data based on the model type.

    Args:
        section (dict): Section configuration from JSON.
        features_df (DataFrame): Features DataFrame.
        labels_df (DataFrame): Labels DataFrame.
        variable (dict): Variable configuration from JSON.
        label_type (str): Type of classification ("binary" or "categorical").
        model_type (str): Type of model ("sequence_classification", "fill_mask", "text_generation").
        tokenizer: Tokenizer object.
        model_name (str): Name of the model.

    Returns:
        list: List of dictionaries with 'input_text' and 'target_text'.
    """
    qa_data = []
    variable_id = variable["id"]
    var_type = variable["type"]  # "binary" or "categorical"
    question = variable.get("question", "No question provided.")
    criteria = "; ".join(variable.get("criteria", []))
    examples = "; ".join(variable.get("examples", []))
    exclusions = "; ".join(variable.get("exclusions", []))
    notes = "; ".join(variable.get("notes", []))
    definition = variable.get("definition", "No definition provided.")

    for idx in features_df.index:
        context = features_df.loc[idx, "NarrativeCME"]
        label = labels_df.loc[idx, variable_id]
        label_word = label_mappings[variable_id].get(label, "Unknown")

        if model_type == "sequence_classification":
            input_text = f"{context}"
            target_text = str(label)  # Integer label as string

        elif model_type == "fill_mask":
            mask_token = tokenizer.mask_token
            if mask_token is None:
                print(f"Warning: {model_name} tokenizer does not have a {mask_token} token. Skipping example.")
                continue  # Skip this example
            else:
                # Ensure a space after the mask token
                input_text = (
                    f"Context: {context}\n"
                    f"Answer: {mask_token} "  # Added space after <mask>
                )
                target_text = label_word  # The word to predict

        elif model_type == "text_generation":
            # Format similar to sequence classification but treated as text generation
            input_text = (
                f"Question: {question}\n"
                f"Definition: {definition}\n"
                f"Criteria: {criteria}\n"
                f"Examples: {examples}\n"
                f"Exclusions: {exclusions}\n"
                f"Notes: {notes}\n"
                f"Context: {context}\n"
                f"Answer:"
            )
            target_text = label_word  # The word to generate

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        qa_data.append({
            "input_text": input_text,
            "target_text": target_text
        })

    return qa_data

# ===============================
# Step 10: Define Tokenization Function (Final)
# ===============================

class TokenizeFunction:
    def __init__(self, tokenizer, max_length, model_type="sequence_classification"):
        self.tokenizer = tokenizer
        self.max_input_length = max_length
        self.model_type = model_type

    def __call__(self, examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]

        if self.model_type in ["sequence_classification", "text_generation"]:
            if self.model_type == "text_generation":
                # For text generation, use text_target instead of as_target_tokenizer
                model_inputs = self.tokenizer(
                    inputs,
                    max_length=self.max_input_length,
                    truncation=True,
                    padding="max_length",
                    text_target=targets,  # Use text_target for labels
                )
            else:
                # For sequence classification
                model_inputs = self.tokenizer(
                    inputs,
                    max_length=self.max_input_length,
                    truncation=True,
                    padding="max_length",
                )
                model_inputs["labels"] = [int(label) for label in targets]

        elif self.model_type == "fill_mask":
            # Tokenize inputs
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.max_input_length,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,  # Ensure attention masks are returned
            )

            # Initialize labels as a copy of input_ids
            labels = []
            mask_token_id = self.tokenizer.mask_token_id

            for i, input_ids in enumerate(model_inputs["input_ids"]):
                labels_ids = [-100] * len(input_ids)  # Initialize all labels as -100
                try:
                    mask_index = input_ids.index(mask_token_id)
                    # Tokenize the target word
                    target_ids = self.tokenizer.encode(targets[i], add_special_tokens=False)
                    if len(target_ids) == 0:
                        target_ids = [self.tokenizer.unk_token_id]
                    elif len(target_ids) > 1:
                        print(f"Warning: Target '{targets[i]}' is tokenized into multiple tokens. Using the first token.")
                    labels_ids[mask_index] = target_ids[0]
                except ValueError:
                    # No [MASK] token found; keep all labels as -100
                    pass  # labels_ids is already initialized to all -100
                labels.append(labels_ids)

            model_inputs["labels"] = labels

            # **Debugging: Print tokenized input and labels**
            for i in range(min(3, len(model_inputs["input_ids"]))):  # Print first 3 examples
                decoded = self.tokenizer.decode(model_inputs["input_ids"][i])
                print(f"Decoded Input {i}: '{decoded}'")
                print(f"Label Sequence {i}: {model_inputs['labels'][i]}")

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        return model_inputs

# ===============================
# Step 11: Define Compute Metrics Function (Final)
# ===============================

def compute_metrics(p, model_type, tokenizer, label_type, section_name):
    if model_type == "fill_mask":
        # p.predictions shape: (batch_size, sequence_length, vocab_size)
        preds = p.predictions
        label_ids = p.label_ids

        # Identify the position of the [MASK] token in each example
        mask_token_id = tokenizer.mask_token_id
        pred_labels = []
        true_labels = []

        for i in range(preds.shape[0]):
            input_ids = p.label_ids[i]
            try:
                mask_index = input_ids.tolist().index(mask_token_id)
                # Get the logits for the [MASK] position
                mask_logits = preds[i, mask_index, :]
                # Get the top prediction
                predicted_token_id = mask_logits.argmax()
                predicted_label = tokenizer.decode([predicted_token_id]).strip()
                pred_labels.append(predicted_label)

                # Decode the true label
                true_label_id = label_ids[i][mask_index]
                if true_label_id == -100:
                    continue  # Skip ignored labels
                true_label = tokenizer.decode([true_label_id]).strip()
                true_labels.append(true_label)
            except ValueError:
                # No [MASK] token found; skip this example
                continue

        # Reverse the label_mappings to get word to label
        word_to_label = {}
        for var_id, mapping in label_mappings.items():
            for val, lbl in mapping.items():
                # Lowercase for case-insensitive matching
                word_to_label[lbl.lower()] = val

        # Convert predictions and labels to integers
        preds_int = []
        labels_int = []
        for pred, true in zip(pred_labels, true_labels):
            # Find the corresponding label value
            pred_val = word_to_label.get(pred.lower(), -1)
            true_val = word_to_label.get(true.lower(), -1)
            preds_int.append(pred_val)
            labels_int.append(true_val)

        # Filter out invalid predictions
        valid_indices = [i for i, (pred, true) in enumerate(zip(preds_int, labels_int)) if pred != -1 and true != -1]
        preds_filtered = [preds_int[i] for i in valid_indices]
        labels_filtered = [labels_int[i] for i in valid_indices]

        if not preds_filtered:
            return {'f1': 0.0, 'accuracy': 0.0}

        # Determine F1 averaging method based on label type
        if label_type == "binary":
            f1_average = 'binary'
        else:
            f1_average = 'macro'  # Use 'macro' for multi-class

        # Calculate F1 score and accuracy
        f1 = f1_score(labels_filtered, preds_filtered, average=f1_average)
        acc = accuracy_score(labels_filtered, preds_filtered)

        return {'f1': f1, 'accuracy': acc}

    elif model_type == "sequence_classification":
        # For sequence classification models
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids

        # Debugging: Print unique labels and predictions
        unique_labels = np.unique(labels)
        unique_preds = np.unique(preds)
        print(f"Unique Labels: {unique_labels}")
        print(f"Unique Predictions: {unique_preds}")

        # Determine F1 averaging method based on label type
        if label_type == "binary":
            f1_average = 'binary'
        else:
            f1_average = 'macro'  # Use 'macro' for multi-class

        # Calculate F1 score and accuracy
        f1 = f1_score(labels, preds, average=f1_average)
        acc = accuracy_score(labels, preds)

        print(f"F1 Score: {f1}, Accuracy: {acc}")

        return {'f1': f1, 'accuracy': acc}

    elif model_type == "text_generation":
        # For text generation models
        preds = p.predictions  # shape: (batch_size, sequence_length)
        labels = p.label_ids

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Convert to integers based on label_mappings
        # Assuming that the generated text corresponds directly to label words
        preds_int = []
        labels_int = []

        for pred, label in zip(decoded_preds, decoded_labels):
            # Map generated text to label value
            found = False
            for var_id, mapping in label_mappings.items():
                for val, lbl in mapping.items():
                    if lbl.lower() == pred.lower():
                        preds_int.append(val)
                        found = True
                        break
                if found:
                    break
            if not found:
                preds_int.append(-1)  # Unknown prediction

            # Map label text to label value
            found = False
            for var_id, mapping in label_mappings.items():
                for val, lbl in mapping.items():
                    if lbl.lower() == label.lower():
                        labels_int.append(val)
                        found = True
                        break
                if found:
                    break
            if not found:
                labels_int.append(-1)  # Unknown label

        # Filter out invalid predictions
        valid_indices = [i for i, pred in enumerate(preds_int) if pred != -1 and labels_int[i] != -1]
        preds_filtered = [preds_int[i] for i in valid_indices]
        labels_filtered = [labels_int[i] for i in valid_indices]

        if not preds_filtered:
            return {'f1': 0.0, 'accuracy': 0.0}

        # Determine F1 averaging method based on label type
        if label_type == "binary":
            f1_average = 'binary'
        else:
            f1_average = 'macro'  # Use 'macro' for multi-class

        # Calculate F1 score and accuracy
        f1 = f1_score(labels_filtered, preds_filtered, average=f1_average)
        acc = accuracy_score(labels_filtered, preds_filtered)

        return {'f1': f1, 'accuracy': acc}

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

# ===============================
# Step 12: Define Prediction Logger Callback (Final)
# ===============================

class PredictionLoggerCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            f1 = metrics.get("f1", None)
            accuracy = metrics.get("accuracy", None)
            loss = metrics.get("eval_loss", None)

            print(f"Evaluation Metrics:")
            if f1 is not None:
                print(f" - F1 Score: {f1:.4f}")
            if accuracy is not None:
                print(f" - Accuracy: {accuracy:.4f}")
            if loss is not None:
                print(f" - Loss: {loss:.4f}")

            # Additional custom logging can be added here

# ===============================
# Step 13: Define Compute Metrics Wrapper (Final)
# ===============================

def compute_metrics_wrapper(p, model_type, tokenizer, label_type, section_name):
    return compute_metrics(p, model_type, tokenizer, label_type, section_name)

# ===============================
# Step 14: Define the Training Loop with Fixed Batch Size (Final)
# ===============================

def train_models(model_list, base_config, desired_eff_batch_size=32, device_id=0):
    """
    Trains multiple models on multiple variables with fixed batch sizing and gradient accumulation.

    Args:
        model_list (list): List of model configurations.
        base_config (dict): Centralized configuration dictionary.
        desired_eff_batch_size (int): Desired effective batch size.
        device_id (int): GPU device ID.
    """
    # For tracking the current model index
    model_index_file = "current_model_index.txt"

    # Initialize model index
    if os.path.exists(model_index_file):
        with open(model_index_file, "r") as f:
            try:
                current_model_index = int(f.read().strip())
                if current_model_index >= len(model_list):
                    current_model_index = 0
            except:
                current_model_index = 0
    else:
        current_model_index = 0

    while True:  # Infinite loop; modify as needed
        model_config = model_list[current_model_index]
        model_type = model_config["model_type"]
        model_name = model_config["model_name"]

        print(f"\n=== Training Model: {model_name} | Type: {model_type} ===")

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        print(f"Mask token: {tokenizer.mask_token}")
        print(f"Mask token ID: {tokenizer.mask_token_id}")

        # Initialize model based on model_type
        if model_type == "sequence_classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
            model.gradient_checkpointing_enable()  # Enable gradient checkpointing
            model.to(device)  # Move model to the correct device
        elif model_type == "fill_mask":
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            model.gradient_checkpointing_enable()  # Enable gradient checkpointing
            model.to(device)
        elif model_type == "text_generation":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.gradient_checkpointing_enable()  # Enable gradient checkpointing
            model.to(device)
        else:
            raise ValueError(f"Invalid model_type specified: {model_type}")

        # Iterate through sections and variables
        for section in json_data["sections"]:
            # Define section_name here
            section_name = section["name"].replace(" ", "_").lower()
            print(f"\nProcessing Section: {section_name}")

            for variable in section["variables"]:
                variable_id = variable["id"]
                label_type = variable.get("type", "binary")  # Define label_type here
                var_type = variable.get("type", "binary")
                print(f"\nProcessing Variable: {variable_id} | Type: {label_type}")

                # For sequence classification, initialize the model with correct num_labels
                if model_type == "sequence_classification":
                    current_num_labels = len(label_mappings[variable_id])
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        num_labels=current_num_labels,
                    )
                    model.to(device)  # Move model to the correct device

                # Create classification data
                qa_data = create_classification_data(
                    section=section,
                    features_df=features_df,
                    labels_df=labels_df,
                    variable=variable,
                    label_type=label_type,
                    model_type=model_type,
                    tokenizer=tokenizer,      # Pass tokenizer here
                    model_name=model_name     # Pass model_name here
                )

                if not qa_data:
                    print(f"No valid QA data found for variable: {variable_id}")
                    continue

                # Create a DataFrame from qa_data
                dataset_df = pd.DataFrame(qa_data)

                # **Separate class labels for balancing**
                if model_type == "fill_mask":
                    # For fill_mask, use the underlying class labels for balancing
                    # Assume label_mappings map integers to words, e.g., {0: "No", 1: "Yes"}
                    dataset_df['class_label'] = dataset_df['target_text'].map(
                        lambda x: {v: k for k, v in label_mappings[variable_id].items()}.get(x, -1)
                    )
                else:
                    # For sequence classification, labels are already integers
                    dataset_df['class_label'] = dataset_df['target_text'].astype(int)

                # Drop any rows with invalid class_label
                dataset_df = dataset_df[dataset_df['class_label'] != -1]

                # Split using stratified sampling based on 'class_label'
                try:
                    train_df, eval_df = train_test_split(
                        dataset_df,
                        test_size=0.2,
                        stratify=dataset_df['class_label'],  # Use class labels for stratification
                        random_state=seed  # 'seed' is now defined
                    )
                except ValueError as e:
                    print(f"Error during train_test_split for variable: {variable_id} | Error: {e}")
                    continue

                # Reset index to prevent '__index_level_0__' duplication
                train_df = train_df.reset_index(drop=True)
                eval_df = eval_df.reset_index(drop=True)

                # Balance the training dataset based on 'class_label'
                train_label_counts = train_df['class_label'].value_counts()
                print(f"Training label distribution before balancing: {train_label_counts}")

                if len(train_label_counts) < 2:
                    print(f"Only one class present in training data for variable: {variable_id}. Skipping.")
                    continue

                majority_class = train_label_counts.idxmax()
                minority_class = train_label_counts.idxmin()

                majority_df = train_df[train_df['class_label'] == majority_class]
                minority_df = train_df[train_df['class_label'] == minority_class]

                # Upsample minority class
                minority_upsampled = minority_df.sample(n=len(majority_df), replace=True, random_state=seed)
                train_df_balanced = pd.concat([majority_df, minority_upsampled]).sample(frac=1, random_state=seed)

                print(f"Training label distribution after balancing: {train_df_balanced['class_label'].value_counts()}")

                # Remove 'class_label' as it's no longer needed
                train_df_balanced = train_df_balanced.drop(columns=['class_label'])

                # Convert back to Dataset
                train_dataset = Dataset.from_pandas(train_df_balanced)
                eval_dataset = Dataset.from_pandas(eval_df)

                # Batch sizes are now fixed as per `base_config`
                dynamic_batch_size = base_config["train_batch_size"]
                eval_batch_size = base_config["eval_batch_size"]

                # Calculate fixed gradient_accumulation_steps
                gradient_accumulation_steps = math.ceil(desired_eff_batch_size / dynamic_batch_size)
                base_config["gradient_accumulation_steps"] = gradient_accumulation_steps
                print(f"Set gradient_accumulation_steps to: {gradient_accumulation_steps} to achieve an effective batch size of {desired_eff_batch_size}")

                # Initialize tokenizer function
                tokenize_function = TokenizeFunction(
                    tokenizer=tokenizer,
                    max_length=base_config["max_input_length"],
                    model_type=model_type
                )

                # Apply preprocessing
                train_dataset = train_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=["input_text", "target_text"],
                    num_proc=4  # Enable multiprocessing
                )
                eval_dataset = eval_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=["input_text", "target_text"],
                    num_proc=4  # Enable multiprocessing
                )

                # Set format for PyTorch tensors
                if model_type == "sequence_classification":
                    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
                    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
                elif model_type == "fill_mask":
                    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
                    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
                elif model_type == "text_generation":
                    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
                    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

                # Determine number of labels (only for sequence classification)
                if model_type == "sequence_classification":
                    current_num_labels = len(label_mappings[variable_id])
                    print(f"Variable '{variable_id}' has {current_num_labels} classes.")

                # Initialize wandb run for this variable and model
                run_name = f"{section_name}_{variable_id}_{model_name}_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                wandb.init(
                    project=base_config["project_name"],
                    name=run_name,  # Ensure this is unique and distinct from output_dir
                    config={
                        **base_config,
                        **model_config,
                        "section_name": section_name,
                        "variable_id": variable_id,
                        "label_type": label_type,
                    },
                    sync_tensorboard=True  # Sync TensorBoard logs
                )

                # Define compute_metrics function
                def compute_metrics_fn(p):
                    return compute_metrics_wrapper(
                        p,
                        model_type=model_type,
                        tokenizer=tokenizer,
                        label_type=label_type,
                        section_name=section_name
                    )

                # Define training arguments
                training_args_dict = {
                    "output_dir": os.path.join(
                        base_config["output_dir"], model_name, section_name, variable_id, "best_model"
                    ),
                    "run_name": run_name,  # Set run_name separately
                    "num_train_epochs": base_config["num_train_epochs"],
                    "per_device_train_batch_size": dynamic_batch_size,
                    "per_device_eval_batch_size": eval_batch_size,
                    "gradient_accumulation_steps": base_config["gradient_accumulation_steps"],
                    "learning_rate": base_config["learning_rate"],
                    "weight_decay": base_config["weight_decay"],
                    "eval_strategy": "epoch",  # Updated from 'evaluation_strategy'
                    "save_strategy": "epoch",
                    "logging_dir": os.path.join(
                        base_config["logging_dir"], model_name, section_name, variable_id
                    ),
                    "logging_strategy": "steps",  # Ensure logging is done per steps
                    "logging_steps": 1,  # Log after every step
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "f1",
                    "greater_is_better": True,
                    "report_to": "wandb",
                    "save_total_limit": base_config["save_total_limit"],  # Keep only the best model
                    "resume_from_checkpoint": base_config["continue_training"],  # Continue training from checkpoint
                    "fp16": True,  # Enable mixed precision
                }

                # Conditionally add 'predict_with_generate' if model is text_generation
                if model_type == "text_generation":
                    training_args_dict["predict_with_generate"] = True

                training_args = TrainingArguments(**training_args_dict)

                # Initialize Trainer with callbacks
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    compute_metrics=compute_metrics_fn,
                    callbacks=[PredictionLoggerCallback()]  # Now correctly defined
                )

                # Train the model
                try:
                    print(f"\nStarting training for variable: {variable_id} with model: {model_name}...")
                    if base_config["continue_training"]:
                        # Load from last checkpoint
                        trainer.train(resume_from_checkpoint=True)
                    else:
                        trainer.train()
                except Exception as e:
                    print(f"\nTraining failed for variable: {variable_id} with model: {model_name} and error: {e}")
                    traceback.print_exc()
                    wandb.finish()
                    continue  # Proceed to the next variable

                # Save the best model
                model_save_path = os.path.join(
                    base_config["output_dir"], model_name, section_name, variable_id, "best_model"
                )
                trainer.save_model(model_save_path)
                tokenizer.save_pretrained(model_save_path)

                # Optionally, zip the model directory
                zip_file_path = os.path.join(
                    base_config["output_dir"], model_name, section_name, variable_id, "best_model.zip"
                )
                with zipfile.ZipFile(zip_file_path, 'w') as zipf:
                    for root, dirs, files in os.walk(model_save_path):
                        for file in files:
                            zipf.write(
                                os.path.join(root, file),
                                os.path.relpath(os.path.join(root, file), model_save_path)
                            )

                # Clear memory
                gc.collect()
                torch.cuda.empty_cache()

                # Finish the wandb run for this variable and model
                wandb.finish()

        # Update the model index for the next iteration
        current_model_index = (current_model_index + 1) % len(model_list)
        with open(model_index_file, "w") as f:
            f.write(str(current_model_index))

        print(f"\nCompleted training for model: {model_name}. Moving to the next model.")


# ===============================
# Step 17: Run the Training Function (Final)
# ===============================

# Run the training function
if __name__ == "__main__":
    train_models(model_list, base_config, desired_eff_batch_size=16, device_id=0)
