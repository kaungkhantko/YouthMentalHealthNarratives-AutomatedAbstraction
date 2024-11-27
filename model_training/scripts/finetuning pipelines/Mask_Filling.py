import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import numpy as np
import pandas as pd
import json
import random
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import zipfile

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Base configuration
base_config = {
    "project_name": "suicide_incident_classification",
    "num_train_epochs": 7,
    "train_batch_size": 64,
    "eval_batch_size": 64,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "output_dir": "./models",
    "logging_dir": "./logs",
    "max_input_length": 512,
    "save_total_limit": 1,
    "continue_training": False,
    'max_steps' : 300

}

# Load JSON configuration
with open("feature_classification_input.json", "r") as f:
    json_data = json.load(f)

# Load feature and label data
features_df = pd.read_csv('/data/llm/ethan_copy/youth-mental-health-runtime/data/train_features.csv')
labels_df = pd.read_csv('/data/llm/ethan_copy/youth-mental-health-runtime/data/train_labels.csv')

# Set 'uid' as index
if 'uid' in features_df.columns:
    features_df.set_index('uid', inplace=True)
if 'uid' in labels_df.columns:
    labels_df.set_index('uid', inplace=True)

# Fill missing values and ensure integer labels
labels_df.fillna(0, inplace=True)
labels_df = labels_df.astype(int)

# ===============================
# Step 5: Create Label Mappings
# ===============================
label_mappings = {}
for section in json_data["sections"]:
    for variable in section["variables"]:
        variable_id = variable["id"]
        var_type = variable["type"]  # "binary" or "categorical"
        response_options = variable.get("responseOptions", {})

        if var_type == "binary":
            label_mappings[variable_id] = {0: "No", 1: "Yes"}
        else:
            label_mappings[variable_id] = {
                option["value"]: option["label"] for option in response_options
            }

# Function to create classification data
def create_classification_data(
    section,
    features_df,
    labels_df,
    variable,
    label_type="binary",
    model_type="fill_mask",
    tokenizer=None,
    model_name=None
):
    qa_data = []
    variable_id = variable["id"]
    question = variable.get("question", "No question provided.")
    for idx in features_df.index:
        context = features_df.loc[idx, "NarrativeCME"]
        label = labels_df.loc[idx, variable_id]
        label_word = label_mappings[variable_id].get(label, "Unknown")

        if model_type == "fill_mask":
            mask_token = tokenizer.mask_token
            if mask_token is None:
                print(f"Warning: {model_name} tokenizer does not have a mask token. Skipping example.")
                continue

            # Format the input with the variable ID and masked label
            input_text = (
                f"Context: {context}\n"
                f"[{variable_id}] {question} answer : {mask_token}"
            )
            target_text = label_word

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        qa_data.append({
            "input_text": input_text,
            "target_text": target_text
        })

    return qa_data

# Tokenization function
class TokenizeFunction:
    def __init__(self, tokenizer, max_length, model_type="fill_mask"):
        self.tokenizer = tokenizer
        self.max_input_length = max_length
        self.model_type = model_type

    def __call__(self, examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]

        if self.model_type == "fill_mask":
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.max_input_length,
                truncation=True,
                padding="max_length",
                return_attention_mask=True
            )

            # Initialize labels for [MASK] prediction
            labels = []
            mask_token_id = self.tokenizer.mask_token_id

            for i, input_ids in enumerate(model_inputs["input_ids"]):
                labels_ids = [-100] * len(input_ids)  # Initialize all labels as -100
                try:
                    mask_index = input_ids.index(mask_token_id)  # Locate [MASK] position
                    target_ids = self.tokenizer.encode(targets[i], add_special_tokens=False)
                    if len(target_ids) > 0:
                        labels_ids[mask_index] = target_ids[0]  # Assign target word to [MASK]
                except ValueError:
                    pass
                labels.append(labels_ids)

            model_inputs["labels"] = labels
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        return model_inputs


# Custom metrics function
def compute_metrics(eval_pred, variable_type="binary"):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Mask -100 labels for valid comparison
    mask = labels != -100
    valid_labels = labels[mask]
    valid_predictions = predictions[mask]

    if valid_labels.shape[0] == 0:  # Handle case with no valid labels
        return {"f1": 0, "accuracy": 0}

    if variable_type == "binary":
        f1 = f1_score(valid_labels, valid_predictions, average="binary")
    else:  # Multiclass variables
        f1 = f1_score(valid_labels, valid_predictions, average="micro")

    accuracy = accuracy_score(valid_labels, valid_predictions)
    return {"f1": f1, "accuracy": accuracy}



# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased', use_fast=True)

# Process sections and variables
datasets_dict = {}
for section in json_data["sections"]:
    section_name = section["name"].replace(" ", "_").lower()
    print(f"\nProcessing Section: {section_name}")

    for variable in section["variables"]:
        variable_id = variable["id"]
        label_type = variable.get("type", "binary")
        print(f"\nProcessing Variable: {variable_id} | Type: {label_type}")

        # Create classification data
        qa_data = create_classification_data(
            section=section,
            features_df=features_df,
            labels_df=labels_df,
            variable=variable,
            label_type=label_type,
            model_type="fill_mask",
            tokenizer=tokenizer,
            model_name='google-bert/bert-base-uncased'
        )

        if not qa_data:
            print(f"No valid QA data found for variable: {variable_id}")
            continue

        dataset_df = pd.DataFrame(qa_data)
        dataset_df = Dataset.from_pandas(dataset_df)

        # Tokenize dataset
        tokenize_function = TokenizeFunction(
            tokenizer=tokenizer,
            max_length=base_config["max_input_length"],
            model_type="fill_mask"
        )
        dataset_df = dataset_df.map(tokenize_function, batched=True, remove_columns=["input_text", "target_text"], num_proc=4)

        # Split into train and test sets
        train_test_split_result = dataset_df.train_test_split(test_size=0.2, seed=seed)
        datasets_dict[variable_id] = train_test_split_result

# Training loop for each variable
for variable_id, split_datasets in datasets_dict.items():
    print(f"\nTraining model for variable: {variable_id}")

    train_dataset = split_datasets["train"]
    eval_dataset = split_datasets["test"]

    # Initialize model
    model = AutoModelForMaskedLM.from_pretrained('google-bert/bert-base-uncased')

    training_args = TrainingArguments(
        output_dir=os.path.join(base_config["output_dir"], variable_id),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=base_config["learning_rate"],
        per_device_train_batch_size=base_config["train_batch_size"],
        per_device_eval_batch_size=base_config["eval_batch_size"],
        num_train_epochs=base_config["num_train_epochs"],
        weight_decay=base_config["weight_decay"],
        logging_dir=base_config["logging_dir"],
        save_total_limit=base_config["save_total_limit"],
        load_best_model_at_end=True,  # Load best model at the end of training
        greater_is_better=True,       # Whether a higher metric is better
        fp16=True,
        max_steps=base_config['max_steps']
)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()
        # Save the best model
    model_save_path = os.path.join(
        base_config["output_dir"], variable_id, "best_model"
    )
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # Optionally, zip the model directory
    zip_file_path = os.path.join(
        base_config["output_dir"], variable_id, "best_model.zip"
    )
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for root, dirs, files in os.walk(model_save_path):
            for file in files:
                zipf.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), model_save_path)
                )

    print(f"Model training and saving complete for variable: {variable_id}")
