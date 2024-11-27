import os
import logging
from pathlib import Path

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoConfig,
    BertForSequenceClassification,  # Use BertForSequenceClassification
    BertTokenizerFast
)
from tqdm import tqdm

# ===============================
# Configuration and Setup
# ===============================

# Get the absolute path to the directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent

# Set up paths for easy modification
SUBMISSION_PATH = SCRIPT_DIR / "submission.csv"
FEATURES_PATH = SCRIPT_DIR / "data/test_features.csv"
SUBMISSION_FORMAT_PATH = SCRIPT_DIR / "data/submission_format.csv"
MODELS_DIR = SCRIPT_DIR / "models/BERT-finetuned"

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set DEBUG flag
DEBUG = True  # Set to True to enable debugging

# Set up logging
log_file = SCRIPT_DIR / "debug_log.txt"

if DEBUG:
    logging_level = logging.DEBUG
else:
    logging_level = logging.INFO

logging.basicConfig(
    level=logging_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Optional: also output to console
    ]
)
logger = logging.getLogger()

# ===============================
# Prediction Function
# ===============================

def generate_predictions(model, tokenizer, features_df, submission_format):
    """
    Generate predictions for all variables using a single multi-label classification model.
    
    Args:
        model: The fine-tuned classification model.
        tokenizer: The tokenizer corresponding to the model.
        features_df (DataFrame): DataFrame containing the features.
        submission_format (DataFrame): DataFrame defining the submission format.
    
    Returns:
        DataFrame: Predictions for all variables.
    """
    predictions = pd.DataFrame(index=submission_format.index, columns=submission_format.columns)

    # Define batch size based on available hardware
    batch_size = 16  # Adjust based on your GPU/TPU memory

    num_samples = features_df.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    logger.info(f"Starting prediction in {num_batches} batches of size {batch_size}")

    for batch_idx in tqdm(range(num_batches), desc="Generating Predictions"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_features = features_df.iloc[start_idx:end_idx]

        # Prepare inputs: Use only the "NarrativeLE" column
        inputs = batch_features["NarrativeLE"].tolist()
        encodings = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=512,  # Ensure this matches your model's max input length
            return_tensors="pt"
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}

        # Perform inference
        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits  # Shape: (batch_size, num_labels)
            probs = torch.sigmoid(logits).cpu().numpy()  # Shape: (batch_size, num_labels)

        # Debugging: Log logits and probabilities for the batch
        if DEBUG:
            logger.debug(f"Batch {batch_idx + 1}/{num_batches}")
            logger.debug(f"Logits:\n{logits.cpu().numpy()}")
            logger.debug(f"Probabilities:\n{probs}")

        # Apply threshold to determine labels (e.g., 0.5)
        threshold = 0.5
        labels = (probs >= threshold).astype(int)  # Binary predictions

        # Assign predictions to the DataFrame
        predictions.iloc[start_idx:end_idx] = labels

    return predictions

# ===============================
# Main Function
# ===============================

def main():
    # ===============================
    # Load Data
    # ===============================
    
    try:
        features_df = pd.read_csv(FEATURES_PATH, index_col=0)
        logger.info(f"Loaded test features of shape {features_df.shape}")
    except FileNotFoundError:
        logger.error(f"Features file not found at '{FEATURES_PATH}'. Please check the path.")
        return

    try:
        submission_format = pd.read_csv(SUBMISSION_FORMAT_PATH, index_col=0)
        logger.info(f"Loaded submission format of shape: {submission_format.shape}")
    except FileNotFoundError:
        logger.error(f"Submission format file not found at '{SUBMISSION_FORMAT_PATH}'. Please check the path.")
        return

    # ===============================
    # Data Alignment
    # ===============================
    
    # Ensure that features_df has all uids in submission_format
    missing_uids = submission_format.index.difference(features_df.index)
    if not missing_uids.empty:
        logger.warning(f"The following uids are missing in features_df and will be filled with default context: {missing_uids.tolist()}")
        # Create empty rows for missing uids with default context
        empty_features = pd.DataFrame(index=missing_uids, columns=features_df.columns)
        empty_features['NarrativeLE'] = "No context provided."
        features_df = pd.concat([features_df, empty_features])

    # Reindex features_df to match submission_format index
    features_df = features_df.reindex(submission_format.index)

    # ===============================
    # Load Model and Tokenizer
    # ===============================
    
    logger.info(f"Loading tokenizer and model from '{MODELS_DIR}'")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR))
        logger.info(f"Tokenizer loaded successfully from '{MODELS_DIR}'.")
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from '{MODELS_DIR}'. Attempting to download 'bert-base-uncased' tokenizer.")
        try:
            # Initialize the tokenizer (assuming 'bert-base-uncased')
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            # Save the tokenizer to MODELS_DIR for future use
            tokenizer.save_pretrained(MODELS_DIR)
            logger.info(f"'bert-base-uncased' tokenizer downloaded and saved to '{MODELS_DIR}' successfully.")
        except Exception as ex:
            logger.error(f"Failed to download and save 'bert-base-uncased' tokenizer: {ex}")
            return

    # Load the configuration and set num_labels to match the number of variables
    config_path = os.path.join(MODELS_DIR, "config.json")
    try:
        config_gemma = AutoConfig.from_pretrained(config_path)
        config_gemma.num_labels = len(submission_format.columns)  # Ensure num_labels matches number of variables
        logger.info(f"Configuration loaded and num_labels set to {config_gemma.num_labels}.")
    except Exception as e:
        logger.error(f"Failed to load or modify config from '{config_path}': {e}")
        return

    # Load the fine-tuned model
    try:
        model = BertForSequenceClassification.from_pretrained(
            str(MODELS_DIR),
            config=config_gemma
        ).to(device)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model from '{MODELS_DIR}': {e}")
        return

    # ===============================
    # Generate Predictions
    # ===============================
    
    try:
        predictions_df = generate_predictions(
            model=model,
            tokenizer=tokenizer,
            features_df=features_df,
            submission_format=submission_format
        )
    except Exception as e:
        logger.error(f"Error during prediction generation: {e}")
        return

    # ===============================
    # Post-processing
    # ===============================
    
    # Ensure predictions_df has the same columns as submission_format
    for col in submission_format.columns:
        if col not in predictions_df.columns:
            predictions_df[col] = 0  # Assign default value if missing
            logger.warning(f"Column '{col}' missing in predictions. Filled with default value 0.")

    # Ensure data types match submission_format
    for col in predictions_df.columns:
        predictions_df[col] = predictions_df[col].astype(int)

    # Reset index to make 'uid' a column
    predictions_df.reset_index(inplace=True)
    predictions_df.rename(columns={'index': 'uid'}, inplace=True)

    # Ensure 'uid' column is of type string
    predictions_df['uid'] = predictions_df['uid'].astype(str)

    # Reorder columns to match submission format
    submission_columns = submission_format.reset_index().columns
    predictions_df = predictions_df[submission_columns]

    # ===============================
    # Debugging: Log a sample of the predictions DataFrame
    # ===============================
    
    if DEBUG:
        logger.debug(f"Sample of predictions DataFrame:\n{predictions_df.head()}")

    # ===============================
    # Save Predictions
    # ===============================
    
    try:
        predictions_df.to_csv(SUBMISSION_PATH, index=False)
        logger.info(f"Predictions saved to '{SUBMISSION_PATH}'")
    except Exception as e:
        logger.error(f"Failed to save predictions to '{SUBMISSION_PATH}': {e}")
        return

    # ===============================
    # Clean Up
    # ===============================
    
    del model
    del tokenizer
    torch.cuda.empty_cache()
    logger.info("Memory cleared and inference completed.")

# ===============================
# Entry Point
# ===============================

if __name__ == "__main__":
    main()
