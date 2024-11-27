import os
import logging
from pathlib import Path

import torch
import pandas as pd
import json
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from tqdm import tqdm

# Get the absolute path to the directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent

# Set up paths for easy modification
SUBMISSION_PATH = SCRIPT_DIR / "submission.csv"
FEATURES_PATH = SCRIPT_DIR / "data/test_features.csv"
SUBMISSION_FORMAT_PATH = SCRIPT_DIR / "data/submission_format.csv"
JSON_DATA_PATH = SCRIPT_DIR / "feature_classification_input.json"
MODELS_DIR = SCRIPT_DIR / "models"

# Define checkpoint numbers
CHECKPOINT_NUMBERS = {
    "mental_health_history_and_current_state": "4.0",
    "specific_mental_health_diagnoses": "2.0",
    "contributing_factors": "4.0",
    "disclosure_of_intent": "4.0",
    "incident_details": "4.0"
}

# Construct model paths using checkpoint numbers
SECTION_MODEL_PATHS = {
    section_name: MODELS_DIR / f"checkpoint_{CHECKPOINT_NUMBERS[section_name]}_section_{section_name}"
    for section_name in CHECKPOINT_NUMBERS
}

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set DEBUG flag
DEBUG = False  # Set to True to enable debugging

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

def map_output_to_label(generated_text, response_options):
    """Map the model's generated text to the corresponding label value."""
    generated_text_lower = generated_text.strip().lower()
    
    # Debugging: Log the generated text and response options
    if DEBUG:
        logger.debug(f"Generated text: '{generated_text}'")
        logger.debug(f"Response options: {response_options}")
    
    # Attempt to match generated text to response option values
    try:
        generated_value = int(generated_text_lower)
        for option in response_options:
            if generated_value == option['value']:
                label_value = option['value']
                if DEBUG:
                    logger.debug(f"Matched value: {label_value}")
                return label_value
    except ValueError:
        if DEBUG:
            logger.debug("Generated text is not an integer.")
    
    # Attempt to match generated text to response option labels
    for option in response_options:
        if option['label'].lower() in generated_text_lower:
            label_value = option['value']
            if DEBUG:
                logger.debug(f"Matched label: '{option['label']}' with value: {label_value}")
            return label_value
    
    # If no match is found, return default value
    if DEBUG:
        logger.debug("No match found. Returning default value 0.")
    return 0

def generate_predictions_for_section(section, model, tokenizer, features_df, submission_format):
    """Generate predictions for all variables in a section."""
    section_predictions = pd.DataFrame(index=submission_format.index)
    variables = section["variables"]
    for variable in variables:
        variable_id = variable.get("id")
        response_options = variable.get("responseOptions", [])
        
        # Debugging: Log the response options for the variable
        if DEBUG:
            logger.debug(f"Response options for Variable {variable_id}: {response_options}")
        
        question = variable.get("question", "No question provided.")
        criteria = "; ".join(variable.get("criteria", []))
        examples = "; ".join(variable.get("examples", []))
        exclusions = "; ".join(variable.get("exclusions", []))
        notes = "; ".join(variable.get("notes", []))
    
        # Full prompt with context, including the system instruction
        full_prompt = (
            "As an expert in mental health assessment, please answer the following question based on the provided context.\n"
            f"Question: {question}\n"
            f"Definition: {variable.get('definition', 'No definition provided.')}\n"
            f"Criteria: {criteria}\n"
            f"Examples: {examples}\n"
            f"Exclusions: {exclusions}\n"
            f"Notes: {notes}\n"
            "Answer:"
        )
    
        # Check if the variable is in the submission format
        if variable_id not in submission_format.columns:
            logger.warning(f"Variable ID '{variable_id}' not found in submission format columns.")
            continue
    
        # Generate predictions for each entry in the submission_format (uids)
        predictions = []
        for uid in tqdm(submission_format.index, total=submission_format.shape[0], desc=f"Predicting {variable_id}"):
            if uid in features_df.index:
                context = features_df.loc[uid, "NarrativeCME"]
            else:
                context = "No context provided."
            input_text = f"{full_prompt} Context: {context}"
    
            # Debugging: Log the input prompt
            if DEBUG:
                logger.debug(f"Input Prompt for UID {uid} and Variable {variable_id}:\n{input_text}\n")
    
            # Tokenize input
            inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
            # Generate output
            outputs = model.generate(inputs, max_length=10)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
            # Debugging: Log the model's response
            if DEBUG:
                logger.debug(f"Model Response for UID {uid} and Variable {variable_id}:\n{generated_text}\n")
    
            # Map generated text to label value
            label_value = map_output_to_label(generated_text, response_options)
    
            # Debugging: Log the label value
            if DEBUG:
                logger.debug(f"Final label value for UID {uid} and Variable {variable_id}: {label_value}")
    
            predictions.append(label_value)
    
        # Convert predictions to integers before adding to DataFrame
        section_predictions[variable_id] = [int(p) for p in predictions]
        # Debugging: Log the predictions for this variable
        if DEBUG:
            logger.debug(f"Predictions for Variable {variable_id}: {predictions}")
    
    return section_predictions

def main():
    # Load JSON and CSV files
    with open(JSON_DATA_PATH, "r") as f:
        json_data = json.load(f)
    
    features_df = pd.read_csv(FEATURES_PATH, index_col=0)
    logger.info(f"Loaded test features of shape {features_df.shape}")
    
    submission_format = pd.read_csv(SUBMISSION_FORMAT_PATH, index_col=0)
    logger.info(f"Loaded submission format of shape: {submission_format.shape}")
    
    # Ensure that features_df has all uids in submission_format
    missing_uids = submission_format.index.difference(features_df.index)
    if not missing_uids.empty:
        logger.warning(f"The following uids are missing in features_df and will be filled with default context: {missing_uids.tolist()}")
        # Create empty rows for missing uids
        empty_features = pd.DataFrame(index=missing_uids, columns=features_df.columns)
        empty_features['NarrativeCME'] = "No context provided."
        features_df = pd.concat([features_df, empty_features])
    
    # Reindex features_df to match submission_format index
    features_df = features_df.reindex(submission_format.index)
    
    # Initialize predictions_df with submission_format index and columns
    predictions_df = submission_format.copy()
    
    # Initialize all columns to zeros
    predictions_df.iloc[:, :] = 0  # Set all values to zero
    
    # Main inference loop
    for section in json_data["sections"]:
        section_name = section["name"].replace(" ", "_").lower()
    
        # Get the model directory for this section
        model_dir = SECTION_MODEL_PATHS.get(section_name)
    
        if model_dir is None or not model_dir.exists():
            logger.warning(f"Model directory '{model_dir}' does not exist. Skipping section '{section_name}'.")
            continue
    
        logger.info(f"\nLoading model and tokenizer for section: {section_name} from '{model_dir}'")
        # Load the tokenizer from the model directory
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        # Load the fine-tuned model
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir)).to(device)
    
        # Generate predictions for the section
        section_predictions = generate_predictions_for_section(
            section, model, tokenizer, features_df, submission_format
        )
    
        # Update predictions_df with the predictions
        predictions_df.update(section_predictions)
    
        # Clean up to free memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
    
    # Ensure predictions_df has the same columns as submission_format
    # Fill missing columns with default values (e.g., 0)
    for col in submission_format.columns:
        if col not in predictions_df.columns:
            predictions_df[col] = 0  # or any default value you prefer
    
    # Ensure data types match submission_format
    for col in predictions_df.columns:
        predictions_df[col] = predictions_df[col].astype(int)
    
    # Reset index to make 'uid' a column
    predictions_df.reset_index(inplace=True)
    predictions_df.rename(columns={'index': 'uid'}, inplace=True)
    
    # Ensure 'uid' column is of type string
    predictions_df['uid'] = predictions_df['uid'].astype(str)
    
    # Reorder columns to match submission format
    predictions_df = predictions_df[submission_format.reset_index().columns]
    
    # Debugging: Log a sample of the predictions DataFrame
    if DEBUG:
        logger.debug(f"Sample of predictions DataFrame:\n{predictions_df.head()}")
    
    # Save predictions to CSV
    predictions_df.to_csv(SUBMISSION_PATH, index=False)
    logger.info(f"Predictions saved to '{SUBMISSION_PATH}'")
    
if __name__ == "__main__":
    main()