import os
import logging
from pathlib import Path
from datetime import datetime  # <--- Added import for datetime

import torch
import pandas as pd
import json
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
)
from tqdm import tqdm
import gc

# Get the absolute path to the directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent

# Set up paths for easy modification
SUBMISSION_PATH = SCRIPT_DIR / "submission.csv"
FEATURES_PATH = SCRIPT_DIR / "data/test_features.csv"
SUBMISSION_FORMAT_PATH = SCRIPT_DIR / "data/submission_format.csv"
JSON_DATA_PATH = SCRIPT_DIR / "feature_classification_input.json"
MODELS_DIR = SCRIPT_DIR / "models"

# Define the number of models per variable (set to 1 as per clarification)
NUM_MODELS_PER_VARIABLE = 1

# Construct model paths for each variable
# Directory structure assumption:
# models/
#   Masked Language Modeling/
#     Argument/
#       best_model/
#         config.json
#         model.safetensors
#         tokenizer_config.json
#         tokenizer.json
#         vocab.txt
#     DepressedMood/
#       best_model/
#         ...
#     ...
MASKED_MODELS_DIR = MODELS_DIR / "Masked Language Modeling"

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set DEBUG flag
DEBUG = True  # Set to True to enable debugging

# Generate a timestamp in the format YYYYMMDD_HHMMSS
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # <--- Added timestamp generation

# Set up logging with a timestamped log file
log_file = SCRIPT_DIR / f"debug_log_{timestamp}.txt"  # <--- Updated log_file with timestamp

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
    
    # Attempt to match generated text to response option values (assuming integer labels)
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
    
    # Attempt to match generated text to response option labels (case-insensitive substring match)
    for option in response_options:
        if option['label'].lower() in generated_text_lower:
            label_value = option['value']
            if DEBUG:
                logger.debug(f"Matched label: '{option['label']}' with value: {label_value}")
            return label_value
    
    # If no match is found, return default value (e.g., 0)
    if DEBUG:
        logger.debug("No match found. Returning default value 0.")
    return 0

def generate_predictions_for_variable(model_path, tokenizer, variable, features_df, submission_format):
    """
    Generate predictions for a single variable using its corresponding model.
    
    Args:
        model_path (Path): Path to the model directory.
        tokenizer (AutoTokenizer): Tokenizer for the model.
        variable (dict): Variable configuration from JSON.
        features_df (DataFrame): Features DataFrame.
        submission_format (DataFrame): Submission format DataFrame.
    
    Returns:
        Series: Predictions for the variable.
    """
    variable_id = variable.get("id")
    response_options = variable.get("responseOptions", [])
    
    # Initialize a list to store predictions
    predictions = []
    
    question = variable.get("question", "No question provided.")
    criteria = "; ".join(variable.get("criteria", []))
    examples = "; ".join(variable.get("examples", []))
    exclusions = "; ".join(variable.get("exclusions", []))
    notes = "; ".join(variable.get("notes", []))
    definition = variable.get("definition", "No definition provided.")
    
    # Construct the prompt with [MASK] token
    full_prompt = (
        # "As an expert in mental health assessment, please answer the following question based on the provided context.\n"
        # f"Question: {question}\n"
        # f"Definition: {definition}\n"
        # f"Criteria: {criteria}\n"
        # f"Examples: {examples}\n"
        # f"Exclusions: {exclusions}\n"
        # f"Notes: {notes}\n"
        f"Answer: [MASK]"
    )
    
    # **Removed redundant loading of test_features.csv**
    
    # Load the model once before the loop
    try:
        model = AutoModelForMaskedLM.from_pretrained(
            str(model_path),
            use_safetensors=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        model.eval()
        logger.info(f"Model loaded successfully for Variable '{variable_id}'.")
    except Exception as e:
        logger.error(f"Failed to load model for Variable '{variable_id}' from '{model_path}': {e}")
        # **Use features_df instead of test_features_df**
        return pd.Series([0]*len(features_df.index), index=features_df.index, name=variable_id)
    
    # **Removed redundant loading of test_features.csv**
    
    # Iterate over each UID in the test_features
    for uid in tqdm(features_df.index, desc=f"Predicting '{variable_id}'"):
        try:
            if uid in features_df.index:
                context = features_df.loc[uid, "NarrativeCME"]
            else:
                context = "No context provided."
            logger.debug(f"UID: {uid}, Context: {context}")
            
            input_text = f" Context: {context} {full_prompt}"
            # Additional processing...
        except Exception as e:
            logger.error(f"Error processing UID '{uid}': {e}")
            predictions.append(0)  # Assign default value on failure
            continue
        
        # Debugging: Log the input prompt
        if DEBUG:
            logger.debug(f"Input Prompt for UID {uid} and Variable '{variable_id}':\n{input_text}\n")
        
        # Tokenize input
        try:
            inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        except Exception as e:
            logger.error(f"Tokenization failed for UID {uid} with Variable '{variable_id}': {e}")
            predictions.append(0)  # Assign default value on failure
            continue
        
        with torch.no_grad():
            try:
                outputs = model(inputs)
                logits = outputs.logits
                mask_token_index = torch.where(inputs == tokenizer.mask_token_id)[1]
                
                # Get the logits for the [MASK] token
                mask_token_logits = logits[0, mask_token_index, :]
                
                # Get the highest scoring token ID
                predicted_token_id = torch.argmax(mask_token_logits, dim=1).item()
                
                # Decode the token ID to get the word
                predicted_token = tokenizer.decode([predicted_token_id]).strip().lower()
                
                # Further Cleaning: Remove any surrounding quotes or special characters
                predicted_token = predicted_token.strip('\'"!?.,;:')
                
            except Exception as e:
                logger.error(f"Model inference failed for UID '{uid}': {e}")
                predictions.append(0)  # Assign default value on failure
                continue
        
        # Debugging: Log the model's response
        if DEBUG:
            logger.debug(f"Model Response for UID {uid} and Variable '{variable_id}':\n{predicted_token}\n")
        
        # Map generated text to label value
        label_value = map_output_to_label(predicted_token, response_options)
        predictions.append(label_value)
    
    # Convert predictions to a pandas Series
    predictions_series = pd.Series(predictions, index=features_df.index, name=variable_id)
    
    # Debugging: Log a sample of predictions
    if DEBUG:
        logger.debug(f"Sample Predictions for Variable '{variable_id}':\n{predictions_series.head()}")
    
    # Clean up to free memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return predictions_series

def main():
    # Load JSON data
    try:
        with open(JSON_DATA_PATH, "r") as f:
            json_data = json.load(f)
        logger.info(f"Loaded JSON data from '{JSON_DATA_PATH}'.")
    except Exception as e:
        logger.error(f"Failed to load JSON data from '{JSON_DATA_PATH}': {e}")
        return
    
    # Load test_features.csv
    try:
        features_df = pd.read_csv(FEATURES_PATH, index_col='uid')
        logger.info(f"Loaded test features of shape {features_df.shape} from '{FEATURES_PATH}'.")
    except Exception as e:
        logger.error(f"Failed to load test features from '{FEATURES_PATH}': {e}")
        return
    
    # Load submission_format.csv for submission structure only
    try:
        submission_format = pd.read_csv(SUBMISSION_FORMAT_PATH, index_col='uid')
        logger.info(f"Loaded submission format of shape: {submission_format.shape} from '{SUBMISSION_FORMAT_PATH}'.")
    except Exception as e:
        logger.error(f"Failed to load submission format from '{SUBMISSION_FORMAT_PATH}': {e}")
        return
    
    # Initialize predictions_df with test_features index
    predictions_df = pd.DataFrame(index=features_df.index)
    
    # Iterate over each section
    for section in json_data.get("sections", []):
        section_name = section.get("name", "").replace(" ", "_").lower()
        logger.info(f"\nProcessing Section: {section_name}")
        
        # Iterate over each variable in the section
        for variable in section.get("variables", []):
            variable_id = variable.get("id")
            logger.info(f"\nProcessing Variable: {variable_id}")
            
            # Construct the variable's model path
            variable_model_dir = MASKED_MODELS_DIR / variable_id / "best_model"
            if not variable_model_dir.exists():
                logger.error(f"Model directory does not exist: {variable_model_dir}")
                # Assign default predictions
                predictions_df[variable_id] = 0
                continue
            
            # Load tokenizer once before generating predictions
            try:
                tokenizer = AutoTokenizer.from_pretrained(variable_model_dir)
                logger.info(f"Loaded tokenizer from '{variable_model_dir}'.")
            except Exception as e:
                logger.error(f"Failed to load tokenizer from '{variable_model_dir}': {e}")
                predictions_df[variable_id] = 0
                continue
            
            # Generate predictions for the current variable using its model
            variable_predictions = generate_predictions_for_variable(
                model_path=variable_model_dir,
                tokenizer=tokenizer,
                variable=variable,
                features_df=features_df,
                submission_format=submission_format  # If needed for reference
            )
            
            # Add predictions to the main predictions_df
            predictions_df[variable_id] = variable_predictions
            
            # Debugging: Log a sample of predictions
            if DEBUG:
                logger.debug(f"Sample Predictions for Variable '{variable_id}':\n{predictions_df[variable_id].head()}")
    
    # Ensure predictions_df has all required columns from submission_format
    for col in submission_format.columns:
        if col not in predictions_df.columns:
            logger.warning(f"Variable '{col}' was not processed. Filling with default value 0.")
            predictions_df[col] = 0  # or any default value you prefer
    
    # Ensure data types match submission_format
    for col in predictions_df.columns:
        predictions_df[col] = predictions_df[col].astype(int)
    
    # Reset index to make 'uid' a column
    predictions_df.reset_index(inplace=True)
    predictions_df.rename(columns={'uid': 'uid'}, inplace=True)
    
    # Ensure 'uid' column is of type string
    predictions_df['uid'] = predictions_df['uid'].astype(str)
    
    # Reorder columns to match submission format
    submission_columns = submission_format.reset_index().columns.tolist()
    if 'uid' in submission_columns:
        submission_columns.remove('uid')  # Assuming 'uid' is already in predictions_df
    predictions_df = predictions_df[['uid'] + submission_columns]
    
    # Debugging: Log a sample of the predictions DataFrame
    if DEBUG:
        logger.debug(f"Sample of predictions DataFrame:\n{predictions_df.head()}")
    
    # Save predictions to CSV
    try:
        predictions_df.to_csv(SUBMISSION_PATH, index=False)
        logger.info(f"Predictions saved to '{SUBMISSION_PATH}'.")
    except Exception as e:
        logger.error(f"Failed to save predictions to '{SUBMISSION_PATH}': {e}")

if __name__ == "__main__":
    main()
