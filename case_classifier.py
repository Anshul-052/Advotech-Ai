import pandas as pd
import os
from transformers import pipeline
from tqdm import tqdm
import torch 
import time

# --- CONFIGURATION ---
INDEX_PATH = 'master_case_index.csv'
OUTPUT_INDEX_PATH = 'master_case_index_classified.csv' # New file to hold the classified data

# Categories defined in xlia_app.py - These are the labels the AI will use!
# FIX APPLIED: Reordering categories to prioritize Constitutional and Criminal Law
# This helps the zero-shot model place foundational cases into the correct specific bucket.
CANDIDATE_LABELS = [
    "Constitutional Law", 
    "Criminal Law", 
    "Contract Law", 
    "Property & Land Law", 
    "Intellectual Property (IP)", 
    "Company & Corporate Law",
    "General Legal Question" # Catch-all moved to the end
]

# --- CORE LOGIC ---

def classify_cases(df):
    """
    Uses a Zero-Shot Classification model (BART-large-mnli) to assign a legal category to each case.
    The classification is performed on the 'Raw_Text_Excerpt' column.
    """
    start_time = time.time()
    
    # --- GPU CHECK AND DEVICE SETUP ---
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Starting classification on device: {device_name}")
    
    print("Loading Zero-Shot Classification Model (facebook/bart-large-mnli)...")
    
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device_name 
        )
    except Exception as e:
        print(f"FATAL ERROR: Could not load the model. Ensure you run: pip install transformers pandas torch. Error: {e}")
        return None

    classified_categories = []
    
    # Ensure the required column for classification is present
    if 'Raw_Text_Excerpt' not in df.columns:
        print("ERROR: Column 'Raw_Text_Excerpt' not found. Ensure data_indexer.py ran correctly.")
        return None

    print(f"Starting classification for {len(df)} cases using BATCH_SIZE=16...")
    
    BATCH_SIZE = 16 
    
    # Use tqdm to show progress during the time-intensive classification process
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Classifying Cases"):
        texts_to_classify = df['Raw_Text_Excerpt'].iloc[i:i + BATCH_SIZE].tolist()
        
        # Run classification on the batch
        results = classifier(texts_to_classify, CANDIDATE_LABELS, multi_label=False)
        
        # Extract the highest scoring label for each result in the batch
        for result in results:
            # The result is sorted by score, so the first label is the best match
            best_label = result['labels'][0]
            classified_categories.append(best_label)

    # Add the new column to the DataFrame
    df['Legal_Category'] = classified_categories
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\nClassification complete.")
    print(f"Total time taken: {duration:.2f} seconds.")
    
    return df

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(INDEX_PATH):
        print(f"ERROR: Master Index CSV not found at: {INDEX_PATH}")
        print("Please run data_indexer.py first to create the index.")
    else:
        try:
            # Load the full, indexed, but unclassified data
            master_df = pd.read_csv(INDEX_PATH)
            
            # --- START CLASSIFICATION ---
            classified_df = classify_cases(master_df)

            if classified_df is not None:
                # Save the new classified index
                classified_df.to_csv(OUTPUT_INDEX_PATH, index=False)
                print(f"✅ SUCCESS: Classified Index saved to {OUTPUT_INDEX_PATH}")
                
                # --- NEXT STEP GUIDE ---
                print("\n--- NEXT STEP: UPDATE EMBEDDINGS & RUN APP ---")
                print("1. Re-run embedding_generator.py (it will now use the new classified index).")
                print("2. Relaunch the application: streamlit run xlia_app.py (it uses the classified index by default).")

        except Exception as e:
            print(f"An unexpected error occurred during execution: {e}")
            print("Please check your environment and file structure.")
