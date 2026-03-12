import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import time

# --- CONFIGURATION ---
MODEL_NAME = 'all-mpnet-base-v2' # Upgraded semantic model
INDEX_PATH = 'master_case_index_classified.csv' # Must exist after running the classifier
EMBEDDINGS_PATH = 'case_embeddings_classified.pt'
BATCH_SIZE = 64 # Optimal batch size for GPU processing

# --- MAIN EXECUTION FUNCTION ---
def generate_embeddings():
    """Loads classified data, generates semantic vectors (embeddings) using the advanced LLM, and saves them."""
    
    if not os.path.exists(INDEX_PATH):
        print(f"ERROR: Classified index not found at: {INDEX_PATH}")
        print("Please ensure master_case_index_classified.csv exists (run case_classifier.py first).")
        return

    start_time = time.time()
    
    try:
        # Load the classified index file
        df = pd.read_csv(INDEX_PATH)
        
        # --- Essential Column Verification ---
        required_columns = ['Full_Text', 'Legal_Category']
        for col in required_columns:
            if col not in df.columns:
                # Use a specific, easy-to-read error for missing data
                raise ValueError(f"DATA ERROR: Required column '{col}' is missing from the CSV file. Did the data indexer/classifier run correctly?")

        # --- GPU/CPU Device Check ---
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Starting model generation on device: {device}...")
        
        # 1. Load AI Model (Transfer Learning)
        print(f"Loading Sentence Transformer Model (AI's semantic core) - {MODEL_NAME}...")
        # Automatically downloads and caches the model
        model = SentenceTransformer(MODEL_NAME, device=device)

        # The AI will learn from the full, clean text
        texts_to_embed = df['Full_Text'].tolist()
        
        print(f"Calculating {len(df)} embeddings with batch size {BATCH_SIZE}. This process is highly optimized for GPU.")
        
        # 2. Encode the Texts into Vectors (Vectorization)
        # Convert texts into high-dimensional vectors (the AI model's "memory")
        embeddings = model.encode(
            texts_to_embed, 
            show_progress_bar=True, 
            convert_to_tensor=True, 
            device=device,
            batch_size=BATCH_SIZE # Use batches for faster GPU processing
        )
        
        # 3. Save the Model
        # Using a more robust, atomic save operation
        temp_path = EMBEDDINGS_PATH + ".tmp"
        torch.save(embeddings, temp_path)
        os.replace(temp_path, EMBEDDINGS_PATH) # Atomically replace the old file
        
        end_time = time.time()
        duration = end_time - start_time

        print("\n--- Embedding Generation SUCCESS ---")
        print(f"Embeddings (AI Model) saved to: {EMBEDDINGS_PATH}")
        print(f"Total size: {len(df)} documents.")
        print(f"Total time taken: {duration:.2f} seconds.")
        
    except FileNotFoundError:
        print(f"\nFILE ERROR: Could not find required index file: {INDEX_PATH}. Please verify the path.")
    except ValueError as ve:
        print(f"\nDATA ERROR: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during model generation: {e}")
        print("\n*** Troubleshooting Tips ***")
        print("1. If GPU intended, verify PyTorch is installed with CUDA support.")
        print("2. Check network connection for model download.")

if __name__ == "__main__":
    generate_embeddings()
