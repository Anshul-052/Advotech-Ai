import os
import fitz # PyMuPDF
import pandas as pd
import re
from tqdm import tqdm # For progress bar

# --- CONFIGURATION ---
# IMPORTANT: Update this path to the root of your 7GB legal data structure
root_data_folder = r'D:\XlAI\archive\supreme_court_judgments'
OUTPUT_CSV_PATH = 'master_case_index.csv'

# --- NO LIMIT - INDEX ALL DATA ---
# We are removing the sample_limit = 50 that was here previously.

def clean_text(text):
    """
    Performs basic cleaning and anonymization suitable for legal text.
    1. Removes common noise (multiple newlines, tabs).
    2. Removes common headers/footers (page numbers, line feeds).
    3. Anonymization: A very simple regex to remove what looks like names (optional).
    """
    if not text:
        return ""
        
    # Remove excessive whitespace, newlines, and tabs
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common digital noise (e.g., page numbers, long dashes)
    text = re.sub(r'\d+ of \d+', '', text)
    text = re.sub(r'--*', '', text)
    
    # Simple Anonymization (Warning: Full anonymization requires advanced NLP)
    # This example removes simple capitalized words followed by another capitalized word (potential names)
    text = re.sub(r'([A-Z][a-z]+ [A-Z][a-z]+)', '[PERSON]', text)
    
    return text.strip()

def index_pdf_files(root_path):
    """
    Traverses the folder structure, extracts text from PDFs, and compiles metadata.
    """
    indexed_data = []
    
    # Use os.walk for robust traversal of the entire directory structure
    print(f"Starting deep index of data folder: {root_path}")
    
    # We will track the total number of files to index for the progress bar
    total_files = sum(len(files) for _, _, files in os.walk(root_path) if any(f.lower().endswith('.pdf') for f in files))
    
    if total_files == 0:
        print("ERROR: No PDF files found in the specified root directory.")
        return None

    # Use tqdm to create a progress bar based on the list of files to process
    pbar = tqdm(total=total_files, desc="Indexing PDFs")

    for dirpath, dirnames, filenames in os.walk(root_path):
        year = os.path.basename(dirpath)
        
        # Filter files for PDFs
        for filename in filenames:
            if filename.lower().endswith('.pdf'):
                
                full_file_path = os.path.join(dirpath, filename)
                
                try:
                    # 1. PDF Text Extraction (using PyMuPDF)
                    doc = fitz.open(full_file_path)
                    raw_text = ""
                    for page in doc:
                        raw_text += page.get_text()
                    doc.close()
                    
                    # 2. Text Cleaning
                    full_text_clean = clean_text(raw_text)
                    
                    # 3. Anonymize the case name and clean metadata
                    case_name_simple = filename.replace('.pdf', '').replace('-', ' ').strip()
                    case_name_simple = clean_text(case_name_simple) # Clean noise from the name too
                    
                    # 4. Store in index
                    indexed_data.append({
                        'File_Path_Relative': os.path.relpath(full_file_path, root_path),
                        'Year': year if year.isdigit() else 'Unknown',
                        'Case_Name_Simple': case_name_simple,
                        'Raw_Text_Excerpt': full_text_clean[:500], # Store an excerpt for easy checking
                        'Full_Text': full_text_clean 
                    })
                    
                except Exception as e:
                    print(f"\n[ERROR] Failed to process {filename}: {e}")
                
                pbar.update(1)

    pbar.close()
    
    # Convert list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(indexed_data)
    return df

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    if not os.path.isdir(root_data_folder):
        print(f"ERROR: Root directory not found at: {root_data_folder}")
        print("Please update the 'root_data_folder' variable in the script.")
    else:
        case_df = index_pdf_files(root_data_folder)
        
        if case_df is not None and not case_df.empty:
            # Save the final index
            case_df.to_csv(OUTPUT_CSV_PATH, index=False)
            print(f"\n✅ SUCCESS: Full case index saved to {OUTPUT_CSV_PATH}")
            print(f"Total indexed documents: {len(case_df)}")

            # --- NEXT STEP GUIDE ---
            print("\n--- NEXT STEP: REBUILD THE AI MODEL ---")
            print("1. Run the embedding_generator.py script to vectorize all cases.")
            print("2. Launch the application: streamlit run xlia_app.py")
