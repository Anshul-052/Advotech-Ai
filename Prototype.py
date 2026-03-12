"""
ADVOTECH XLAI: SUPREME COURT JURISPRUDENCE ENGINE
VERSION: 2.5.0
LINE COUNT TARGET: 1000 LINES (PART 1 OF 5)
DESCRIPTION: Professional legal analysis tool for Indian Supreme Court Precedents.
Developed for high-precision semantic retrieval and explainable AI logic.
"""

import streamlit as st
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import json
from json import JSONEncoder
import requests
import time
import re

# ==============================================================================
# --- GLOBAL CONFIGURATION AND SYSTEM PARAMETERS ---
# ==============================================================================

# THE MODEL NAME CORRESPONDS TO THE SBERT ARCHITECTURE USED FOR EMBEDDINGS
MODEL_NAME = 'all-mpnet-base-v2'

# FILE PATHS FOR THE CLASSIFIED INDEX AND PRE-COMPUTED PYTORCH EMBEDDINGS
INDEX_PATH = 'master_case_index_classified.csv'
EMBEDDINGS_PATH = 'case_embeddings_classified.pt'

# GOOGLE GEMINI API KEY FOR GENERATIVE LEGAL EXPLANATIONS
API_KEY = "AIzaSyCwRGjMr2mAOcwO1Zgw3NE_5so8liuMlWk"

# RETRIEVAL CONFIGURATION
TOP_K_RESULTS = 2
MAX_CONTEXT_CHARS = 750
MAX_SINGLE_CASE_CHARS = 4000
MAX_NEW_CASE_INPUT_CHARS = 1500
MAX_SIMULATION_CASE_CHARS = 3000

# LEGAL DOMAIN PERSONA DEFINITIONS FOR SYSTEM PROMPTING
LEGAL_CATEGORIES = {
    "General Legal Question": "General principles of law and jurisprudence.",
    "Criminal Law": "Focus on IPC, CrPC, evidence standards, bail, and prosecution.",
    "Contract Law": "Focus on agreement validity, breach, performance, and damages.",
    "Constitutional Law": "Focus on Fundamental Rights, judicial review, and state action.",
    "Property & Land Law": "Focus on ownership, title disputes, tenancy, and partition.",
    "Intellectual Property (IP)": "Focus on patent, trademark, and copyright infringement.",
    "Company & Corporate Law": "Focus on shareholder rights, mergers, and insolvency/winding up."
}

DEFAULT_PERSONA = "General Legal Question"

# ==============================================================================
# --- CUSTOM CSS FOR PROFESSIONAL BLACK, OFF-WHITE, AND GOLD THEME ---
# ==============================================================================

def inject_custom_css():
    """
    INJECTS A HIGH-CONTRAST PROFESSIONAL LEGAL THEME INTO THE STREAMLIT APP.
    PALETTE:
    - BACKGROUND: OFF-WHITE (#FAF9F6)
    - NAVBAR/SIDEBAR: BLACK (#000000)
    - ACCENTS/BUTTONS: GOLD (#D4AF37)
    - TEXT: DEEP BLACK (#1A1A1A)
    """
    st.markdown("""
    <style>
    /* IMPORT EXTERNAL FONTS FOR LEGAL AESTHETIC */
    @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    /* 1. MAIN APPLICATION CONTAINER STYLING */
    .stApp {
        background-color: #FAF9F6 !important;
        color: #1A1A1A !important;
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* 2. SIDEBAR / NAVIGATION BAR STYLING (THE BLACK NAVBAR) */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 3px solid #D4AF37 !important;
        min-width: 320px !important;
        z-index: 100;
    }

    /* OVERRIDE ALL SIDEBAR TEXT TO WHITE */
    [data-testid="stSidebar"] .stText, 
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #FFFFFF !important;
        font-family: 'Crimson Pro', serif;
    }

    /* SIDEBAR RADIO BUTTONS AND NAVIGATION LINKS */
    div[data-testid="stSidebarUserContent"] .stRadio div[role="radiogroup"] label {
        background-color: transparent !important;
        color: #FFFFFF !important;
        padding: 10px !important;
        border-radius: 5px !important;
        transition: background 0.3s ease;
    }

    div[data-testid="stSidebarUserContent"] .stRadio div[role="radiogroup"] label:hover {
        background-color: #D4AF37 !important;
        color: #000000 !important;
    }

    /* 3. HEADERS AND TYPOGRAPHY SETTINGS */
    h1, h2, h3, h4 {
        font-family: 'Crimson Pro', serif !important;
        color: #1A1A1A !important;
        font-weight: 700 !important;
        border-bottom: none !important;
    }

    h1 {
        font-size: 3rem !important;
        border-bottom: 3px solid #D4AF37 !important;
        padding-bottom: 15px !important;
        margin-bottom: 25px !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* 4. PRIMARY BUTTON STYLING (THE GOLDEN BUTTONS) */
    .stButton > button {
        background: linear-gradient(135deg, #D4AF37 0%, #B8860B 100%) !important;
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border: 2px solid #000000 !important;
        border-radius: 4px !important;
        padding: 0.8rem 2rem !important;
        width: 100% !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
        transition: all 0.4s ease-in-out !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
    }

    .stButton > button:hover {
        background: #FFD700 !important;
        color: #000000 !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 15px rgba(212, 175, 55, 0.4) !important;
        border: 2px solid #D4AF37 !important;
    }

    .stButton > button:active {
        transform: translateY(-1px) !important;
    }

    /* 5. INPUT FIELDS, TEXT AREAS, AND WIDGETS */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #1A1A1A !important;
        border: 2px solid #D4AF37 !important;
        border-radius: 8px !important;
        font-size: 1.1rem !important;
        padding: 15px !important;
        line-height: 1.6 !important;
    }

    .stTextArea textarea:focus {
        border-color: #B8860B !important;
        box-shadow: 0 0 10px rgba(212, 175, 55, 0.3) !important;
    }

    .stTextInput input {
        background-color: #FFFFFF !important;
        color: #1A1A1A !important;
        border: 2px solid #D4AF37 !important;
        border-radius: 5px !important;
    }

    /* 6. EXPANDERS AND CARDS */
    .stExpander {
        background-color: #FFFFFF !important;
        border: 1px solid #D4AF37 !important;
        border-radius: 8px !important;
        margin-bottom: 15px !important;
    }

    /* 7. SIMULATION CHAT BUBBLES AND BOXES */
    .simulation-turn-box {
        background: #FFFFFF !important;
        border-left: 10px solid #D4AF37 !important;
        padding: 25px !important;
        margin-bottom: 25px !important;
        border-radius: 0 12px 12px 0 !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.1) !important;
        color: #1A1A1A !important;
        line-height: 1.8 !important;
    }

    /* 8. SCROLLBAR CUSTOMIZATION */
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: #FAF9F6; }
    ::-webkit-scrollbar-thumb { background: #D4AF37; border-radius: 5px; }
    ::-webkit-scrollbar-thumb:hover { background: #B8860B; }

    /* 9. DIVIDERS AND HORIZONTAL RULES */
    hr {
        border-top: 2px solid #D4AF37 !important;
        margin: 2rem 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# --- DATA STRUCTURES AND ENCODING LOGIC ---
# ==============================================================================

class CustomEncoder(JSONEncoder):
    """
    CUSTOM JSON ENCODER TO HANDLE PYTORCH TENSORS DURING DATA TRANSFERS.
    """
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
# ==============================================================================
# --- RESOURCE INITIALIZATION AND CACHED LOADING ---
# ==============================================================================

# ==============================================================================
# --- RESOURCE INITIALIZATION WITH AUTOMATIC CUDA FALLBACK ---
# ==============================================================================

@st.cache_resource
def load_resources():
    """
    PERFORMS A SECURE LOAD OF THE AI MODEL AND THE LEGAL EMBEDDINGS.
    DYNAMICALLY DETECTS GPU (CUDA) AVAILABILITY AND FALLS BACK TO CPU IF UNSTABLE.
    """
    # 1. PATH VALIDATION: VERIFY DATA INTEGRITY BEFORE INITIALIZING TENSORS
    if not os.path.exists(INDEX_PATH) or not os.path.exists(EMBEDDINGS_PATH):
        st.error(f"FILE NOT FOUND: Check if {INDEX_PATH} is in the root folder.")
        st.info("System requires both the CSV index and the .pt embedding file.")
        return None, None, None

    # 2. DEVICE DETERMINATION LOGIC: HARNESSING NVIDIA GPU POWER
    # 
    device = "cpu"
    if torch.cuda.is_available():
        try:
            # EXECUTE A VOLATILE TENSOR OPERATION TO VERIFY CUDA KERNEL HEALTH
            test_tensor = torch.zeros(1).cuda()
            device = "cuda"
            st.sidebar.success("🚀 GPU ACCELERATION ACTIVE (CUDA)")
        except Exception:
            # FALLBACK MECHANISM: IF KERNELS ARE MISMATCHED, SWITCH TO CPU
            st.sidebar.warning("⚠️ CUDA DETECTED BUT UNSTABLE. FALLING BACK TO CPU.")
            device = "cpu"
    else:
        # DEFAULT TO CPU MODE FOR SYSTEMS WITHOUT NVIDIA HARDWARE
        st.sidebar.info("💻 RUNNING ON CPU MODE")

    try:
        # 3. LOAD THE CLASSIFIED METADATA INDEX VIA PANDAS
        # 
        df = pd.read_csv(INDEX_PATH)
        df.fillna('', inplace=True)
        
        # 4. INITIALIZE THE SENTENCE TRANSFORMER (SBERT) ON THE SELECTED DEVICE
        # 
        model = SentenceTransformer(MODEL_NAME, device=device)
        
        # 5. LOAD PRE-COMPUTED TENSORS (Map explicitly to the determined device)
        embeddings = torch.load(
            EMBEDDINGS_PATH, 
            map_location=torch.device(device)
        )
        
        st.success(f"SYSTEM READY: {len(df)} JUDGMENTS LOADED ON {device.upper()}")
        return df, model, embeddings
        
    except Exception as e:
        # CRITICAL ERROR CATCH: HANDLES MEMORY OVERFLOW OR CORRUPT FILES
        st.error(f"INTERNAL SYSTEM ERROR DURING RESOURCE LOAD: {str(e)}")
        return None, None, None

# ==============================================================================
# --- DUAL-ENGINE SEARCH LOGIC: KEYWORD + SEMANTIC ---
# ==============================================================================

def retrieve_relevant_cases(query, df, model, embeddings, top_k=TOP_K_RESULTS):
    """
    IMPLEMENTS A HYBRID RETRIEVAL STRATEGY TO MINIMIZE FALSE NEGATIVES.
    LAYER 1: BOOLEAN KEYWORD FILTERING (FOR EXACT MATCHES).
    LAYER 2: VECTOR SIMILARITY (FOR CONCEPTUAL LEGAL MATCHES).
    """
    if not query or not query.strip():
        return []

    # --- LAYER 1: KEYWORD MATCHING ENGINE ---
    # WE TOKENIZE THE QUERY TO ALLOW FOR OUT-OF-ORDER WORD MATCHING
    query_tokens = query.lower().split()
    keyword_filter = pd.Series([True] * len(df))
    
    # APPLY TOKEN-BASED FILTERING TO THE 'CASE_NAME_SIMPLE' COLUMN
    for token in query_tokens:
        if len(token) > 2:
            # ESCAPE REGEX CHARACTERS TO PREVENT QUERY INJECTION ERRORS
            escaped_token = re.escape(token)
            keyword_filter &= df['Case_Name_Simple'].str.lower().str.contains(escaped_token)

    exact_matches_df = df[keyword_filter]

    # IF EXACT KEYWORD MATCHES ARE FOUND, WE RETURN THEM IMMEDIATELY
    if not exact_matches_df.empty:
        results = []
        for i, (idx, row) in enumerate(exact_matches_df.head(top_k).iterrows()):
            results.append({
                'Score': 1.0 - (i * 0.01),
                'Case_Name': row['Case_Name_Simple'],
                'Year': row['Year'],
                'Full_Text': row['Full_Text'],
                'Category': row['Legal_Category']
            })
        if results:
            return results

    # --- LAYER 2: SEMANTIC VECTOR ENGINE ---
    # 
    if embeddings.numel() == 0:
        return []

    # TRANSFORM TEXT QUERY INTO A HIGH-DIMENSIONAL NUMERICAL VECTOR
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # MEASURE THE ANGULAR DISTANCE (COSINE SIMILARITY) BETWEEN VECTORS
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    
    # SORT AND EXTRACT THE TOP-K MOST RELEVANT JUDGMENTS
    top_results = torch.topk(cos_scores, k=min(top_k, len(embeddings)))
    
    results = []
    for score, index_tensor in zip(top_results[0], top_results[1]):
        original_idx = index_tensor.item()
        row = df.iloc[original_idx]
        results.append({
            'Score': score.item(),
            'Case_Name': row['Case_Name_Simple'],
            'Year': row['Year'],
            'Full_Text': row['Full_Text'],
            'Category': row['Legal_Category']
        })
    return results

# ==============================================================================
# --- GENERATIVE AI INTERFACE (GEMINI API GATEWAY) ---
# ==============================================================================

def generate_explanation_with_gemini(user_query, context, category, mode="Summary"):
    """
    HANDLES THE OUTBOUND COMMUNICATION WITH THE GENERATIVE AI MODEL.
    DYNAMICALLY INJECTS SYSTEM INSTRUCTIONS BASED ON THE ACTIVE MODULE.
    """
    # DEFINE BEHAVIORAL PROMPTS BASED ON THE SELECTED MODE
    if mode == "Deep Analysis":
        instr = "ANALYZE THE HISTORICAL CONTEXT AND LEGAL TESTS OF THE JUDGMENT."
        title = "**FINAL RATIONALE AND COMPREHENSIVE ANALYSIS:**"
        rag = "USE ONLY THE PROVIDED CONTEXT."
    elif mode == "Comparison":
        instr = "COMPARE THE NEW CASE AGAINST PRECEDENTS AND SUGGEST AN OUTCOME."
        title = "**FINAL COMPARATIVE LEGAL ANALYSIS:**"
        rag = "USE ONLY THE PROVIDED CONTEXT."
    elif mode == "Query Generation":
        instr = "GENERATE A 15-WORD SEMANTIC SEARCH QUERY FROM THE FACTS."
        title = ""
        rag = "USE ONLY THE PROVIDED CONTEXT."
    elif mode == "Advisory":
        instr = "ACT AS A LEGAL TUTOR. PROVIDE ADVICE BASED ON IPC GENERAL KNOWLEDGE."
        title = "**ADVISORY RESPONSE:**"
        rag = "USE GENERAL LEGAL PRINCIPLES."
        context = "" 
    elif mode == "Critique":
        instr = "CRITIQUE THE DOCUMENT FOR STRUCTURAL AND LEGAL FLAWS."
        title = "**LEGAL EDITOR CRITIQUE:**"
        rag = "USE SENIOR EDITOR PERSONA."
        context = "" 
    elif mode == "Simulator_Opponent":
        instr = "ACT AS OPPOSING COUNSEL. CONSTRUCT AN AGGRESSIVE COUNTER-ARGUMENT."
        title = "**OPPOSING COUNSEL'S ARGUMENT:**"
        rag = "USE CONTEXT FACTS."
    elif mode == "Simulator_Judge":
        instr = "ACT AS A SUPREME COURT JUSTICE. PROVIDE A FIRM RULING."
        title = "**COURT'S RULING AND DIRECTIVE:**"
        rag = "USE ARGUMENTS PROVIDED."
    else:
        instr = "SUMMARIZE THE KEY LEGAL PRINCIPLES."
        title = "**FINAL ANSWER:**"
        rag = "USE ONLY THE PROVIDED CONTEXT."

    # CONSTRUCT SYSTEM AND LLM PROMPTS
    system_prompt = (
        f"YOU ARE AN EXPERT LEGAL AI SPECIALIST IN {category}. "
        f"{instr} {rag}"
    )
    llm_prompt = f"QUERY: {user_query}\n\nCONTEXT:\n{context}\n\n{title}"
    
    # REQUEST PAYLOAD CONSTRUCTION
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": llm_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }
    
    # EXECUTION WITH RETRY MECHANISM
    json_payload = json.dumps(payload, cls=CustomEncoder)
    for attempt in range(3):
        try:
            response = requests.post(api_url, headers={'Content-Type': 'application/json'}, data=json_payload, timeout=30)
            response.raise_for_status()
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception:
            if attempt == 2: return "FAILED TO RETRIEVE RESPONSE FROM AI SERVICE."
            time.sleep(2)

# ==============================================================================
# --- FEATURE-SPECIFIC WRAPPERS AND CACHING ---
# ==============================================================================

@st.cache_data(show_spinner="Analyzing Case: Generating Rationale...")
def get_deep_analysis_result(user_query, case_full_text, case_name, case_year, max_chars, category):
    """
    WRAPS THE DEEP ANALYSIS LOGIC WITH STREAMLIT DATA CACHING.
    ENSURES THE MODEL DOES NOT RE-PROCESS THE SAME CASE MULTIPLE TIMES.
    """
    # 
    deep_context = f"CASE 1 ({case_name}, {case_year}): {case_full_text[:max_chars]}\n---\n"
    explanation = generate_explanation_with_gemini(user_query, deep_context, category, mode="Deep Analysis")
    return explanation

@st.cache_data(show_spinner="AI Pre-Processing: Extracting Legal Issues...")
def get_analysis_query(case_text, category):
    """
    EXTRACTS CORE LEGAL QUERIES FROM RAW CASE FACTS TO IMPROVE SEARCH ACCURACY.
    """
    query_output = generate_explanation_with_gemini(
        user_query="Generate a high-quality semantic search query based on the case facts.",
        context=f"New Case Text:\n{case_text}",
        category=category,
        mode="Query Generation"
    )
    return query_output.strip().replace('"', '').replace("'", "")

@st.cache_data(show_spinner="Senior Editor Review: Critiquing Document Structure...")
def get_document_critique(document_text):
    """
    PROVIDES A STRUCTURAL AND LEGAL CRITIQUE OF USER-PROVIDED DOCUMENTS.
    """
    critique_response = generate_explanation_with_gemini(
        user_query=document_text,
        context="",
        category="Senior Legal Editor",
        mode="Critique"
    )
    return critique_response

@st.cache_data(show_spinner="Consulting Legal Principles...")
def get_general_advisory(query):
    """
    GENERATES GENERAL LEGAL ADVICE BASED ON IPC AND JURISPRUDENCE.
    """
    advisory_response = generate_explanation_with_gemini(
        user_query=query,
        context="",
        category="Indian Jurisprudence",
        mode="Advisory"
    )
    return advisory_response

@st.cache_data(show_spinner="Running Court Proceeding...")
def run_simulation_turn(user_argument, full_case_text, party_role, simulation_history):
    """
    MANAGES THE ROLEPLAY LOGIC BETWEEN USER ARGUMENTS AND AI RESPONSES.
    """
    # 
    history_context = "\n---\n".join([f"{t['role']}: {t['text']}" for t in simulation_history])

    if simulation_history and simulation_history[-1]['role'] == 'User Argument':
        ai_mode = "Simulator_Judge"
        user_query = f"The User (as {party_role}) just argued: {user_argument}. Provide the Court's ruling."
    else:
        ai_mode = "Simulator_Opponent"
        user_query = f"The User (as {party_role}) is presenting: {user_argument}. Provide the counter-argument."
    
    full_context = (
        f"**Supreme Court Case Facts:** {full_case_text[:MAX_SIMULATION_CASE_CHARS]}\n"
        f"**Party Role:** The User is the '{party_role}'.\n"
        f"**Simulation History:**\n{history_context}"
    )

    ai_response = generate_explanation_with_gemini(
        user_query=user_query,
        context=full_context,
        category=DEFAULT_PERSONA,
        mode=ai_mode
    )
    return ai_mode, ai_response

# ==============================================================================
# --- SESSION STATE INITIALIZATION ---
# ==============================================================================

def initialize_session_state():
    """
    ENSURES ALL SESSION VARIABLES ARE DEFINED TO PREVENT KEYERRORS.
    """
    if 'active_analysis_id' not in st.session_state:
        st.session_state.active_analysis_id = None
    if 'user_query' not in st.session_state:
        st.session_state.user_query = "What are the legal tests for 'grave and sudden provocation'?"
    if 'retrieved_cases' not in st.session_state:
        st.session_state.retrieved_cases = []
    if 'search_performed' not in st.session_state:
        st.session_state.search_performed = False
    if 'comparison_step' not in st.session_state:
        st.session_state.comparison_step = 0
    if 'new_case_facts' not in st.session_state:
        st.session_state.new_case_facts = ""
    if 'advisory_response' not in st.session_state:
        st.session_state.advisory_response = ""
    if 'critique_response' not in st.session_state:
        st.session_state.critique_response = ""
    if 'document_to_critique' not in st.session_state:
        st.session_state.document_to_critique = ""
    if 'simulation_state' not in st.session_state:
        st.session_state.simulation_state = "Select Case"
    if 'simulation_case' not in st.session_state:
        st.session_state.simulation_case = None
    if 'party_role' not in st.session_state:
        st.session_state.party_role = "Appellant/Petitioner"
    if 'simulation_history' not in st.session_state:
        st.session_state.simulation_history = []

# ==============================================================================
# --- MAIN UI EXECUTION: SIDEBAR NAVIGATION ---
# ==============================================================================

# CALL INITIALIZATION AND CSS INJECTION
initialize_session_state()
inject_custom_css()
df, model, embeddings = load_resources()

# RENDER SIDEBAR LOGO AND BRANDING
with st.sidebar:
    st.markdown("<h1 style='color: #D4AF37; border:none; text-align:center;'>⚖️ ADVOTECH</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:white;'>EXPLAINABLE LEGAL AI (XLAI)</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 
    
    # NAVBAR NAVIGATION SELECTOR
    st.session_state.analysis_mode = st.radio(
        "SELECT SYSTEM MODULE:",
        (
            "🏠 Dashboard", 
            "🔍 Existing Case Deep Dive", 
            "🔄 New Case Comparison", 
            "📝 Document Critique", 
            "🎓 General Advisory", 
            "👨‍⚖️ AI Judge Simulator"
        ),
        index=0,
        key="main_nav"
    )

    st.markdown("---")
    st.info("System Status: Online")
    st.caption("Jurisprudence Database: Indian Supreme Court (2025 Edition)")
    
    # GLOBAL RESET BUTTON
    if st.button("🔄 RESET CURRENT SESSION"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# ==============================================================================
# --- MAIN PAGE HEADER AND ROUTING ---
# ==============================================================================

if df is not None:
    # 1. DASHBOARD MODULE
    if st.session_state.analysis_mode == "🏠 Dashboard":
        st.title("⚖️ ADVOTECH KNOWLEDGE ENGINE")
        st.markdown("### Professional Legal Suite for Supreme Court Jurisprudence")
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class='simulation-turn-box'>
            <h4>SEARCH ENGINE</h4>
            <p>Dual-layer semantic and keyword search across 7GB of classified judgments.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='simulation-turn-box'>
            <h4>DEEP ANALYSIS</h4>
            <p>Generative RAG logic to extract rationale, impact, and historical context.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class='simulation-turn-box'>
            <h4>COURT SIMULATOR</h4>
            <p>Interactive roleplay to test legal arguments against AI counsel and judges.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # ==============================================================================
# --- MODULE 1: EXISTING CASE DEEP DIVE (FEATURE 1) ---
# ==============================================================================

    if st.session_state.analysis_mode == "🔍 Existing Case Deep Dive":
        st.title("🔍 Case Analysis & Deep Dive")
        st.markdown("Query the historical corpus and generate deep-layered AI rationales.")
        
        # SEARCH INPUT CONTAINER
        with st.container():
            st.session_state.user_query = st.text_area(
                "❓ ENTER LEGAL QUESTION OR LANDMARK CASE NAME:",
                value=st.session_state.user_query,
                key="deep_dive_query_area",
                height=150
            )

            if st.button("RUN SEMANTIC RETRIEVAL", type="primary"):
                st.session_state.active_analysis_id = None
                with st.spinner("Executing Vector Search..."):
                    st.session_state.retrieved_cases = retrieve_relevant_cases(
                        st.session_state.user_query, df, model, embeddings
                    )
                st.session_state.search_performed = True
                st.rerun()

        # RESULTS DISPLAY LOGIC
        if st.session_state.search_performed and st.session_state.retrieved_cases:
            st.subheader(f"FOUND {len(st.session_state.retrieved_cases)} RELEVANT PRECEDENTS")
            st.markdown("---")
            
            for i, case in enumerate(st.session_state.retrieved_cases):
                case_id = i + 1
                unique_id = f"Analysis_{case_id}"
                
                # CASE METADATA HEADER
                st.markdown(f"### {case_id}. {case['Case_Name']} ({case['Year']})")
                st.caption(f"VECTOR SIMILARITY SCORE: {case['Score']:.4f} | CATEGORY: {case['Category']}")
                
                # COLUMNAR DISPLAY: TEXT VS ACTION
                col_text, col_action = st.columns([0.6, 0.4])

                with col_text:
                    with st.expander(f"📖 VIEW TEXT SEGMENT (CASE {case_id})"):
                        st.markdown(f"<div style='color:#1A1A1A;'>{case['Full_Text']}</div>", unsafe_allow_html=True)
                
                with col_action:
                    if st.button(f"🔎 ANALYZE CASE {case_id}", key=f"deep_btn_{i}"):
                        st.session_state.active_analysis_id = unique_id
                        st.session_state.analysis_case_data = {
                            'Case_Name': case['Case_Name'],
                            'Year': case['Year'],
                            'Full_Text': case['Full_Text']
                        }
                        st.rerun()

                # RAG GENERATION DISPLAY
                if st.session_state.active_analysis_id == unique_id:
                    case_data = st.session_state.analysis_case_data
                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.container():
                        st.markdown(f"<div class='simulation-turn-box'>", unsafe_allow_html=True)
                        st.subheader(f"💡 AI RATIONALE: {case_data['Case_Name']}")
                        
                        # EXECUTE GEMINI DEEP ANALYSIS
                        explanation = get_deep_analysis_result(
                            st.session_state.user_query,
                            case_data['Full_Text'],
                            case_data['Case_Name'],
                            case_data['Year'],
                            MAX_SINGLE_CASE_CHARS,
                            DEFAULT_PERSONA
                        )
                        st.markdown(explanation)
                        st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---")
        
        elif st.session_state.search_performed:
            st.warning("NO RESULTS FOUND. PLEASE REFRESH YOUR QUERY PARAMETERS.")

# ==============================================================================
# --- MODULE 2: NEW CASE COMPARISON (FEATURE 2) ---
# ==============================================================================

    elif st.session_state.analysis_mode == "🔄 New Case Comparison":
        st.title("🔄 New Case Facts Comparison")
        st.markdown("Analyze current fact patterns against historical Supreme Court benchmarks.")
        
        # FACT PATTERN INPUT
        st.session_state.new_case_facts = st.text_area(
            "📝 PASTE THE FULL FACTS OF THE NEW CASE:",
            value=st.session_state.new_case_facts,
            key="comparison_facts_input",
            height=300,
            placeholder="Example: The appellant was convicted under Section 302 IPC..."
        )
        
        # STEP 0: INITIALIZATION
        if st.session_state.comparison_step == 0:
            if st.button("INITIATE COMPARATIVE SEARCH", type="primary"):
                if st.session_state.new_case_facts.strip():
                    st.session_state.comparison_step = 1
                    st.rerun()
                else:
                    st.error("ERROR: CASE FACTS CANNOT BE EMPTY.")
        
        # STEP 1: QUERY GENERATION & SEARCH
        elif st.session_state.comparison_step == 1:
            # AI GENERATES A SEARCH QUERY FROM THE RAW FACTS
            search_query = get_analysis_query(
                st.session_state.new_case_facts[:MAX_NEW_CASE_INPUT_CHARS],
                DEFAULT_PERSONA
            )
            
            st.info(f"GENERATED SEMANTIC SEARCH QUERY: '{search_query}'")
            
            with st.spinner("Identifying Historical Precedents..."):
                precedents = retrieve_relevant_cases(search_query, df, model, embeddings)
            
            st.session_state.retrieved_precedents = precedents
            st.session_state.comparison_query = search_query
            st.session_state.comparison_step = 2
            st.rerun()

        # STEP 2: COMPARATIVE ANALYSIS
        elif st.session_state.comparison_step == 2:
            st.subheader("⚖️ IDENTIFIED HISTORICAL BENCHMARKS")
            
            precedents = st.session_state.retrieved_precedents
            if not precedents:
                st.error("NO COMPARABLE PRECEDENTS FOUND.")
                if st.button("RESTART ANALYSIS"):
                    st.session_state.comparison_step = 0
                    st.rerun()
            else:
                precedent_context = ""
                for i, case in enumerate(precedents):
                    st.markdown(f"**BENCHMARK {i+1}:** {case['Case_Name']} ({case['Year']})")
                    precedent_context += f"BENCHMARK {i+1} ({case['Case_Name']}): {case['Full_Text'][:MAX_CONTEXT_CHARS]}\n---\n"
                
                st.markdown("---")
                
                # FINAL COMPARISON CALL
                final_context = (
                    f"NEW CASE FACTS:\n{st.session_state.new_case_facts[:MAX_CONTEXT_CHARS]}\n\n"
                    f"HISTORICAL PRECEDENTS:\n{precedent_context}"
                )
                
                with st.spinner("Generating Comparative Legal Report..."):
                    comparison_report = generate_explanation_with_gemini(
                        user_query="COMPARE NEW CASE AGAINST HISTORICAL BENCHMARKS.",
                        context=final_context,
                        category=DEFAULT_PERSONA,
                        mode="Comparison"
                    )

                st.subheader("💡 COMPARATIVE LEGAL ANALYSIS REPORT")
                st.markdown(f"<div class='simulation-turn-box'>{comparison_report}</div>", unsafe_allow_html=True)
                
                if st.button("NEW COMPARISON"):
                    st.session_state.comparison_step = 0
                    st.session_state.new_case_facts = ""
                    st.rerun()

# ==============================================================================
# --- MODULE 3: DOCUMENT CRITIQUE (FEATURE 3) ---
# ==============================================================================

    elif st.session_state.analysis_mode == "📝 Document Critique":
        st.title("📝 Legal Document Critique & Review")
        st.markdown("Upload or paste draft documents for structural and completeness analysis.")
        
        # DOCUMENT INPUT
        st.session_state.document_to_critique = st.text_area(
            "📑 PASTE LEGAL DRAFT FOR REVIEW:",
            value=st.session_state.document_to_critique,
            key="critique_doc_area",
            height=400,
            placeholder="Paste your draft contract, pleading, or memorial section here..."
        )
        
        if st.button("EXECUTE SENIOR EDITOR REVIEW", type="primary"):
            if st.session_state.document_to_critique.strip():
                st.session_state.critique_response = get_document_critique(
                    st.session_state.document_to_critique
                )
                st.rerun()
            else:
                st.warning("WARNING: NO CONTENT DETECTED FOR REVIEW.")
        
        # DISPLAY CRITIQUE OUTPUT
        if st.session_state.critique_response:
            st.markdown("---")
            st.subheader("💡 SENIOR EDITOR'S ASSESSMENT")
            st.markdown(f"<div class='simulation-turn-box'>{st.session_state.critique_response}</div>", unsafe_allow_html=True)
            
            if st.button("CLEAR CRITIQUE"):
                st.session_state.critique_response = ""
                st.session_state.document_to_critique = ""
                st.rerun()
    
    # ==============================================================================
# --- MODULE 4: GENERAL LEGAL ADVISORY (FEATURE 4) ---
# ==============================================================================

    elif st.session_state.analysis_mode == "🎓 General Advisory":
        st.title("🎓 General Legal Advisory & Tutoring")
        st.markdown("Consult on broad legal concepts, definitions, or procedural rules.")
        
        # USER INPUT FOR GENERAL LEGAL QUERY
        advisory_query = st.text_area(
            "❓ ENTER YOUR LEGAL QUESTION (e.g., 'What is the doctrine of Res Judicata?'):",
            key="advisory_query_area",
            height=150,
            placeholder="Describe the legal principle or procedural doubt you have..."
        )
        
        # EXECUTION BUTTON FOR ADVISORY
        if st.button("CONSULT JURISPRUDENCE", type="primary"):
            if advisory_query.strip():
                # RESET PREVIOUS RESPONSE CACHE TO ENSURE FRESH ADVICE
                get_general_advisory.clear()
                st.session_state.advisory_response = get_general_advisory(advisory_query)
                st.rerun()
            else:
                st.warning("WARNING: PLEASE ENTER A VALID LEGAL QUERY.")
        
        # DISPLAY THE AI GENERATED ADVISORY
        if st.session_state.advisory_response:
            st.markdown("---")
            st.subheader("💡 ADVISORY RESPONSE")
            st.markdown(
                f"<div class='simulation-turn-box'>{st.session_state.advisory_response}</div>", 
                unsafe_allow_html=True
            )
            
            # SUB-ADVISORY METADATA
            st.caption("Note: This response is for educational purposes and does not constitute formal legal advice.")

# ==============================================================================
# --- MODULE 5: AI JUDGE SIMULATOR (FEATURE 5) ---
# ==============================================================================

    elif st.session_state.analysis_mode == "👨‍⚖️ AI Judge Simulator":
        st.title("👨‍⚖️ Supreme Court AI Simulator")
        st.markdown("Argue your case against AI-powered Counsel and receive rulings from an AI Justice.")
        
        # STAGE 1: CASE SELECTION FOR SIMULATION
        if st.session_state.simulation_state == "Select Case":
            st.subheader("1. INITIALIZE COURT PROCEEDING")
            
            sim_query = st.text_input(
                "ENTER LANDMARK CASE NAME TO SIMULATE:",
                placeholder="Example: Kesavananda Bharati v. State of Kerala",
                key="sim_case_input"
            )

            if st.button("LOAD CASE FOR SIMULATION", type="primary"):
                if sim_query.strip():
                    with st.spinner("Retrieving Case Record for Context..."):
                        # RETRIEVE THE MOST RELEVANT CASE TO ACT AS THE SIMULATION FOUNDATION
                        sim_results = retrieve_relevant_cases(sim_query, df, model, embeddings, top_k=1)
                    
                    if sim_results:
                        st.session_state.simulation_case = sim_results[0]
                        st.session_state.simulation_state = "Select Party"
                        st.rerun()
                    else:
                        st.error("ERROR: CASE NOT FOUND. PLEASE TRY A KNOWN LANDMARK JUDGMENT.")
        
        # STAGE 2: ROLE SELECTION
        elif st.session_state.simulation_state == "Select Party":
            st.subheader("2. SELECT STANDING AND ROLE")
            
            selected_case = st.session_state.simulation_case
            st.info(f"PROCEEDING LOADED: **{selected_case['Case_Name']} ({selected_case['Year']})**")
            
            st.session_state.party_role = st.selectbox(
                "SELECT YOUR ROLE IN THE PROCEEDING:",
                ("Appellant/Petitioner", "Respondent/Defendant"),
                key="party_selection_dropdown"
            )
            
            if st.button("BEGIN FORMAL HEARING", type="primary"):
                # INITIALIZE THE SIMULATION HISTORY WITH A SYSTEM START NOTIFICATION
                st.session_state.simulation_state = "Active Hearing"
                st.session_state.simulation_history.append({
                    'role': 'System', 
                    'text': f"Simulation started. Proceeding as the {st.session_state.party_role}."
                })
                
                # TRIGGER AN INITIAL RESPONSE FROM THE COURT TO OPEN THE HEARING
                initial_prompt = f"The Court has convened. State your opening arguments for {selected_case['Case_Name']}."
                ai_mode, ai_resp = run_simulation_turn(
                    initial_prompt, selected_case['Full_Text'], 
                    st.session_state.party_role, st.session_state.simulation_history
                )
                st.session_state.simulation_history.append({'role': ai_mode, 'text': ai_resp})
                st.rerun()

        # STAGE 3: ACTIVE HEARING LOOP
        elif st.session_state.simulation_state == "Active Hearing":
            current_case = st.session_state.simulation_case
            st.title(f"👨‍⚖️ HEARING: {current_case['Case_Name']}")
            st.markdown(f"**YOUR ROLE:** {st.session_state.party_role}")
            
            # 
            
            # DISPLAY THE FULL TRANSCRIPT OF THE PROCEEDING
            st.markdown("### COURT TRANSCRIPT")
            with st.container():
                for turn in st.session_state.simulation_history:
                    role_label = ""
                    if turn['role'] == 'User Argument':
                        role_label = f"**COUNSEL FOR {st.session_state.party_role.upper()} (YOU):**"
                    elif turn['role'] == 'Simulator_Opponent':
                        role_label = "**OPPOSING COUNSEL:**"
                    elif turn['role'] == 'Simulator_Judge':
                        role_label = "**THE COURT (AI JUSTICE):**"
                    else:
                        role_label = ""

                    st.markdown(
                        f"<div class='simulation-turn-box'>{role_label}<br>{turn['text']}</div>", 
                        unsafe_allow_html=True
                    )
            
            # USER ARGUMENT INPUT AREA
            with st.form(key="argument_form", clear_on_submit=True):
                user_arg_input = st.text_area("PRESENT YOUR ARGUMENT OR RESPONSE:", height=150)
                submit_arg = st.form_submit_button("SUBMIT TO COURT")
                
                if submit_arg:
                    if user_arg_input.strip():
                        # LOG THE USER ARGUMENT
                        st.session_state.simulation_history.append({
                            'role': 'User Argument', 
                            'text': user_arg_input
                        })
                        
                        # GENERATE AI COUNTER-ARGUMENT OR JUDICIAL RULING
                        ai_type, ai_reply = run_simulation_turn(
                            user_arg_input, current_case['Full_Text'], 
                            st.session_state.party_role, st.session_state.simulation_history
                        )
                        st.session_state.simulation_history.append({'role': ai_type, 'text': ai_reply})
                        st.rerun()
            
            if st.button("🔚 CONCLUDE PROCEEDING"):
                st.session_state.simulation_state = "Select Case"
                st.session_state.simulation_history = []
                st.rerun()

# ==============================================================================
# --- SYSTEM FOOTER AND TERMINATION LOGIC ---
# ==============================================================================

else:
    # DATA LOADING FAILURE STATE
    st.error("CRITICAL SYSTEM ERROR: DATABASE NOT DETECTED.")
    st.info("Check INDEX_PATH and EMBEDDINGS_PATH configurations.")

# FINAL PAGE DIVIDER
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888888; font-size: 0.9rem;'>"
    "ADVOTECH XLAI JURISPRUDENCE ENGINE © 2026 | PROPRIETARY LEGAL TECHNOLOGY"
    "</div>", 
    unsafe_allow_html=True
)

# ==============================================================================
# --- END OF ADVOTECH XLAI SOURCE CODE (TOTAL 1000 LINES) ---
# ==============================================================================