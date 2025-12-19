import os
import json
import csv
import re
import time
import requests
from typing import List, Dict, Any, Set
from datetime import datetime

from pypdf import PdfReader

# ---------------------------
# Configurations
# ---------------------------

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-chat-v3.1"
SLEEP_BETWEEN_REQUESTS = 5  # seconds
MAX_PAGES_TO_EXTRACT = 4  # Only extract first N pages

def log(message: str):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def setup_openrouter_api():
    """Setup OpenRouter API authentication"""
    log(f"OpenRouter API configured")
    log(f"Model: {MODEL}")
    log(f"Base URL: {OPENROUTER_BASE_URL}")
    return OPENROUTER_API_KEY


def extract_text_from_pdf(pdf_path: str, max_pages: int = MAX_PAGES_TO_EXTRACT) -> str:
    """Extract text from first N pages of PDF using pypdf"""
    log(f"Extracting text from first {max_pages} pages of PDF...")
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        pages_to_extract = min(max_pages, total_pages)
        
        log(f"PDF has {total_pages} pages, extracting first {pages_to_extract}")
        
        all_text = []
        for page_num in range(pages_to_extract):
            try:
                text = reader.pages[page_num].extract_text()
                if text and text.strip():
                    all_text.append(f"=== PAGE {page_num + 1} ===\n{text.strip()}\n")
                    log(f"  Page {page_num + 1}: {len(text)} chars extracted")
                else:
                    log(f"  Page {page_num + 1}: No text (may be scanned/image)")
            except Exception as e:
                log(f"  Page {page_num + 1}: Error - {e}")
        
        full_text = "\n".join(all_text)
        log(f"Total extracted: {len(full_text)} characters from {len(all_text)} pages")
        
        return full_text
    except Exception as e:
        log(f"Error reading PDF: {e}")
        return ""


def call_deepseek_with_text(api_key: str, text_content: str, prompt: str, system_prompt: str, temperature: float = 0.3) -> str:
    """Call DeepSeek API via OpenRouter with text content"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/clinical-trial-analyzer",
        "X-Title": "Clinical Trial Protocol Analyzer"
    }
    
    # Build message with text content
    full_prompt = f"{prompt}\n\n--- PROTOCOL TEXT (First {MAX_PAGES_TO_EXTRACT} pages) ---\n\n{text_content}"
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": full_prompt
        }
    ]
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    
    log(f"Sending text to OpenRouter (DeepSeek v3)...")
    log(f"Text content length: {len(text_content)} characters")
    log(f"Request payload size: {len(json.dumps(payload)) // 1024} KB")
    
    start_time = time.time()
    response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=600)
    elapsed_time = time.time() - start_time
    
    log(f"Response received in {elapsed_time:.2f} seconds")
    log(f"Response status: {response.status_code}")
    
    if response.status_code != 200:
        log(f"ERROR - Response body: {response.text[:500]}")
        response.raise_for_status()
    
    raw_response = response.json()
    
    # Log usage info if available
    if "usage" in raw_response:
        usage = raw_response["usage"]
        log(f"Tokens used - Prompt: {usage.get('prompt_tokens', 'N/A')}, Completion: {usage.get('completion_tokens', 'N/A')}, Total: {usage.get('total_tokens', 'N/A')}")
    
    if "error" in raw_response:
        log(f"API Error: {raw_response['error']}")
        raise Exception(f"API Error: {raw_response['error']}")
    
    content = raw_response["choices"][0]["message"]["content"]
    log(f"Response content length: {len(content)} characters")
    
    return content


def get_user_processing_parameters():
    """Get user input for processing parameters."""
    print("\n" + "="*60)
    print("PDF Processing Configuration")
    print("="*60)
    
    # Get starting index
    while True:
        try:
            start_idx = int(input("Enter starting index (0-based, default 0): ").strip() or "0")
            if start_idx >= 0:
                break
            else:
                print("Starting index must be >= 0")
        except ValueError:
            print("Please enter a valid number")
    
    # Get number of PDFs to process
    while True:
        try:
            num_pdfs_input = input("Enter number of PDFs to process (default: all remaining): ").strip()
            if num_pdfs_input == "":
                num_pdfs = None  # Process all
                break
            num_pdfs = int(num_pdfs_input)
            if num_pdfs > 0:
                break
            else:
                print("Number of PDFs must be > 0")
        except ValueError:
            print("Please enter a valid number")
    
    # Get output CSV filename
    csv_filename = input("Enter output CSV filename (default: clinical_trial_results.csv): ").strip()
    if not csv_filename:
        csv_filename = "clinical_trial_results.csv"
    elif not csv_filename.endswith('.csv'):
        csv_filename += '.csv'
    
    print(f"\nConfiguration:")
    print(f"  Starting index: {start_idx}")
    print(f"  Number of PDFs: {num_pdfs if num_pdfs else 'All remaining'}")
    print(f"  Output CSV: {csv_filename}")
    
    confirm = input("\nProceed with this configuration? (y/n, default y): ").strip().lower()
    if confirm in ['n', 'no']:
        print("Processing cancelled.")
        return None
    
    return {
        'start_index': start_idx,
        'num_pdfs': num_pdfs,
        'csv_filename': csv_filename
    }


def get_processed_filenames(csv_filename: str) -> Set[str]:
    """Read existing CSV and return set of already-processed filenames."""
    processed = set()
    if os.path.exists(csv_filename):
        try:
            with open(csv_filename, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'filename' in row and row['filename']:
                        processed.add(row['filename'])
            print(f"Found {len(processed)} already-processed files in {csv_filename}")
        except Exception as e:
            print(f"Warning: Could not read existing CSV: {e}")
    return processed


def initialize_csv_if_needed(output_csv_name: str) -> None:
    """Create CSV with headers if it doesn't exist."""
    if os.path.exists(output_csv_name):
        return
    
    windows = ['screening_window', '1_month', '3_months', '6_months', '9_months', '12_months']
    categories = ['core_treatment', 'imaging_diagnostics', 'labs', 'clinic_visits']

    headers = ['filename', 'arm_name', 'intervention_type']
    headers += [w for w in windows]
    for cat in categories:
        for w in windows:
            headers.append(f"{cat}_{w}")
    headers += ['cycle_length_days', 'treatment_duration_rule', 'visit_pattern', 'assumptions']

    with open(output_csv_name, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
    print(f"Created new CSV file: {output_csv_name}")


def append_result_to_csv(result: Dict[str, Any], output_csv_name: str) -> None:
    """Append a single result (all arms) to CSV immediately after processing."""
    windows = ['screening_window', '1_month', '3_months', '6_months', '9_months', '12_months']
    categories = ['core_treatment', 'imaging_diagnostics', 'labs', 'clinic_visits']

    def get(d, *path, default=''):
        cur = d
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    with open(output_csv_name, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        filename = result.get('filename', '')
        for arm_data in result.get('arms', []):
            row = [filename, arm_data.get('arm_name', ''), arm_data.get('intervention_type', '')]
            for w in windows:
                row.append(get(arm_data, 'healthcare_contact_days', w, default=''))
            for cat in categories:
                for w in windows:
                    row.append(get(arm_data, 'category_breakdown', cat, w, default=''))
            row.extend([
                get(arm_data, 'extraction_notes', 'cycle_length_days', default=''),
                get(arm_data, 'extraction_notes', 'treatment_duration_rule', default=''),
                get(arm_data, 'extraction_notes', 'visit_pattern', default=''),
                '; '.join(get(arm_data, 'extraction_notes', 'assumptions', default=[])) if isinstance(get(arm_data, 'extraction_notes', 'assumptions', default=[]), list) else get(arm_data, 'extraction_notes', 'assumptions', default='')
            ])
            writer.writerow(row)
    
    print(f"✓ Saved {len(result.get('arms', []))} arm(s) from {filename} to CSV")


def get_multiple_pdfs_from_summaries(start_index=0, num_pdfs=None, processed_filenames: Set[str] = None):
    """Get multiple PDFs from summaries folder for batch processing with custom parameters."""
    summaries_folder = os.path.join(os.getcwd(), "summaries")
    pdf_files = []
    
    if not os.path.exists(summaries_folder):
        print(f"Summaries folder not found: {summaries_folder}")
        return pdf_files

    if processed_filenames is None:
        processed_filenames = set()

    all_files = []
    for filename in os.listdir(summaries_folder):
        if filename.lower().endswith('.pdf'):
            all_files.append(filename)
    
    all_files.sort()
    print(f"Found {len(all_files)} total PDFs in summaries folder")
    
    if start_index >= len(all_files):
        print(f"Error: Starting index {start_index} is beyond available files (max index: {len(all_files)-1})")
        return pdf_files
    
    selected_files = all_files[start_index:]
    
    unprocessed_files = []
    for filename in selected_files:
        pmid = extract_pmid_from_filename(filename)
        if pmid not in processed_filenames:
            unprocessed_files.append(filename)
        else:
            print(f"Skipping already-processed: {filename}")
    
    if num_pdfs is not None:
        unprocessed_files = unprocessed_files[:num_pdfs]
    
    for filename in unprocessed_files:
        file_path = os.path.join(summaries_folder, filename)
        pdf_files.append((filename, file_path))  # Changed: store path instead of content
        print(f"Selected PDF: {filename}")

    print(f"\nTotal PDFs to process: {len(pdf_files)} (from index {start_index})")
    return pdf_files


def extract_pmid_from_filename(filename: str) -> str:
    """Extract PMID number from filename if present."""
    m = re.search(r'PMID\s+(\d+)', filename, flags=re.IGNORECASE)
    return m.group(1) if m else filename


def _strip_code_fences(text: str) -> str:
    """Remove common Markdown code fences around JSON/text."""
    # Handle ```json ... ``` blocks (with optional language specifier)
    text = re.sub(r"```(?:json)?\s*\n?(.*?)```", r"\1", text, flags=re.DOTALL | re.IGNORECASE)
    # Handle '''json ... ''' blocks (single quotes)
    text = re.sub(r"'''(?:json)?\s*\n?(.*?)'''", r"\1", text, flags=re.DOTALL | re.IGNORECASE)
    # Handle """json ... """ blocks (double quotes)
    text = re.sub(r'"""(?:json)?\s*\n?(.*?)"""', r"\1", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove any remaining fence markers
    text = text.replace("```", "").replace("'''", "").replace('"""', "")
    return text.strip()


def parse_json_response(response_text: str) -> List[Dict[str, Any]]:
    """Parse one or more JSON objects for arms from the model response."""
    text = _strip_code_fences(response_text)

    def _validate(obj: Any) -> bool:
        return isinstance(obj, dict) and 'arm_name' in obj and 'healthcare_contact_days' in obj

    # First, try to find JSON array starting with [ and ending with ]
    array_match = re.search(r'\[\s*\{.*\}\s*\]', text, flags=re.DOTALL)
    if array_match:
        try:
            data = json.loads(array_match.group(0))
            if isinstance(data, list):
                valid_items = [d for d in data if _validate(d)]
                if valid_items:
                    return valid_items
        except Exception:
            pass

    # Try parsing the whole cleaned text as JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if _validate(d)]
        if _validate(data):
            return [data]
    except Exception:
        pass

    # Try to find JSON objects individually (nested brace matching)
    objs = []
    # Find all potential JSON object starts
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Try to find matching closing brace
            brace_count = 0
            start = i
            for j in range(i, len(text)):
                if text[j] == '{':
                    brace_count += 1
                elif text[j] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        chunk = text[start:j+1]
                        try:
                            candidate = json.loads(chunk)
                            if _validate(candidate):
                                objs.append(candidate)
                        except Exception:
                            pass
                        break
        i += 1
    
    return objs


def _validate_category_sums(arm: Dict[str, Any]) -> None:
    """Warn if category sums don't match total contact day counts for any window."""
    windows = ['screening_window', '1_month', '3_months', '6_months', '9_months', '12_months']
    categories = ['core_treatment', 'imaging_diagnostics', 'labs', 'clinic_visits']

    totals = arm.get('healthcare_contact_days', {}) or {}
    breakdown = arm.get('category_breakdown', {}) or {}

    for w in windows:
        try:
            expected = int(totals.get(w))
        except Exception:
            expected = None
        if expected is None:
            continue
        s = 0
        missing_cat = False
        for c in categories:
            try:
                v = breakdown.get(c, {}).get(w)
                s += int(v) if v is not None and str(v).strip() != '' else 0
            except Exception:
                missing_cat = True
        if not missing_cat and expected != s:
            print(f"WARNING: Arm '{arm.get('arm_name','?')}', window '{w}' -> total={expected} but category sum={s}.")


SYSTEM_PROMPT = """role: "Expert Clinical Trial Protocol Analyst"

primary_objective: |
  Extract the exact number of in-person healthcare contact days per treatment arm
  across six fixed time intervals from clinical trial protocols.

core_definitions:
  healthcare_contact_day:
    definition: |
      Any calendar date requiring the participant to physically attend a healthcare
      facility or meet face-to-face with study staff for trial-mandated procedures.
    counting_rule: "Count 1 per unique date, regardless of how many procedures occur that day."
  not_contact_days:
    description: "The following do NOT count as contact days:"
    exclusions:
      - "Tele-visits or phone calls"
      - "Home diaries or self-administered assessments"
      - "Couriered drug deliveries"
      - "Truly optional ('as needed') visits"
  anchor_point:
    day_zero: "Cycle 1 Day 1 (C1D1)"
    definition: "The first on-treatment visit OR first dose administration, whichever occurs earlier."
  time_windows:  # All bounds are inclusive
    screening_window:
      range: "Day -28 to Day -1 (inclusive)"
      start: -28
      end: -1
    "1_month":
      range: "Day 0 to Day +30 (inclusive)"
      start: 0
      end: 30
    "3_months":
      range: "Day 0 to Day +90 (inclusive)"
      start: 0
      end: 90
    "6_months":
      range: "Day 0 to Day +180 (inclusive)"
      start: 0
      end: 180
    "9_months":
      range: "Day 0 to Day +270 (inclusive)"
      start: 0
      end: 270
    "12_months":
      range: "Day 0 to Day +365 (inclusive)"
      start: 0
      end: 365

extraction_algorithm:
  note: "Think through this process silently. Do NOT output your chain of thought."
  defaults:
    cycle_length_days: 28
    treatment_duration_rule: "until progression"
    horizon_day: 365
    cycle_length_overrides:
      - pattern: "presence of a recurring mid-cycle 'Day 14–21' column or equivalent across cycles"
        set_cycle_length_days: 21
  steps:
    - name: "Isolate Treatment Arms"
      action: "Identify each treatment arm's Schedule of Events (SoE) or equivalent table."
    - name: "Build Visit Pattern"
      actions:
        - "List every column that represents an in-person visit (e.g., 'Screening', 'Randomisation', 'Day 1', 'Day 14–21')."
        - "For rows labelled 'SUBSEQUENT CYCLES' (or similar), record which visit columns persist; if a column is absent, assume that visit does NOT occur for cycles ≥ 2."
    - name: "Identify In-Person Visit Columns"
      actions:
        - "Mark ANY column as an in-person visit if required in-person assessments appear in its rows (e.g., vitals, physical exam, hematology/biochemistry, ECG/ECHO, imaging), even if no drug administration occurs in that column."
        - "Columns labeled with ranges (e.g., 'Day 14–21') represent ONE in-person visit scheduled within that window unless the row text explicitly indicates multi-day treatment (e.g., 'Days 1–5 radiation')."
    - name: "Infer Cycle Length"
      actions:
        - "If mid-cycle columns are labeled 'Day 14–21' (or equivalent) consistently, set cycle_length_days=21 unless the protocol states otherwise."
    - name: "Create Canonical Schedule"
      components:
        baseline_visits: "Fixed dates between Day -28 and Day -1."
        cycle_1_visits: "Exact offsets from Day 0 (e.g., Day 0, +7, +14)."
        repeating_cycles: "Apply offsets every cycle_length_days (default 28; use inferred override if present)."
    - name: "Enumerate Calendar Dates"
      actions:
        - "Enumerate all nominal visit dates through Day +365."
        - "Assume regimen continues 'until progression' unless the protocol states otherwise."
    - name: "Count Unique Dates"
      actions:
        - "Count unique calendar dates falling within each time window."
        - "Include partial cycles: if a visit's nominal date falls inside a window, count it."
    - name: "Quality Check"
      actions:
        - "Independently recompute counts via a second method; if disagreement, reconcile before answering."

special_rules:
  baseline_split:
    condition: "If 'Screening' and 'Randomisation' appear as separate columns"
    action: "Treat them as two distinct days"
    exception: "Only consolidate if an SoE footnote explicitly allows same-day completion."
  conditional_optional_visits:
    exclude_if_labeled:
      - "Optional"
      - "If clinically indicated"
      - "As needed"
    include_if_drug_continues: "If conditional on continuing a drug, assume the drug continues unless stated otherwise."
  multi_day_blocks:
    example: "'Days 1–5' radiotherapy = 5 separate contact days"
    rule: "Count only those days that fall inside the target time window."
  range_as_single_visit:
    rule: "A column labeled as a day-range (e.g., 'Day 14–21') is a single mandatory in-person visit scheduled within that window, unless the text states multi-day administration (then apply multi-day rules)."
    nominal_day_choice: "Use the earliest day in the range for calendarization (e.g., Day 14) unless a footnote specifies a different nominal day."
  mandatory_mid_cycle_visits:
    critical: true
    description: |
      For any cycle where a non-drug column (e.g., 'Day 14–21') includes required in-person assessments
      (vitals, labs, physical exam, AE checks, ECG/ECHO, imaging), COUNT it as a separate healthcare-contact day.
      This applies even when active treatment is given only on Day 1.
    validation: "If an SoE shows both Day 1 and mid-cycle in-person assessments, per-cycle contact days MUST be ≥ 2 unless mid-cycle is explicitly optional."
  ambiguity_handling:
    approach: "Use the most conservative lower estimate."
    requirement: "Document every assumption in extraction_notes."
  tie_breaker_precedence:
    rule: "Assign each in-person calendar day to EXACTLY ONE primary category; if multiple activities occur, use precedence:"
    hierarchy:
      - "core_treatment"
      - "imaging_diagnostics"
      - "labs"
      - "clinic_visits"

category_classification:
  categories:
    core_treatment:
      description: "Active treatment delivery (chemo, immunotherapy, targeted therapy, infusions, radiation fractions, interventional procedures as part of treatment, port care related to treatment, surgery)."
    imaging_diagnostics:
      description: "Imaging (CT/MRI/PET/US/X-ray), ECG/EKG, echocardiogram, diagnostic biopsies for assessment, pulmonary function tests, stress tests, other diagnostics."
    labs:
      description: "Phlebotomy visits and urine collections requiring in-person attendance."
    clinic_visits:
      description: "H&P, AE checks, vitals, weight, questionnaires/ePRO in clinic, nurse/physician visits, consent updates, counseling, education sessions."

output_format:
  requirements:
    - "Return a single JSON array."
    - "One object per treatment arm."
    - "OUTPUT ONLY THE JSON (no additional text)."
    - "Within each window, category subtotals must equal total healthcare_contact_days."
  json_structure:
    type: "array"
    items:
      - arm_name:
          type: "string"
          description: "Exact arm name from the protocol."
        intervention_type:
          type: "string"
          enum: ["intervention", "control"]
        healthcare_contact_days:
          screening_window: { type: "integer" }
          "1_month": { type: "integer" }
          "3_months": { type: "integer" }
          "6_months": { type: "integer" }
          "9_months": { type: "integer" }
          "12_months": { type: "integer" }
        category_breakdown:
          core_treatment:
            screening_window: { type: "integer" }
            "1_month": { type: "integer" }
            "3_months": { type: "integer" }
            "6_months": { type: "integer" }
            "9_months": { type: "integer" }
            "12_months": { type: "integer" }
          imaging_diagnostics:
            screening_window: { type: "integer" }
            "1_month": { type: "integer" }
            "3_months": { type: "integer" }
            "6_months": { type: "integer" }
            "9_months": { type: "integer" }
            "12_months": { type: "integer" }
          labs:
            screening_window: { type: "integer" }
            "1_month": { type: "integer" }
            "3_months": { type: "integer" }
            "6_months": { type: "integer" }
            "9_months": { type: "integer" }
            "12_months": { type: "integer" }
          clinic_visits:
            screening_window: { type: "integer" }
            "1_month": { type: "integer" }
            "3_months": { type: "integer" }
            "6_months": { type: "integer" }
            "9_months": { type: "integer" }
            "12_months": { type: "integer" }
        extraction_notes:
          assumptions:
            type: "array"
            items: { type: "string" }
            description: "Bullet list of assumptions or judgments made."
          visit_pattern:
            type: "string"
            description: "Enumerate the visit columns counted as in-person (e.g., 'Screening', 'C1D1', 'C1D14–21', ... 'CkD1', 'CkD14–21') and state that range-columns were mapped to one nominal date."
          cycle_length_days:
            type: "integer"
          treatment_duration_rule:
            type: "string"
            example: "until progression"

example_json: |
  [
    {
      "arm_name": "<exact arm name from the protocol>",
      "intervention_type": "<intervention or control>",
      "healthcare_contact_days": {
        "screening_window": 0,
        "1_month": 0,
        "3_months": 0,
        "6_months": 0,
        "9_months": 0,
        "12_months": 0
      },
      "category_breakdown": {
        "core_treatment": {
          "screening_window": 0, "1_month": 0, "3_months": 0, "6_months": 0, "9_months": 0, "12_months": 0
        },
        "imaging_diagnostics": {
          "screening_window": 0, "1_month": 0, "3_months": 0, "6_months": 0, "9_months": 0, "12_months": 0
        },
        "labs": {
          "screening_window": 0, "1_month": 0, "3_months": 0, "6_months": 0, "9_months": 0, "12_months": 0
        },
        "clinic_visits": {
          "screening_window": 0, "1_month": 0, "3_months": 0, "6_months": 0, "9_months": 0, "12_months": 0
        }
      },
      "extraction_notes": {
        "assumptions": ["<assumption 1>", "<assumption 2>"],
        "visit_pattern": "<concise description>",
        "cycle_length_days": 21,
        "treatment_duration_rule": "until progression"
      }
    }
  ]

final_validation_checklist:
  note: "Verify silently before submitting."
  checks:
    - "Each calendar date counted exactly once (no duplicates)."
    - "Category precedence applied correctly (core_treatment > imaging_diagnostics > labs > clinic_visits)."
    - "Within each window, category sums equal total healthcare_contact_days."
    - "Per-cycle sanity check: If cycle_length_days=21 and mid-cycle in-person assessments are present, verify ≥2 contact days per cycle through Day +365; if not, recompute and record an assumption explaining why."
    - "Category distribution check: If totals are dominated by 'core_treatment' when mid-cycle assessments exist, recompute to ensure mid-cycle contacts are counted."
    - "All assumptions documented in extraction_notes."
    - "JSON is valid and properly formatted."

instruction: "NOW PROCEED WITH THE ANALYSIS OF THE PROVIDED CLINICAL TRIAL PROTOCOL"
"""


def process_single_pdf(api_key: str, filename: str, pdf_path: str, file_index: int, total_files: int):
    """Process a single PDF file and return a result entry or None."""
    log(f"")
    log(f"{'='*60}")
    log(f"PROCESSING FILE {file_index}/{total_files}: {filename}")
    log(f"{'='*60}")
    log(f"PDF path: {pdf_path}")
    
    # Check file exists and get size
    if not os.path.exists(pdf_path):
        log(f"ERROR: File not found: {pdf_path}")
        return None
    
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    log(f"File size: {file_size_mb:.2f} MB")

    try:
        # Extract text from first N pages
        log("")
        log("[Step 1] Extracting text from PDF...")
        extracted_text = extract_text_from_pdf(pdf_path, MAX_PAGES_TO_EXTRACT)
        
        if not extracted_text or len(extracted_text.strip()) < 100:
            log("ERROR: Could not extract sufficient text from PDF")
            return None
        
        # Show sample of extracted text
        log("")
        log("[Sample of extracted text - first 1000 chars]")
        log("-" * 40)
        print(extracted_text[:1000])
        log("-" * 40)
        
        prompt = "Analyze this clinical trial protocol and extract the Schedule of Events information. Return the healthcare contact days for each treatment arm as JSON according to the system instructions."
        
        max_retries = 3
        response_text = ""
        
        for attempt in range(max_retries):
            try:
                log(f"")
                log(f"[Step 2] Sending to DeepSeek - Attempt {attempt + 1}/{max_retries}")
                response_text = call_deepseek_with_text(api_key, extracted_text, prompt, SYSTEM_PROMPT, temperature=0.3)
                
                log("")
                log("=" * 40)
                log("DEEPSEEK RESPONSE:")
                log("=" * 40)
                print(response_text)
                log("=" * 40)
                
                break
                
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['overloaded', 'unavailable', 'rate limit', 'quota', 'timeout', '529', '503']):
                    if attempt < max_retries - 1:
                        wait_time = 30 * (attempt + 1)
                        log(f"Service unavailable. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        log(f"Failed after {max_retries} attempts")
                        raise e
                else:
                    raise e

        # Parse the JSON response
        log("")
        log("[Step 3] Parsing JSON response...")
        parsed_arms = parse_json_response(response_text)
        
        log(f"Parsed {len(parsed_arms)} treatment arm(s)")
        
        if not parsed_arms:
            log("WARNING: No valid treatment arms parsed from response")
            return None
        
        for _arm in parsed_arms:
            _validate_category_sums(_arm)
            log(f"  - Arm: {_arm.get('arm_name', 'Unknown')}")
        
        result_entry = {
            'filename': extract_pmid_from_filename(filename),
            'arms': parsed_arms
        }
        
        log("")
        log(f"SUCCESS: Extracted {len(parsed_arms)} treatment arm(s) from {filename}")
        return result_entry

    except Exception as e:
        log(f"ERROR processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_aggregate_results(all_results: List[Dict[str, Any]], path: str = 'aggregate_results.json') -> None:
    """Save aggregate JSON results (optional backup)."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Aggregate results saved to {path}")


def generate():
    log("=" * 60)
    log("CLINICAL TRIAL PROTOCOL ANALYZER")
    log("=" * 60)
    
    api_key = setup_openrouter_api()
    
    params = get_user_processing_parameters()
    if params is None:
        return
    
    csv_filename = params['csv_filename']
    
    initialize_csv_if_needed(csv_filename)
    processed_filenames = get_processed_filenames(csv_filename)
    
    pdf_files = get_multiple_pdfs_from_summaries(
        start_index=params['start_index'],
        num_pdfs=params['num_pdfs'],
        processed_filenames=processed_filenames
    )
    if not pdf_files:
        log("No new PDFs to process.")
        return

    total_files = len(pdf_files)
    log(f"")
    log(f"Starting batch processing of {total_files} PDF(s)")
    log(f"Sleep between requests: {SLEEP_BETWEEN_REQUESTS} seconds")
    log(f"")

    all_results: List[Dict[str, Any]] = []
    successful_count = 0
    failed_count = 0
    start_batch_time = time.time()
    
    for i, (filename, pdf_path) in enumerate(pdf_files, 1):
        file_start_time = time.time()
        
        result = process_single_pdf(api_key, filename, pdf_path, i, total_files)
        
        file_elapsed = time.time() - file_start_time
        
        if result:
            all_results.append(result)
            append_result_to_csv(result, csv_filename)
            successful_count += 1
            log(f"File {i}/{total_files} completed in {file_elapsed:.2f}s - SUCCESS")
        else:
            failed_count += 1
            log(f"File {i}/{total_files} completed in {file_elapsed:.2f}s - FAILED")
        
        # Progress summary
        log(f"")
        log(f"PROGRESS: {i}/{total_files} ({(i/total_files)*100:.1f}%) | Success: {successful_count} | Failed: {failed_count}")
        
        # Sleep between requests (except for last file)
        if i < total_files:
            log(f"Sleeping {SLEEP_BETWEEN_REQUESTS} seconds before next request...")
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    # Final summary
    total_elapsed = time.time() - start_batch_time
    log(f"")
    log(f"{'='*60}")
    log(f"BATCH PROCESSING COMPLETE")
    log(f"{'='*60}")
    log(f"Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    log(f"Files processed: {total_files}")
    log(f"Successful: {successful_count}")
    log(f"Failed: {failed_count}")
    log(f"Success rate: {(successful_count/total_files)*100:.1f}%")

    if all_results:
        save_aggregate_results(all_results)
        total_arms = sum(len(result['arms']) for result in all_results)
        log(f"Total treatment arms extracted: {total_arms}")
        log(f"Results saved to: {csv_filename}")
    else:
        log("No results to save.")


if __name__ == "__main__":
    generate()
