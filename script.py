import os
import tempfile
import json
import csv
import re
import time
from typing import List, Dict, Any

from google import genai
from google.genai import types

def setup_gemini_api_v2():
    """Setup Gemini API authentication for second section"""
    # Hardcoded API key - replace with your actual API key
    api_key = "YOUR_GEMINI_API_KEY_HERE"
    os.environ["GEMINI_API_KEY"] = api_key
    print("Gemini API key configured successfully.")


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

def get_multiple_pdfs_from_summaries(start_index=0, num_pdfs=None):
    """Get multiple PDFs from summaries folder for batch processing with custom parameters."""
    summaries_folder = os.path.join(os.getcwd(), "summaries")
    pdf_files = []
    
    if not os.path.exists(summaries_folder):
        print(f"Summaries folder not found: {summaries_folder}")
        return pdf_files

    # Get all PDF files first
    all_files = []
    for filename in os.listdir(summaries_folder):
        if filename.lower().endswith('.pdf'):
            all_files.append(filename)
    
    # Sort files for consistent ordering
    all_files.sort()
    
    print(f"Found {len(all_files)} total PDFs in summaries folder")
    
    # Apply start index
    if start_index >= len(all_files):
        print(f"Error: Starting index {start_index} is beyond available files (max index: {len(all_files)-1})")
        return pdf_files
    
    selected_files = all_files[start_index:]
    
    # Apply number limit
    if num_pdfs is not None:
        selected_files = selected_files[:num_pdfs]
    
    # Load the selected files
    for filename in selected_files:
        file_path = os.path.join(summaries_folder, filename)
        with open(file_path, 'rb') as f:
            content = f.read()
        pdf_files.append((filename, content))
        print(f"Selected PDF: {filename}")

    print(f"\nTotal PDFs to process: {len(pdf_files)} (from index {start_index})")
    return pdf_files


def extract_pmid_from_filename(filename: str) -> str:
    """Extract PMID number from filename if present."""
    m = re.search(r'PMID\s+(\d+)', filename, flags=re.IGNORECASE)
    return m.group(1) if m else filename


def _strip_code_fences(text: str) -> str:
    """Remove common Markdown code fences around JSON/text."""
    text = re.sub(r"```(?:json)?\n(.*?)```", r"\1", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("```", "")
    return text.strip()


def parse_json_response(response_text: str) -> List[Dict[str, Any]]:
    """Parse one or more JSON objects for arms from the model response."""
    text = _strip_code_fences(response_text)

    def _validate(obj: Any) -> bool:
        return isinstance(obj, dict) and 'arm_name' in obj and 'healthcare_contact_days' in obj

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if _validate(d)]
        if _validate(data):
            return [data]
    except Exception:
        pass

    brace_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    objs = []
    for m in re.finditer(brace_pattern, text, flags=re.DOTALL):
        chunk = m.group(0)
        try:
            candidate = json.loads(chunk)
            if _validate(candidate):
                objs.append(candidate)
        except Exception:
            continue
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


def process_single_pdf(client: genai.Client, filename: str, pdf_content: bytes, model: str):
    """Process a single PDF file and return a result entry or None."""
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
        temp_pdf.write(pdf_content)
        temp_pdf_path = temp_pdf.name

    try:
        print("Uploading PDF to Gemini for analysis...")
        uploaded_file = client.files.upload(file=temp_pdf_path)
        print("PDF uploaded successfully.")

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type=uploaded_file.mime_type,
                    ),
                    types.Part.from_text(text=(
                        "Please analyze this clinical trial protocol PDF and extract the "
                        "Schedule of Events information according to the system instructions."
                    )),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=0.3,
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text(text="""SYSTEM PROMPT
You are an expert clinical‑trial protocol analyst.
Your task is to read a cancer‑trial Schedule of Events (SoE) or equivalent from the text and output the exact number of in‑person healthcare‑contact days per treatment arm at six fixed intervals. You will be provided clinical trial protocols with content deemed relevant to be able to extract healthcare contact days.
If there is nothing in the PDF that allows you to extract events or assessment schedules, please PRINT: NO DATA FOUND.

1  Key definitions
Healthcare‑contact day – Any calendar date on which the participant must physically attend a healthcare facility or meet face‑to‑face with study staff for trial‑mandated procedures. Count 1 per date, no matter how many procedures occur that day.

NOT a contact day – Tele‑visits, phone calls, home diaries, couriered drug, or truly optional ("as needed") visits.

Anchor (Day 0) – Cycle 1 Day 1 (C1D1): the first on‑treatment visit or first dose, whichever is earlier.

Time windows (inclusive):
  • Screening window  = Day ‑28 → Day ‑1
  • 1 month   = Day 0 → Day +30
  • 3 months  = Day 0 → Day +90
  • 6 months  = Day 0 → Day +180
  • 9 months  = Day 0 → Day +270
  • 12 months = Day 0 → Day +365

2  Extraction algorithm (think silently—do not output your chain of thought)
Isolate each arm's SoE table.

Create a visit pattern:
   • List every column that represents an in‑person visit (e.g., "Screening", "Randomisation", "Day 1", "Day 15").
   • For rows labelled "SUBSEQUENT CYCLES" or similar, record which visit columns are present; if a column is absent, assume that visit does not occur for cycles ≥ 2.

Build a canonical schedule:
   • Baseline visits → fixed dates: Day ‑28 ≤ date ≤ ‑1.
   • Cycle 1 visits → exact offsets (Day 0, Day +14, Day +7 etc.).
   • Repeating cycles → apply offsets every cycle_length days (default 28 unless otherwise stated).

Enumerate calendar dates for each visit through Day +365, assuming the regimen continues "until progression."

Count unique dates that fall inside each window.
   Do not require the entire cycle to fit—include any visit whose nominal date is inside the window.

Quality check: independently recompute the counts; if the two methods disagree, reconcile before answering.

3  Special rules & assumptions
Baseline split – If "Screening" and "Randomisation" appear as separate columns, treat them as two distinct days unless an SoE footnote explicitly allows same‑day consolidation.

Footnotes that make visits conditional – If a visit is labelled "optional" or "if clinically indicated," exclude it; if it is conditional on continuing a drug (e.g., bevacizumab), assume the drug continues.

Multi‑day blocks – "Days 1‑5" radiotherapy = 5 contact days; count only those days inside the window.

When data are ambiguous – Use the most conservative lower estimate and list the assumption.

4  Category classification (assign each in‑person calendar day to exactly ONE primary category)
Categories:
  a) core_treatment – active treatment delivery (chemotherapy, immunotherapy, targeted therapy, infusion, radiation fractions, interventional procedures including biopsies strictly as part of treatment delivery, port care related to treatment, surgery days).
  b) imaging_diagnostics – imaging (CT/MRI/PET/US/X‑ray), ECG/EKG, echocardiogram, diagnostic biopsies performed for assessment (not as part of treatment delivery), pulmonary function tests, stress tests, other diagnostic procedures.
  c) labs – phlebotomy visits, urine collections that require in‑person attendance.
  d) clinic_visits – H&P, AE checks, vitals, weight, questionnaires/ePRO done in clinic, nurse/physician visits, consent updates, counseling, education sessions.

Tie‑breaker precedence if multiple activities occur on the same calendar day:
  core_treatment > imaging_diagnostics > labs > clinic_visits.
That is, assign the day to the highest‑priority category present.

5  Output format (return a single JSON array; one object per arm) — output only JSON
[
  {
    "arm_name": "<exact arm name from the protocol>",
    "intervention_type": "<intervention or control>",
    "healthcare_contact_days": {
      "screening_window": <integer>,
      "1_month": <integer>,
      "3_months": <integer>,
      "6_months": <integer>,
      "9_months": <integer>,
      "12_months": <integer>
    },
    "category_breakdown": {
      "core_treatment": {
        "screening_window": <integer>, "1_month": <integer>, "3_months": <integer>, "6_months": <integer>, "9_months": <integer>, "12_months": <integer>
      },
      "imaging_diagnostics": {
        "screening_window": <integer>, "1_month": <integer>, "3_months": <integer>, "6_months": <integer>, "9_months": <integer>, "12_months": <integer>
      },
      "labs": {
        "screening_window": <integer>, "1_month": <integer>, "3_months": <integer>, "6_months": <integer>, "9_months": <integer>, "12_months": <integer>
      },
      "clinic_visits": {
        "screening_window": <integer>, "1_month": <integer>, "3_months": <integer>, "6_months": <integer>, "9_months": <integer>, "12_months": <integer>
      }
    },
    "extraction_notes": {
      "assumptions": ["<bullet point list of any assumptions or judgments made>"],
      "visit_pattern": "<concise description>",
      "cycle_length_days": <integer>,
      "treatment_duration_rule": "<e.g. 'until progression'>"
    }
  }
]

6  Final checklist (silent)
Each calendar date counted once?
Category precedence applied?
Category sums within a window must equal total healthcare‑contact days for that window?
Assumptions listed?"""),
            ],
        )

        print(f"Analyzing {filename}...")

        response_text = ""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Attempt to generate content
                for chunk in client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if chunk.text is not None:
                        print(chunk.text, end="")
                        response_text += chunk.text
                break  # If successful, exit the retry loop
            except Exception as e:
                # Check if it's a service unavailable or rate limit error
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['overloaded', 'unavailable', 'rate limit', 'quota', 'timeout']):
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds...
                        print(f"\nService temporarily unavailable. Retrying in {wait_time} seconds... ({attempt + 1}/{max_retries})")
                        print(f"Error details: {e}")
                        time.sleep(wait_time)
                    else:
                        print(f"\nFailed after {max_retries} attempts due to service issues.")
                        raise e  # Re-raise the exception if all retries fail
                else:
                    # For other types of errors, don't retry
                    raise e

        parsed_arms = parse_json_response(response_text)
        # Consistency checks
        for _arm in parsed_arms:
            _validate_category_sums(_arm)
        if parsed_arms:
            result_entry = {
                'filename': extract_pmid_from_filename(filename),
                'arms': parsed_arms
            }
            print(f"\n\nSuccessfully processed {filename} - Found {len(parsed_arms)} treatment arms")
            return result_entry
        else:
            print(f"\n\nWarning: No valid results found for {filename}")
            return None

    except Exception as e:
        print(f"\nError processing {filename}: {e}")
        return None

    finally:
        if os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)


def save_aggregate_results(all_results: List[Dict[str, Any]], path: str = 'aggregate_results.json') -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Aggregate results saved to {path}")


def save_csv_results(all_results: List[Dict[str, Any]], output_csv_name: str = 'clinical_trial_results.csv'):
    """Save CSV file with flattened results, including category breakdowns per window."""
    if not all_results:
        return

    windows = ['screening_window', '1_month', '3_months', '6_months', '9_months', '12_months']
    categories = ['core_treatment', 'imaging_diagnostics', 'labs', 'clinic_visits']

    headers = [
        'filename',
        'arm_name',
        'intervention_type',
    ]

    # total counts per window
    headers += [w for w in windows]

    # category breakdown columns
    for cat in categories:
        for w in windows:
            headers.append(f"{cat}_{w}")

    headers += [
        'cycle_length_days',
        'treatment_duration_rule',
        'visit_pattern',
        'assumptions'
    ]

    with open(output_csv_name, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        def get(d, *path, default=''):
            cur = d
            for k in path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    return default
            return cur

        for result in all_results:
            filename = result.get('filename', '')
            for arm_data in result.get('arms', []):
                row = [
                    filename,
                    arm_data.get('arm_name', ''),
                    arm_data.get('intervention_type', ''),
                ]

                # totals per window
                for w in windows:
                    row.append(get(arm_data, 'healthcare_contact_days', w, default=''))

                # per-category per window
                for cat in categories:
                    for w in windows:
                        row.append(get(arm_data, 'category_breakdown', cat, w, default=''))

                # notes
                row.extend([
                    get(arm_data, 'extraction_notes', 'cycle_length_days', default=''),
                    get(arm_data, 'extraction_notes', 'treatment_duration_rule', default=''),
                    get(arm_data, 'extraction_notes', 'visit_pattern', default=''),
                    '; '.join(get(arm_data, 'extraction_notes', 'assumptions', default=[])) if isinstance(get(arm_data, 'extraction_notes', 'assumptions', default=[]), list) else get(arm_data, 'extraction_notes', 'assumptions', default='')
                ])

                writer.writerow(row)

    print(f"CSV results saved to {output_csv_name}")


def generate():
    setup_gemini_api_v2()
    
    # Get user parameters
    params = get_user_processing_parameters()
    if params is None:
        return
    
    pdf_files = get_multiple_pdfs_from_summaries(
        start_index=params['start_index'],
        num_pdfs=params['num_pdfs']
    )
    if not pdf_files:
        return

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-2.5-flash"

    all_results: List[Dict[str, Any]] = []
    for i, (filename, pdf_content) in enumerate(pdf_files, 1):
        print(f"\n\nProcessing file {i} of {len(pdf_files)}")
        result = process_single_pdf(client, filename, pdf_content, model)
        if result:
            all_results.append(result)

    print(f"\n\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(all_results)} out of {len(pdf_files)} PDFs")

    if all_results:
        save_aggregate_results(all_results)
        save_csv_results(all_results, params['csv_filename'])
        total_arms = sum(len(result['arms']) for result in all_results)
        print(f"Total treatment arms extracted: {total_arms}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    generate()