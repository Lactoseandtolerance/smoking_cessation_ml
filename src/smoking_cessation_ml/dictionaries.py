"""Extract TUS-CPS variable layouts from PDF codebooks using pdfplumber.

This module locates "Attachment 7: Supplement Record Layout" sections and extracts
variable layout tables into structured CSV/JSON outputs consumed by the fixed-width
reader in subsequent pipeline steps.

Design notes:
- Uses the project's logging utilities (`setup_logger`, `log_step`) for instrumention.
- The pdf parsing functions are defensive: they return empty results and log warnings
  rather than raising when PDFs are missing or tables are malformed. This makes the
  module safe to import in test environments where PDFs are not present.

"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import re
import json
import logging

import pandas as pd

# optional import; will raise if not installed at runtime when functions are executed
try:
    import pdfplumber  # type: ignore
    PDFPLUMBER_AVAILABLE = True
except Exception:  # pragma: no cover - defensive import handling
    pdfplumber = None
    PDFPLUMBER_AVAILABLE = False

from .utils import setup_logger, log_step, log_dataframe_info

LOGGER = logging.getLogger("smoking_cessation.dictionaries")


@log_step
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract and return text content from all pages of a PDF using pdfplumber.

    Returns empty string if the file cannot be opened or has no text.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        LOGGER.error("PDF not found: %s", pdf_path)
        return ""

    text_parts: List[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                text_parts.append(t)
    except Exception as exc:
        LOGGER.exception("Failed to read PDF %s: %s", pdf_path, exc)
        return ""

    return "\n".join(text_parts)


@log_step
def find_attachment_7_pages(pdf_path: Path) -> List[int]:
    """Locate pages mentioning "Attachment 7" or "Supplement Record Layout".

    Returns a list of 0-based page indices where the attachment appears.
    """
    if not PDFPLUMBER_AVAILABLE:
        raise RuntimeError("pdfplumber is required to scan PDF pages. Install pdfplumber in your environment.")

    pages_found: List[int] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if re.search(r"Attachment\s*7", text, re.IGNORECASE) or re.search(r"Supplement Record Layout", text, re.IGNORECASE):
                    # include this page and attempt to include subsequent contiguous layout pages
                    pages_found.append(i)
                    # lookahead to include contiguous pages until the next 'Attachment' header
                    max_look = min(i + 10, num_pages - 1)
                    for j in range(i + 1, max_look + 1):
                        next_text = pdf.pages[j].extract_text() or ""
                        # stop if a new Attachment header is found
                        if re.search(r"Attachment\s*\d+", next_text, re.IGNORECASE) and not re.search(r"Attachment\s*7", next_text, re.IGNORECASE):
                            break
                        # include page if it looks like a continuation (has table-like structure or similar column headers)
                        if re.search(r"\bcolumn\b|\bvariable\b|\bstart\b|\bend\b|\bposition\b", next_text, re.IGNORECASE) or len(next_text.splitlines()) > 5:
                            pages_found.append(j)
                        else:
                            # if the page seems short and not table-like, assume end of layout
                            break
    except Exception as exc:
        LOGGER.exception("Error scanning PDF %s for Attachment 7: %s", pdf_path, exc)
    # dedupe and sort
    pages_found = sorted(list(dict.fromkeys(pages_found)))
    LOGGER.info("Attachment 7 pages for %s: %s", pdf_path.name, pages_found)
    return pages_found


def _table_settings() -> dict:
    # heuristics for pdfplumber table extraction; callers may tune these values
    return {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "text_x_tolerance": 3,
        "text_y_tolerance": 3,
    }


@log_step
def extract_variable_tables(pdf_path: Path, pages: List[int]) -> List[pd.DataFrame]:
    """Extract tables from the specified pages and return as pandas DataFrames.

    If pdfplumber fails to detect tables, returns an empty list.
    """
    dfs: List[pd.DataFrame] = []
    if not PDFPLUMBER_AVAILABLE:
        raise RuntimeError("pdfplumber is required to extract tables from PDFs. Install pdfplumber in your environment.")

    if not pages:
        LOGGER.warning("No pages provided for table extraction: %s", pdf_path)
        return dfs

    try:
        with pdfplumber.open(pdf_path) as pdf:
            settings = _table_settings()
            for p in pages:
                if p < 0 or p >= len(pdf.pages):
                    continue
                page = pdf.pages[p]
                # try extract_tables with heuristics
                try:
                    tables = page.extract_tables(table_settings=settings) or []
                except Exception:
                    # fallback: try without settings
                    try:
                        tables = page.extract_tables() or []
                    except Exception:
                        tables = []

                for table in tables:
                    # table is list-of-rows; first row may be header
                    try:
                        df = pd.DataFrame(table)
                        # drop empty rows
                        df = df.dropna(how="all")
                        if df.shape[1] > 1 and df.shape[0] > 0:
                            dfs.append(df)
                    except Exception:
                        LOGGER.exception("Failed to convert extracted table on page %s to DataFrame", p)
    except Exception as exc:
        LOGGER.exception("Failed to open PDF %s for table extraction: %s", pdf_path, exc)

    LOGGER.info("Extracted %d raw tables from %s", len(dfs), pdf_path.name)
    return dfs


def _normalize_colname(name: str) -> str:
    if name is None:
        return ""
    s = str(name).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_position_field(pos_field: str) -> Optional[Tuple[int, int]]:
    """Parse position strings like '951-952' or '951 - 952' or '951' into (start, end).

    Returns None on malformed input.
    """
    if pos_field is None:
        return None
    s = str(pos_field).strip()
    m = re.search(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        try:
            start = int(m.group(1))
            end = int(m.group(2))
            return (start, end)
        except Exception:
            return None
    # single number -> treat as start with width 1
    m2 = re.search(r"^(\d+)$", s)
    if m2:
        n = int(m2.group(1))
        return (n, n)
    return None


@log_step
def parse_variable_layout(tables: List[pd.DataFrame]) -> pd.DataFrame:
    """Attempt to parse a list of raw DataFrames (from pdfplumber) into a layout DataFrame.

    The function is heuristic and tolerant of varying column names. It looks for columns
    that contain 'variable' / 'name', 'column' / 'pos' / 'start' / 'end', 'type', and 'label' texts.
    """
    rows: List[dict] = []
    for df in tables:
        # attempt to infer header row: prefer first row that contains the word 'variable' or 'name'
        header_row = None
        try:
            # convert all cells to str for inspection
            inspect_df = df.fillna("").astype(str)
            for ridx in range(min(3, inspect_df.shape[0])):
                row_vals = " ".join(inspect_df.iloc[ridx].values.tolist()).lower()
                if "variable" in row_vals or "name" in row_vals:
                    header_row = ridx
                    break
            if header_row is None:
                # assume first row is header
                header_row = 0

            header = [ _normalize_colname(c) for c in inspect_df.iloc[header_row].tolist() ]
            data_rows = inspect_df.iloc[header_row+1:]
            data_rows.columns = header

            for _, r in data_rows.iterrows():
                # build a normalization map for common columns
                rdict = {k.lower(): v for k, v in r.items()}
                # heuristics to extract fields
                var_candidates = [v for k,v in rdict.items() if re.search(r"var|variable|name", k)]
                pos_candidates = [v for k,v in rdict.items() if re.search(r"col|pos|start|end|column", k)]
                type_candidates = [v for k,v in rdict.items() if re.search(r"type|format", k)]
                label_candidates = [v for k,v in rdict.items() if re.search(r"label|value|code|desc|question|text", k)]
                missing_candidates = [v for k,v in rdict.items() if re.search(r"miss|missing|na|n/a|not applicable", k)]

                var_name = var_candidates[0] if var_candidates else ""
                pos_field = pos_candidates[0] if pos_candidates else ""
                dtype = type_candidates[0] if type_candidates else ""
                label = label_candidates[0] if label_candidates else ""
                missing_direct = missing_candidates[0] if missing_candidates else ""

                pos = _parse_position_field(pos_field)
                if pos is None:
                    # sometimes position is split into two columns, try to get numeric pairs
                    starts = [v for k,v in rdict.items() if re.search(r"start", k)]
                    ends = [v for k,v in rdict.items() if re.search(r"end", k)]
                    if starts and ends:
                        try:
                            s = int(str(starts[0]).strip())
                            e = int(str(ends[0]).strip())
                            pos = (s, e)
                        except Exception:
                            pos = None

                start_pos = pos[0] if pos else None
                end_pos = pos[1] if pos else None
                width = (end_pos - start_pos + 1) if (start_pos and end_pos) else None

                # derive description vs value labels
                value_labels = ""
                description = ""
                label_text = str(label).strip()
                if label_text:
                    # if the label contains mapping-like syntax (e.g. '1 = Yes; 2 = No') treat as value labels
                    if re.search(r"\d+\s*[=:\)]|;|\bNo\b|\bYes\b", label_text):
                        value_labels = label_text
                    else:
                        # long text likely the question/description
                        if len(label_text) > 80:
                            description = label_text
                        else:
                            # short labels may still be value labels
                            value_labels = label_text

                # missing codes: prefer dedicated missing column, else infer canonical negative sentinels
                missing_codes_raw = ""
                if missing_direct:
                    # Use dedicated missing column as-is but normalize separators and extract signed codes
                    raw = str(missing_direct).strip()
                    # extract signed or unsigned integers, preserving sign if present
                    tokens = re.findall(r"-?\d+", raw)
                    # normalize to comma-separated signed tokens
                    # preserve negative sign; if positive tokens appear here, assume they're intentional
                    # (the dedicated column takes precedence)
                    seen = set()
                    normalized = []
                    for t in tokens:
                        if t not in seen:
                            seen.add(t)
                            normalized.append(t)
                    missing_codes_raw = ",".join(normalized)
                else:
                    # Infer only canonical negative missing sentinels from label_text: -1, -7, -8, -9
                    if label_text:
                        found = re.findall(r"-(?:1|7|8|9)\b", label_text)
                        # found contains strings like '-1', '-7'
                        if found:
                            # dedupe while preserving order
                            seen = set()
                            normalized = []
                            for t in found:
                                if t not in seen:
                                    seen.add(t)
                                    normalized.append(t)
                            missing_codes_raw = ",".join(normalized)

                # final normalized missing codes string (empty if none)
                missing_codes = missing_codes_raw or ""
                if missing_codes:
                    # use var_name (available) rather than row which is not yet constructed
                    LOGGER.debug("Detected missing codes for %s: %s", str(var_name).strip() or "<unknown>", missing_codes)

                row = {
                    "variable_name": str(var_name).strip(),
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "width": width,
                    "data_type": str(dtype).strip(),
                    "value_labels": value_labels,
                    "missing_codes": missing_codes,
                    "description": description,
                }
                # minimal sanity: variable name must exist
                if row["variable_name"]:
                    rows.append(row)
        except Exception:
            LOGGER.exception("Failed to parse one of the extracted tables")

    layout_df = pd.DataFrame(rows)
    # basic cleanup
    layout_df["variable_name"] = layout_df["variable_name"].str.replace("\s+", " ", regex=True).str.strip()
    # dedupe
    layout_df = layout_df.drop_duplicates(subset=["variable_name", "start_pos"]) if not layout_df.empty else layout_df

    log_dataframe_info(layout_df, name="parsed_layout", logger=LOGGER)
    LOGGER.info("Parsed %d variables from tables", len(layout_df))
    return layout_df


KEY_VARIABLE_PATTERNS = [
    r"^HRHHID$",
    r"^PENXTPR$",
    r"^GESTFIPS$",
    r"^GTCO$",
    r"^PEA1$",
    r"^PEA2$",
    r"^PEA3$",
    r"^ZA1$",
    r"^H1NUM$",
    r"^H1UNT$",
    r"^PRTAGE$",
    r"^PESEX$",
    r"^PEEDUCA$",
    r"^PTDTRACE$",
    r"^PEHSPNON$",
    r"^HEFAMINC$",
    # person identifiers and person-line variables
    r"^PULINENO$",
    r"^PEPERNUM$",
    # common cessation/section variables (heuristic prefixes)
    r"^B3[A-Z0-9]+$",
    r"^C[0-9]+[A-Z]*$",
]


@log_step
def filter_key_variables(layout_df: pd.DataFrame) -> pd.DataFrame:
    """Filter the parsed layout to the key variables relevant for analysis.

    Uses both exact matches and simple regex patterns.
    """
    if layout_df is None or layout_df.empty:
        LOGGER.warning("Empty layout provided to filter_key_variables")
        return layout_df

    keep_mask = pd.Series(False, index=layout_df.index)
    for pat in KEY_VARIABLE_PATTERNS:
        mask = layout_df["variable_name"].str.match(pat, case=False, na=False)
        keep_mask = keep_mask | mask

    # also keep variables that contain common prefixes for person-level variables
    pref_mask = layout_df["variable_name"].str.contains(r"^PE|^PA|^ZA|^H1|^B3|^C\d|NRT|SMK|QUIT|PULINENO|PEPERNUM", case=False, na=False)
    keep_mask = keep_mask | pref_mask

    filtered = layout_df[keep_mask].copy()
    LOGGER.info("Filtered layout: kept %d of %d variables", filtered.shape[0], layout_df.shape[0])
    return filtered


@log_step
def parse_may_2010_codebook(pdf_path: Path) -> pd.DataFrame:
    """Orchestrate parsing for the May 2010 baseline codebook."""
    pages = find_attachment_7_pages(pdf_path)
    tables = extract_variable_tables(pdf_path, pages)
    parsed = parse_variable_layout(tables)
    parsed["wave"] = "baseline"
    # TUS variables often start at column 951 in May 2010 - annotate if missing
    LOGGER.debug("May 2010 parsing complete: %d vars", len(parsed))
    filtered = filter_key_variables(parsed)
    return filtered


@log_step
def parse_jan_2011_codebook(pdf_path: Path) -> pd.DataFrame:
    """Orchestrate parsing for the Jan 2011 follow-up codebook."""
    pages = find_attachment_7_pages(pdf_path)
    tables = extract_variable_tables(pdf_path, pages)
    parsed = parse_variable_layout(tables)
    parsed["wave"] = "followup"
    LOGGER.debug("Jan 2011 parsing complete: %d vars", len(parsed))
    filtered = filter_key_variables(parsed)
    return filtered


@log_step
def merge_baseline_followup_layouts(baseline_df: pd.DataFrame, followup_df: pd.DataFrame) -> pd.DataFrame:
    """Merge baseline and followup layouts, adding a 'wave' column.

    Ensures variable_name + wave is unique.
    """
    if baseline_df is None:
        baseline_df = pd.DataFrame()
    if followup_df is None:
        followup_df = pd.DataFrame()

    base = baseline_df.copy() if not baseline_df.empty else pd.DataFrame()
    fol = followup_df.copy() if not followup_df.empty else pd.DataFrame()

    merged = pd.concat([base, fol], ignore_index=True, sort=False)
    # ensure width computed where start_pos and end_pos are present; preserve existing non-null widths
    if "start_pos" in merged.columns and "end_pos" in merged.columns:
        def _compute_width(r):
            try:
                if pd.notna(r.get("width")) and r.get("width") not in (None, ""):
                    return r.get("width")
                s = r.get("start_pos")
                e = r.get("end_pos")
                if pd.notna(s) and pd.notna(e):
                    return int(e) - int(s) + 1
            except Exception:
                return None
            return None
        merged["width"] = merged.apply(_compute_width, axis=1)

    # drop exact duplicates
    merged = merged.drop_duplicates(subset=["variable_name", "wave"], keep="first")
    LOGGER.info("Merged layouts: total %d variables", merged.shape[0])
    log_dataframe_info(merged, name="merged_layout", logger=LOGGER)
    return merged


@log_step
def export_layout_to_csv(layout_df: pd.DataFrame, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        layout_df.to_csv(output_path, index=False, encoding="utf-8")
        LOGGER.info("Exported layout CSV to %s (rows=%d)", output_path, len(layout_df))
    except Exception:
        LOGGER.exception("Failed to export layout to CSV: %s", output_path)


@log_step
def export_layout_to_json(layout_df: pd.DataFrame, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        records = layout_df.to_dict(orient="records")
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2)
        LOGGER.info("Exported layout JSON to %s (records=%d)", output_path, len(records))
    except Exception:
        LOGGER.exception("Failed to export layout to JSON: %s", output_path)


def main(raw_dir: Optional[Path] = None) -> None:
    """Entry point: parse baseline and follow-up PDFs and export merged layouts.

    By default, looks for PDFs in `data/raw/` and writes outputs to `data/interim/`.
    """
    logger = setup_logger(name="smoking_cessation.dictionaries")
    logger.info("Starting TUS-CPS dictionary extraction pipeline...")

    root = Path.cwd()
    raw_dir = Path(raw_dir) if raw_dir else root / "data" / "raw"
    interim_dir = root / "data" / "interim"

    may_pdf = raw_dir / "cpsmay10.pdf"
    # follow-up codebook: prefer 'cpsuse10-11.pdf' if present in repo, else fall back to 'cpsjan11.pdf'
    followup_candidates = [raw_dir / "cpsuse10-11.pdf", raw_dir / "cpsjan11.pdf"]
    jan_pdf = None
    for p in followup_candidates:
        if p.exists():
            jan_pdf = p
            break
    # if none exist, default to the first candidate for messaging; actual existence checked below
    if jan_pdf is None:
        jan_pdf = followup_candidates[0]

    baseline_df = pd.DataFrame()
    followup_df = pd.DataFrame()

    if may_pdf.exists():
        baseline_df = parse_may_2010_codebook(may_pdf)
    else:
        logger.warning("May 2010 PDF not found at %s", may_pdf)

    if jan_pdf.exists():
        followup_df = parse_jan_2011_codebook(jan_pdf)
    else:
        logger.warning("Follow-up PDF not found. Checked: %s or %s", followup_candidates[0], followup_candidates[1])

    merged = merge_baseline_followup_layouts(baseline_df, followup_df)

    csv_out = interim_dir / "tus_cps_layout.csv"
    json_out = interim_dir / "tus_cps_layout.json"

    if not merged.empty:
        export_layout_to_csv(merged, csv_out)
        export_layout_to_json(merged, json_out)
    else:
        logger.warning("No layout data parsed; skipping export")

    logger.info("Dictionary extraction pipeline finished.")


if __name__ == "__main__":
    main()
