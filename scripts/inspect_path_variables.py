#!/usr/bin/env python3
"""
Inspect PATH .dta variable labels and value labels to accelerate mapping.

Usage: python scripts/inspect_path_variables.py [wave]
- wave: optional wave number 1-5 (default: 1)

What it prints:
- Candidate AC9xxx variables (quit intentions/methods) with variable labels
- Any variables whose label contains keywords for education, home smoking rules, counseling, quitline
- Household/income variables for reference
"""
import sys
from pathlib import Path
import re

import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

KEYWORDS = {
    "education": ["educ", "school", "degree", "grade", "college"],
    "home_rules": ["home", "house", "indoors", "inside", "smoke-free", "smokefree", "allowed", "ban", "permit"],
    "counseling": ["counsel", "program", "support", "class"],
    "quitline": ["quitline", "helpline", "800", "1-800"],
}


def normalize_label(s: str) -> str:
    return (s or "").lower()


def show_matches(meta, varnames, heading):
    print(f"\n{heading}")
    print("=" * 70)
    if not varnames:
        print("  (none)")
        return
    # Build a name->label map
    name_to_label = {n: l for n, l in zip(meta.column_names, meta.column_labels)}
    for v in sorted(varnames):
        label = name_to_label.get(v, "")
        print(f"  {v:>20} : {label}")


def main():
    wave = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    dta_path = RAW / f"PATH_W{wave}_Adult_Public.dta"
    if not dta_path.exists():
        print(f"ERROR: {dta_path} not found")
        sys.exit(1)

    print(f"Reading labels from: {dta_path}")
    df, meta = pyreadstat.read_dta(str(dta_path), apply_value_formats=False)

    # AC9xxx candidates (quit intentions/methods cluster)
    ac9_vars = [c for c in df.columns if re.match(rf"R0{wave}_AC9\d+", c)]
    show_matches(meta, ac9_vars, heading=f"Wave {wave} AC9xxx candidates")

    # Search by label keywords
    for topic, kws in KEYWORDS.items():
        matches = []
        name_to_label = {n: l for n, l in zip(meta.column_names, meta.column_labels)}
        for c in df.columns:
            label = normalize_label(name_to_label.get(c, ""))
            if any(kw in label for kw in kws):
                matches.append(c)
        show_matches(meta, matches, heading=f"Wave {wave} label matches for {topic}")

    # Household/income quick refs
    quick = [f"R0{wave}R_POVCAT3", f"R0{wave}R_POVCAT2", f"R0{wave}R_HHSIZE5", f"R0{wave}R_HHYOUTH",
             f"R0{wave}R_A_SEX", f"R0{wave}R_A_HISP", f"R0{wave}R_A_RACECAT3", f"R0{wave}R_A_AGECAT7"]
    quick = [c for c in quick if c in df.columns]
    show_matches(meta, quick, heading=f"Wave {wave} quick demographic refs")


if __name__ == "__main__":
    main()
