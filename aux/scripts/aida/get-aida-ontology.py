import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Extract AIDA ontology from Excel spreadsheet")
parser.add_argument("--sheet", type=str, default="", help="{events / entities}")
parser.add_argument("--path", type=str, default="", help="Path to the LDC resource directory")
ARGS = parser.parse_args()

sheet = pd.read_excel(f"{ARGS.path}/docs/LDC_AIDAAnnotationOntologyWithMapping_V8.xlsx", sheet_name=ARGS.sheet)

ldc_ontology = (
    (
        row['AnnotIndexID'],
        row['Output Value for Type'],
        row['Output Value for Subtype'],
        row['Output Value for Sub-subtype'] if 'Output Value for Sub-subtype' in row else row['Output Value for Sub-Subtype'],
        row['Definition']
    )
    for i, row in sheet.iterrows() if row['AnnotIndexID'].startswith('LDC_')
)

for cid, ct1, ct2, ct3, cdef in ldc_ontology:
    if ct2 == "unspecified":
        print(f"/{ct1}")
    elif ct3 == "unspecified":
        print(f"/{ct1}/{ct2}")
    else:
        print(f"/{ct1}/{ct2}/{ct3}")
