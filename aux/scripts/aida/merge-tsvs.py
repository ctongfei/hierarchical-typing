import pandas as pd
import sys

files = sys.argv[1:]

rows = pd.concat(
    pd.read_csv(f, delimiter='\t')
    for f in files
)


def get_type(t1: str, t2: str, t3: str) -> str:
    assert t1 != "unspecified"
    if t2 == "unspecified":
        return f"/{t1}"
    elif t3 == "unspecified":
        return f"/{t1}/{t2}"
    else:
        return f"/{t1}/{t2}/{t3}"


for _, r in rows.iterrows():
    if r['text_string'] != "EMPTY_NA":
        t1 = r['type']
        t2 = r['subtype']
        t3 = r['subsubtype']

        print(f"{r['child_uid']}:{r['textoffset_startchar']}:{r['textoffset_endchar']}\t{r['text_string']}\t{get_type(t1, t2, t3)}")
