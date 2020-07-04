import json
import sys


def normalize(t: str) -> str:
    if t == "``":
        return "\""
    elif t == "''":
        return "\""
    else:
        return t


for line in sys.stdin:
    j = json.loads(line)
    sentence = ' '.join(normalize(t) for t in j["tokens"])

    for m in j["mentions"]:
        left = m["start"]
        right = m["end"]
        types = [t.lower() for t in m["labels"]]

        print(f"{sentence}\t{left}:{right}\t{' '.join(types)}")
