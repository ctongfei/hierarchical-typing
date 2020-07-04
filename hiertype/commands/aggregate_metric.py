import typing
import os
import sys
import re
import json
from colors import blue


metric = sys.argv[1]
glob = sys.argv[2:]

r = re.compile(r"metrics_epoch_(.*)\.json")

for path in glob:
    try:
        files = {
            int(r.findall(f)[0]): f
            for f in os.listdir(f"{path}/out/") if r.match(f)
        }

        max_epoch = max(files.keys())
        j = json.load(open(f"{path}/out/metrics_epoch_{max_epoch}.json"))

        print(f"{blue(path)}: {j[f'best_validation_{metric}']}")

    except:
        pass
