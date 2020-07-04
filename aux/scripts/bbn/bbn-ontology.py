import sys

all_types = set()

for line in sys.stdin:
    _, _, types = line.strip().split('\t')
    for t in types.split(' '):
        all_types.add(t)

for t in sorted(list(all_types)):
    print(t)
