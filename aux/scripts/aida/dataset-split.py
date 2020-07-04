import argparse
import itertools
import random

parser = argparse.ArgumentParser()
parser.add_argument("--all", type=str, default="")
parser.add_argument("--train", type=str, default="")
parser.add_argument("--dev", type=str, default="")
parser.add_argument("--test", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
ARGS = parser.parse_args()

random.seed = ARGS.seed
k = 0.8
k1 = 0.9
data = itertools.groupby(open(ARGS.all), key=lambda l: l.split(':')[0])
data = list(map(lambda t: (t[0], list(map(lambda s: s.strip(), t[1]))), data))
random.shuffle(data)

n = len(data)

out_train = open(ARGS.train, mode='w')
out_dev = open(ARGS.dev, mode='w')
out_test = open(ARGS.test, mode='w')

for _, lines in data[0:int(k * n)]:
    for l in lines:
        print(l, file=out_train)
for _, lines in data[int(k * n):int(k1 * n)]:
    for l in lines:
        print(l, file=out_dev)
for _, lines in data[int(k1 * n):n]:
    for l in lines:
        print(l, file=out_test)

out_train.close()
out_dev.close()
out_test.close()
