import sys
import random


random.seed(0xCAFEBABE)

dev_size = 2000

lines = [l.strip() for l in open(sys.argv[1])]
indices = set(random.sample(range(len(lines)), dev_size))

train = open(sys.argv[2], mode='w')
dev = open(sys.argv[3], mode='w')

for i, l in enumerate(lines):
    if i in indices:
        print(l, file=dev)
    else:
        print(l, file=train)

train.close()
dev.close()
