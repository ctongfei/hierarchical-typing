from typing import *
from hiertype.contextualizers import get_contextualizer

s = "He found a leprechaun in his walnut shell .".split(' ')

contextualizer = get_contextualizer("xlm-roberta-base", device='cuda:0')

t, m = contextualizer.tokenize_with_mapping(s)
encoded = contextualizer.encode([t], frozen=True)
pass
