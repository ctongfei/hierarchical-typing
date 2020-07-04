from typing import *
import concrete
import sys
import concrete.util.file_io as cio
import numpy as np
import argparse
import csv
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("--tsv", type=str, default="", help="")
parser.add_argument("--concrete_dir", type=str, default="", help="")
parser.add_argument("--lang", type=str, default="eng")
ARGS = parser.parse_args()

tsv = csv.reader(open(ARGS.tsv), delimiter='\t')

for comm_id, rows in itertools.groupby(tsv, key=lambda r: r[0].split(':')[0]):

    try:
        comm = cio.read_communication_from_file(f"{ARGS.concrete_dir}/{comm_id}.comm")

        # remove non-English documents
        lang_dist = comm.lidList[0].languageToProbabilityMap
        lang = max(lang_dist.items(), key=lambda t: t[1])[0]
        if ARGS.lang != "all" and lang != ARGS.lang:  # if ARGS.lang == "all", retain all language samples
                continue

        sentences = [
            sentence
            for section in comm.sectionList
            for sentence in section.sentenceList
        ]

        sentence_indices: np.ndarray = np.array([sentence.textSpan.start for sentence in sentences])
        token_indices: List[np.ndarray] = [
            np.array([token.textSpan.start for token in sentence.tokenization.tokenList.tokenList])
            for sentence in sentences
        ]

        for row in rows:
                _, lidx, ridx = row[0].split(":")
                text = row[1]
                label = row[2]
                lidx = int(lidx)
                ridx = int(ridx)
                sentence_index = np.digitize([lidx], sentence_indices).item() - 1
                left_token_index = np.digitize([lidx], token_indices[sentence_index]).item() - 1
                right_token_index = np.digitize([ridx], token_indices[sentence_index]).item()

                sentence = ' '.join(t.text for t in sentences[sentence_index].tokenization.tokenList.tokenList)

                print(f"{text}\t{' '.join(map(lambda t: t.text, sentences[sentence_index].tokenization.tokenList.tokenList[left_token_index:right_token_index]))}", file=sys.stderr)
                print(f"{sentence}\t{left_token_index}:{right_token_index}\t{label}")

    except FileNotFoundError:
        pass  # some communications are not found in the LTF files
