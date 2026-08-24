"""
Reproduces Section 3.1's label-flip statistic: on the LaMP-1 queries shared by
both generators' relevance mappings, what fraction of (qid, pid) relevance
labels disagree between Flan-T5-Small and Flan-T5-Base, and what useful-rate
each generator shows over that same shared set. No LLM calls -- reads the
already-built relevance_mapping.tsv files directly.
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

SMALL_FP = os.path.join(ROOT, "data/lamp_utility_labels_flanT5Small/1_relevance_mapping.tsv")
BASE_FP = os.path.join(ROOT, "data/lamp_utility_labels_flanT5Base/1_relevance_mapping.tsv")


def load(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[(row["qid"], row["pid"])] = int(row["relevance_label"])
    return out


small = load(SMALL_FP)
base = load(BASE_FP)

small_qids = {q for q, _ in small}
base_qids = {q for q, _ in base}
shared_qids = small_qids & base_qids

shared_pairs = [k for k in small if k in base]

flips = sum(1 for k in shared_pairs if small[k] != base[k])
small_useful = sum(small[k] for k in shared_pairs)
base_useful = sum(base[k] for k in shared_pairs)

print(f"Shared LaMP-1 queries: {len(shared_qids)}")
print(f"Shared (qid, pid) label pairs: {len(shared_pairs)}")
print(f"Flip rate: {100 * flips / len(shared_pairs):.1f}%")
print(f"Flan-T5-Small useful-rate: {100 * small_useful / len(shared_pairs):.1f}%")
print(f"Flan-T5-Base useful-rate: {100 * base_useful / len(shared_pairs):.1f}%")
