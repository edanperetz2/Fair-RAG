"""
EU normalization matching the source paper's own canonical formula (Kim & Diaz
2025, Appendix C.3 / legacy/normalize_eu.py - confirmed run as a universal
post-processing step over every experiment condition in their original codebase,
not a special-purpose comparison-table formula):

    EU_norm(q) = EU(q) / max(gold_ceiling(q), model_ceiling(q))

Appendix C.3 defines the ceiling as an empirical approximation of a true upper
bound "given a fixed RAG model RAG(R,G,k)": u_max = max(u_1..u_N, u*_1..u*_N), the
model's own sampled utilities together with the oracle's. R, G (and k, always 5
here and in the source paper, so never a free dimension) are fixed per ceiling -
confirmed against legacy/normalize_eu.py, which takes retriever_name as a required
argument baked into both the gold and model file paths it reads.

Per query:
  - gold_ceiling  = max EU across the gold ranker's PL(alpha=8, samples=3) draws
                    (this project's cheap stand-in for the paper's fuller
                    stochastic-oracle search u*_1..u*_N; shared across rankers,
                    since gold doesn't depend on which deterministic ranker is
                    being scored)
  - model_ceiling = max EU across every row run for that (lamp_num, generator,
                    ranker) cell, regardless of rerank method - not just PL.
                    Kim & Diaz's own u_1..u_N is PL-only because PL was the only
                    stochastic policy in their study; MMR (and RandMMR) are this
                    reproduction's own addition, and are just as much real,
                    empirically observed samples of what that same fixed
                    RAG(R,G,k) can produce. Restricting to PL would be an
                    artifact of what they happened to run, not a requirement -
                    more empirical samples from the same fixed system only
                    tighten the true-upper-bound approximation.

`analysis.normalization.normalize_query_rows` instead divides by the plain
empirical max observed within the group (across whatever methods/settings happen
to be in the input frame) - a different, non-paper-matching convention. This
module exists so every result in the paper uses the one formula the source paper
itself uses throughout, not just the Table 4 baseline comparison it was first
built for (see experiments/table4_with_gold.py, the original single-purpose
version this generalizes).

Only EU is affected. EE-R and ILD normalization (plain empirical-max within
group) are unrelated to this formula and unchanged.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import pandas as pd

from analysis.normalization import EU_UPPER_BOUND

GOLD_ALPHA = 8.0
GOLD_SAMPLES = 3.0


def _eu_value(row: pd.Series, is_lower_better: bool) -> float:
    """Per-row utility used for ceiling purposes: max_utility (or expected_utility
    as a fallback for single-list rows like deterministic), flipped for LaMP-3."""
    val = row.get("max_utility")
    if pd.isna(val):
        val = row.get("expected_utility")
    if pd.isna(val):
        return float("nan")
    return (EU_UPPER_BOUND - float(val)) if is_lower_better else float(val)


def build_with_gold_ceiling(
    raw_df: pd.DataFrame,
    lamp_num: int,
    generator: str,
    ranker: str,
) -> Dict[str, float]:
    """
    Per-qid ceiling = max(gold_ceiling, model_ceiling) for one (lamp_num, generator,
    ranker) cell - i.e. one fixed RAG(R,G,k). `raw_df` must be a
    `build_query_metric_rows`-shaped frame that includes both this cell's own rows
    (every rerank method run for it) and the shared gold-PL(a=8,s=3) rows for this
    (lamp_num, generator).
    """
    is_lower_better = int(lamp_num) == 3

    gold_sub = raw_df[
        (raw_df["lamp_num"] == lamp_num) & (raw_df["generator_name"] == generator)
        & (raw_df["ranker"] == "gold") & (raw_df["rerank_method"] == "pl")
        & (raw_df["pl_alpha"] == GOLD_ALPHA) & (raw_df["pl_samples"] == GOLD_SAMPLES)
    ]
    ceiling: Dict[str, float] = {}
    for _, row in gold_sub.iterrows():
        v = _eu_value(row, is_lower_better)
        if pd.isna(v):
            continue
        qid = row["qid"]
        if qid not in ceiling or v > ceiling[qid]:
            ceiling[qid] = v

    # model_ceiling: every row actually run for this fixed (generator, ranker) -
    # PL at any alpha, MMR at any lambda, RandMMR, deterministic - not just PL
    # (see module docstring).
    model_sub = raw_df[
        (raw_df["lamp_num"] == lamp_num) & (raw_df["generator_name"] == generator)
        & (raw_df["ranker"] == ranker)
    ]
    for _, row in model_sub.iterrows():
        v = _eu_value(row, is_lower_better)
        if pd.isna(v):
            continue
        qid = row["qid"]
        if qid not in ceiling or v > ceiling[qid]:
            ceiling[qid] = v

    return ceiling


def normalize_query_rows_with_gold(
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    cells: Iterable[Tuple[int, str, str]],
    *,
    base_group_cols: Sequence[str] = ("lamp_num", "generator_name", "qid"),
) -> pd.DataFrame:
    """
    Like `analysis.normalization.normalize_query_rows`, but `expected_utility_norm`
    uses the with-gold ceiling (see module docstring) instead of the plain
    empirical max observed within `df`. EE-R/ILD/EE-D normalization is delegated
    unchanged to the existing plain-empirical-max convention (`base_group_cols`
    matches that function's own default - pooled across rankers - so this is a
    true drop-in replacement for callers that only care about fixing EU).

    `df` is the frame to normalize (e.g. det + all PL alphas + all MMR lambdas for
    the cells in question) - it should NOT itself include the gold-PL rows (they
    exist only to inform the ceiling, not as a method being compared).
    `raw_df` is the full unfiltered `build_query_metric_rows` frame, used to look
    up each cell's gold and model ceiling inputs - it SHOULD include every rerank
    method run for these cells (PL, MMR, RandMMR, deterministic), since
    model_ceiling now draws on all of them, not just PL.
    `cells` is the list of (lamp_num, generator, ranker) triples to build ceilings
    for; `df` should only contain rows from within these cells.
    """
    from analysis.normalization import normalize_query_rows

    if df.empty:
        return df.copy()

    # EE-R / ILD / EE-D normalization: reuse the existing plain-empirical-max
    # logic unchanged, then overwrite expected_utility_norm below.
    base = normalize_query_rows(df, group_cols=base_group_cols)

    ceilings = {
        (lamp_num, generator, ranker): build_with_gold_ceiling(raw_df, lamp_num, generator, ranker)
        for (lamp_num, generator, ranker) in cells
    }

    def _norm_eu(row) -> float:
        key = (row["lamp_num"], row["generator_name"], row["ranker"])
        ceiling = ceilings.get(key, {})
        denom = ceiling.get(row["qid"])
        is_lower_better = int(row["lamp_num"]) == 3
        numer = (EU_UPPER_BOUND - row["expected_utility"]) if is_lower_better else row["expected_utility"]
        # Matches legacy/normalize_eu.py's own fallback exactly (try/except
        # ZeroDivisionError: normalized_eu = 1.0), not analysis.normalization's
        # deliberately-different None-on-zero convention - see module docstring.
        if denom is None or pd.isna(denom) or denom == 0.0:
            return 1.0
        return float(numer) / float(denom)

    base["expected_utility_norm"] = base.apply(_norm_eu, axis=1)
    return base
