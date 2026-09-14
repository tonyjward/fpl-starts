"""Classify-then-lookup: converts the agent's category classification for a
player into a probability, via the same continuous-shrinkage mechanism the
private repo's own news arm validated (reimplemented here, not imported --
this package has no dependency on that repo).

The model's job (predict.py) is only ever to pick a category and cite a
verbatim quote. The probability that category maps to is decided here, by
code, from observed outcomes -- never by the model. See predict.py's module
docstring for why: an LLM emitting a probability directly was tested against
this approach twice in the private repo and lost both times.

Python 3.7 target: no walrus operator, no `X | Y` unions, no f-string `=`.
"""

import pandas as pd

# Hand-set starting values -- reasoned guesses, not fitted. Provisional
# until fit_category_rates has enough of this arm's own history to say
# otherwise; that's expected to take several gameweeks, not this first one.
# confirmed_out is deliberately absent: an uncontested confirmed_out
# classification hard-gates to exactly 0.0 in predict.py, the same
# treatment predict_gameweek_refined's own status gate gets, rather than
# going through shrinkage.
CATEGORY_PRIORS = {
    "confirmed_starting": 0.90,
    "rotation_risk": 0.50,
    "returning_from_injury": 0.35,
}

# How many same-category, this-arm's-own-history observations it takes for
# real data to roughly match the prior's weight. A guess, like the priors
# themselves -- see shrink's docstring.
SHRINKAGE_K = 10


def fit_category_rates(conn, season, target_round, model_version="refined_availability_agent_news"):
    """{category: (observed_sum, observed_n)} of actual starts, from every
    round of `season` strictly before `target_round` where this arm
    produced a real (non-fallback) classification. Empty on the very first
    runs -- there's no history yet, which is exactly when CATEGORY_PRIORS
    carries all the weight.

    Joins this arm's own archived predictions (method column encodes the
    category as "agent_<category>") to actual outcomes in
    player_gameweek_stats -- only this project's own current-season archive
    has either, so there's no cross-season equivalent the way the base
    model has for prev/roll4.
    """
    df = pd.read_sql(
        "SELECT pr.method AS method, pgs.starts AS y "
        "FROM predictions pr "
        "JOIN player_gameweek_stats pgs "
        "  ON pgs.code = pr.code AND pgs.season = pr.season "
        "     AND pgs.round = pr.target_round "
        "WHERE pr.season = ? AND pr.target_round < ? AND pr.model_version = ? "
        "  AND pr.method LIKE 'agent\\_%' ESCAPE '\\'",
        conn, params=(season, target_round, model_version),
    )
    if len(df) == 0:
        return {}
    df["category"] = df["method"].str.replace("^agent_", "", regex=True)
    df = df[df["category"] != "fallback_no_news"]
    if len(df) == 0:
        return {}
    grouped = df.groupby("category")["y"].agg(["sum", "size"])
    return dict(zip(grouped.index, zip(grouped["sum"], grouped["size"])))


def shrink(prior, observed_sum, observed_n, k=SHRINKAGE_K):
    """Blend a hand-set prior with this arm's own observed outcomes for one
    category, continuously rather than behind a hard cliff: with 0
    observations this is exactly `prior`; each real observation nudges it
    by 1/(k + n) of the gap between the prior and that observation,
    converging toward the true observed rate as n grows. Equivalent to a
    Beta-prior posterior mean with `prior` expressed as `k`
    pseudo-observations.
    """
    return (prior * k + observed_sum) / (k + observed_n)


def category_to_p_start(category, category_rates, priors=None, k=SHRINKAGE_K):
    """Final p_start for one classified category. `category_rates` is
    fit_category_rates's output. Returns None for a category this module
    doesn't know how to price (caller should treat that as no_news).
    """
    if priors is None:
        priors = CATEGORY_PRIORS
    if category == "confirmed_out":
        return 0.0
    if category not in priors:
        return None
    observed_sum, observed_n = category_rates.get(category, (0.0, 0))
    return shrink(priors[category], observed_sum, observed_n, k=k)
