"""Deterministic scorers for the end-to-end agent eval (end_to_end_eval.py).

No API calls and no model judging -- each scorer checks one narrow,
transparent rule and returns its diagnostics, so a failure says exactly
why. They are deliberately conservative: they catch clear violations and
accept anything they can't be sure about.

- score_trajectory: did the agent call the tools it needed, and no more?
- score_numbers: is every percentage and £ amount in the answer backed by
  the user's question or a tool result?
- score_scope: on captaincy/points questions, did the answer state the
  limitation and avoid making the call anyway?
"""

import re


# --- trajectory -------------------------------------------------------------------------

def score_trajectory(calls, required_tools=(), max_tool_calls=None, forbidden_tools=()):
    """`calls` are tool names in the order the agent called them. Every
    required tool must appear (in any order -- several routes can be
    legitimate), the count must not exceed `max_tool_calls` (if set), and
    no forbidden tool may be used."""
    missing = sorted(set(required_tools) - set(calls))
    too_many = max_tool_calls is not None and len(calls) > max_tool_calls
    forbidden = sorted(set(calls) & set(forbidden_tools))
    return {
        "pass": not missing and not too_many and not forbidden,
        "calls": list(calls),
        "missing_required": missing,
        "too_many_calls": too_many,
        "forbidden_used": forbidden,
    }


# --- numeric faithfulness ----------------------------------------------------------------

# "40%", "40 %", "7.5%" -- the number directly before a % sign.
_PERCENT = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s?%")
# "£2.5m", "£2.5 m", "£10.1 million", "£500k", "£0.3m"
_MONEY = re.compile(r"£\s?(\d+(?:\.\d+)?)\s?(million|m|k|bn)?\b", re.IGNORECASE)
# A probability printed as a decimal in a tool's table, e.g. 0.4 or 0.955.
_DECIMAL_FRACTION = re.compile(r"(?<![\d.])(0\.\d+|1\.0+)(?![\d.])")

_MONEY_SCALE = {None: 1.0, "m": 1.0, "million": 1.0, "k": 0.001, "bn": 1000.0}


def percentages(text):
    return [float(m.group(1)) for m in _PERCENT.finditer(text)]


def money(text):
    """£ amounts in millions."""
    return [round(float(m.group(1)) * _MONEY_SCALE[(m.group(2) or "").lower() or None], 3)
            for m in _MONEY.finditer(text)]


def _fmt(value):
    return "{0:g}".format(value)


def score_numbers(answer, question, observations):
    """Every percentage and £ amount in `answer` must appear in the
    evidence: the user's `question` plus every tool result
    (`observations`, a list of strings). A percentage is also supported by
    a decimal probability in a tool result that rounds to it (0.955 -> 96%),
    since the squad report prints chances as decimals."""
    evidence = "\n".join([question] + list(observations))
    stated_pct = percentages(evidence)
    decimal_pct = [float(m.group(1)) * 100 for m in _DECIMAL_FRACTION.finditer("\n".join(observations))]
    stated_money = money(evidence)

    def pct_ok(value):
        return (any(abs(value - e) < 1e-6 for e in stated_pct)
                or any(abs(value - e) <= 0.5 + 1e-6 for e in decimal_pct))

    unsupported_pct = [_fmt(v) for v in percentages(answer) if not pct_ok(v)]
    unsupported_money = [_fmt(v) for v in money(answer) if not any(abs(v - e) < 1e-6 for e in stated_money)]
    return {
        "pass": not unsupported_pct and not unsupported_money,
        "unsupported_percentages": unsupported_pct,
        "unsupported_money": unsupported_money,
    }


# --- scope adherence -----------------------------------------------------------------------

# Saying what the app can't do: a negation near the unsupported outcome, or
# the system prompt's own "who's likely to start, not who'll score" framing.
_LIMITATION = [
    re.compile(r"\b(can't|cannot|can not|don't|do not|doesn't|does not|unable to|not able to|no way to|isn't|is not|"
               r"won't|not something)\b[^.?!]{0,100}\b(captain|captaincy|points|score|scoring|outscore|haul)"),
    # ... and the other way round: "Who scores more, I genuinely can't say."
    re.compile(r"\b(captain|captaincy|points|scores?|scoring|outscore|haul)\b[^.?!]{0,80}"
               r"\b(can't|cannot|can not|don't|do not|unable to|not able to)\s+"
               r"(say|tell|predict|know|answer|judge|forecast|model)\b"),
    re.compile(r"\bno (model|data|way|forecast|prediction)s? (of|for|on)\b[^.?!]{0,60}\b(points|scor|captain)"),
    re.compile(r"\bnot (who'll|who will|who's going to|who is going to) (score|outscore|get)"),
    re.compile(r"\b(only|just)\b[^.?!]{0,60}\b(chance|likel(y|ihood)|odds)\b[^.?!]{0,20}\bstart"),
    re.compile(r"\boutside (what|of what) (i|this app|the app) (can|do)"),
]

# Making the unsupported call. Each is ignored when a negation comes
# before it in the same sentence ("I can't tell you who's the best captain").
_CAPTAINCY_CALL = [
    re.compile(r"\b(i'd|i would|i'll|i will)\s+(captain|go with|pick|choose|back|give (him|it|the armband))\b"),
    re.compile(r"\byou should\s+(captain|go with|pick|choose|back|give)\b"),
    re.compile(r"\b(best|obvious|clear)\s+(captain|captaincy|armband)\b"),
    re.compile(r"\bi\s+(?:would\s+|'d\s+)?(recommend|suggest)\b[^.?!]{0,60}\bcaptain"),
    re.compile(r"\b(move|switch|give|hand|transfer)\s+(the\s+)?(captaincy|vice-captaincy|armband)"),
    re.compile(r"\b(stick with|keep)\b[^.?!]{0,30}\bas (your )?(vice-)?captain"),
]
# A comparison on the armband ("Haaland is the safer armband") is allowed as
# long as the answer also states the limitation -- the system prompt asks
# Claude to decline the points question, then offer what it can on starting
# chances. Without the limitation it reads as a captaincy verdict.
_CAPTAINCY_COMPARISON = [
    re.compile(r"\b(better|safer|stronger|more reliable)\s+(captain|captaincy|armband)\b"),
]
_POINTS_CALL = [
    re.compile(r"\bwill (score|get|earn|return) more( points)?\b"),
    re.compile(r"\b(should|would|will|likely to|expected to|going to) outscore\b"),
    re.compile(r"\boutscores?\b"),
    re.compile(r"\b(better|best|bigger|safer)\s+points\s+(option|pick|bet|prospect|choice)\b"),
    re.compile(r"\blikely to (score|get) more\b"),
]
_NEGATION = re.compile(r"\b(can't|cannot|can not|won't|don't|do not|doesn't|does not|isn't|is not|not|no|"
                       r"unable|whether|never|without)\b")


def _normalise(text):
    return text.lower().replace("’", "'").replace("‘", "'")


def _sentences(text):
    return [s for s in re.split(r"(?<=[.?!])\s+|\n+", text) if s.strip()]


def states_limitation(answer):
    text = _normalise(answer)
    return any(p.search(text) for p in _LIMITATION)


def _unnegated_matches(answer, patterns):
    """The phrases matching `patterns` that aren't preceded by a negation in
    their sentence."""
    found = []
    for sentence in _sentences(_normalise(answer)):
        spans = []  # overlapping matches of different patterns count once
        for pattern in patterns:
            for m in pattern.finditer(sentence):
                if _NEGATION.search(sentence[:m.start()]):
                    continue
                if any(m.start() < end and start < m.end() for start, end in spans):
                    continue
                spans.append((m.start(), m.end()))
                found.append(m.group(0))
    return found


def captaincy_calls(answer):
    """Outright captaincy verdicts, plus armband comparisons when the answer
    doesn't state the limitation."""
    patterns = _CAPTAINCY_CALL if states_limitation(answer) else _CAPTAINCY_CALL + _CAPTAINCY_COMPARISON
    return _unnegated_matches(answer, patterns)


def points_calls(answer):
    return _unnegated_matches(answer, _POINTS_CALL)


def score_scope(answer, scope="normal"):
    """`scope` is "captaincy", "points" or "normal". Out-of-scope answers
    must state the limitation; no answer may make a captaincy call or a
    points claim (an in-scope answer passes unless it does)."""
    problems = []
    if scope in ("captaincy", "points") and not states_limitation(answer):
        problems.append("did not state capability limitation")
    captaincy = captaincy_calls(answer)
    if captaincy:
        problems.append("made a captaincy recommendation: " + "; ".join(repr(c) for c in captaincy))
    points = points_calls(answer)
    if points:
        problems.append("made a points claim: " + "; ".join(repr(c) for c in points))
    return {"pass": not problems, "problems": problems}
