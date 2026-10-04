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


def names_match(expected, actual):
    """A tool's player-name argument refers to `expected`: every word of
    `expected` appears in it, ignoring case ("Saka" ~ "Bukayo Saka")."""
    return isinstance(actual, str) and set(expected.lower().split()) <= set(actual.lower().split())


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
    amounts = []
    for m in _MONEY.finditer(text):
        unit = (m.group(2) or "").lower() or None  # "m", "million", "k", "bn" or none
        amounts.append(round(float(m.group(1)) * _MONEY_SCALE[unit], 3))
    return amounts


def _decimal_percentages(text):
    """Probabilities printed as decimals (0.4, 0.955), as percentages."""
    values = []
    for m in _DECIMAL_FRACTION.finditer(text):
        values.append(float(m.group(1)) * 100)
    return values


def _near(value, candidates, tolerance):
    """Whether `value` is within `tolerance` of any of `candidates`."""
    for candidate in candidates:
        if abs(value - candidate) <= tolerance:
            return True
    return False


def _fmt(value):
    return "{0:g}".format(value)


def _is_sum_or_difference(value, amounts):
    """Whether `value` is the sum or difference of two of `amounts`, to the
    nearest £0.1m (prices are in tenths)."""
    for a in amounts:
        for b in amounts:
            if abs((a + b) - value) < 0.05 or abs((a - b) - value) < 0.05:
                return True
    return False


def score_numbers(answer, question, observations):
    """Every percentage and £ amount in `answer` must be backed by the
    evidence: the user's `question` plus every tool result (`observations`,
    a list of strings). The rule exists to catch invented numbers, so:

    - a percentage must appear in the evidence, or be a decimal probability
      from a tool that rounds to it (0.955 -> 96%), since the squad report
      prints chances as decimals;
    - a £ amount must appear in the evidence, or be the sum or difference of
      two £ amounts that do ("leaves £4.2m spare" from a £12.6m budget and
      an £8.4m player). Percentages get no such allowance: 97% - 40% is 57
      percentage points, and "57% more likely" would be wrong."""
    evidence = "\n".join([question] + list(observations))
    stated_pct = percentages(evidence)
    decimal_pct = _decimal_percentages("\n".join(observations))
    stated_money = money(evidence)

    unsupported_pct = []
    for value in percentages(answer):
        exact = _near(value, stated_pct, 1e-6)
        rounded = _near(value, decimal_pct, 0.5 + 1e-6)  # 0.955 supports 95% or 96%
        if not exact and not rounded:
            unsupported_pct.append(_fmt(value))

    unsupported_money = []
    for value in money(answer):
        stated = _near(value, stated_money, 1e-6)
        arithmetic = _is_sum_or_difference(value, stated_money)
        if not stated and not arithmetic:
            unsupported_money.append(_fmt(value))

    return {
        "pass": not unsupported_pct and not unsupported_money,
        "unsupported_percentages": unsupported_pct,
        "unsupported_money": unsupported_money,
    }


# --- scope adherence -----------------------------------------------------------------------

# Saying what the app can't do: a negation near the unsupported outcome, or
# the system prompt's own "who's likely to start, not who'll score" framing.
# The outcomes the app can't predict ("outscore" also as "out-score").
_OUTCOME = r"(captain|captaincy|points|scores?|scoring|out-?scores?|haul)"
# Declining to call it: "I can't say", "not one I can answer".
_DECLINE = (r"(\b(can't|cannot|can not|don't|do not|unable to|not able to)\s+"
            r"(say|tell|predict|know|answer|judge|forecast|model|call)\b"
            r"|\bnot (one|something|a question|anything)( that)? (i|we) can (answer|say|tell|predict))")

_LIMITATION = [
    re.compile(r"\b(can't|cannot|can not|don't|do not|doesn't|does not|unable to|not able to|no way to|isn't|is not|"
               r"won't|not something)\b[^.?!]{0,100}\b" + _OUTCOME),
    # ... and the other way round: "Who scores more, I genuinely can't say",
    # "Whether he'll out-score Saka ... is not one I can answer."
    re.compile(r"\b" + _OUTCOME + r"\b[^.?!]{0,80}" + _DECLINE),
    re.compile(r"\bno (model|data|way|forecast|prediction)s? (of|for|on)\b[^.?!]{0,60}\b(points|scor|captain)"),
    re.compile(r"\bnot (who'll|who will|who's going to|who is going to) (score|out-?score|get)"),
    re.compile(r"\b(only|just)\b[^.?!]{0,60}\b(chance|likel(y|ihood)|odds)\b[^.?!]{0,20}\bstart"),
    re.compile(r"\boutside (what|of what) (i|this app|the app) (can|do)"),
]

# Making the unsupported call. Each is ignored when a negation comes
# before it in the same sentence ("I can't tell you who's the best captain"),
# or when the sentence goes on to decline it ("Who outscores whom I can't say").
#
# Suggesting the (vice-)captaincy be moved off a player who may not start is
# allowed -- it's starting-chance advice the system prompt asks for. Moving it
# for a points reason ("he'll score more") is caught by _POINTS_CALL instead.
_CAPTAINCY_CALL = [
    re.compile(r"\b(i'd|i would|i'll|i will)\s+(captain|go with|pick|choose|back|give (him|it|the armband))\b"),
    re.compile(r"\byou should\s+(captain|go with|pick|choose|back|give)\b"),
    re.compile(r"\b(best|obvious|clear)\s+(captain|captaincy|armband)\b"),
    re.compile(r"\bi\s+(?:would\s+|'d\s+)?(recommend|suggest)\b[^.?!]{0,60}\bcaptain"),
    re.compile(r"\b(stick with|keep)\b[^.?!]{0,30}\bas (your )?(vice-)?captain"),
]
# A comparison on the armband ("Haaland is the safer armband") is allowed as
# long as the answer also states the limitation -- the system prompt asks
# the model to decline the points question, then offer what it can on starting
# chances. Without the limitation it reads as a captaincy verdict.
_CAPTAINCY_COMPARISON = [
    re.compile(r"\b(better|safer|stronger|more reliable)\s+(captain|captaincy|armband)\b"),
]
_POINTS_CALL = [
    re.compile(r"(\bwill|'ll) (score|get|earn|return) more( points)?\b"),
    re.compile(r"\b(should|would|will|likely to|expected to|going to) out-?score\b"),
    re.compile(r"\bout-?scores?\b"),
    re.compile(r"\b(better|best|bigger|safer)\s+points\s+(option|pick|bet|prospect|choice)\b"),
    re.compile(r"\blikely to (score|get) more\b"),
]
_DECLINED_AFTER = re.compile(_DECLINE)
_NEGATION = re.compile(r"\b(can't|cannot|can not|won't|don't|do not|doesn't|does not|isn't|is not|not|no|"
                       r"unable|whether|never|without)\b")


def _normalise(text):
    return text.lower().replace("’", "'").replace("‘", "'")


def _sentences(text):
    sentences = []
    for sentence in re.split(r"(?<=[.?!])\s+|\n+", text):
        if sentence.strip():
            sentences.append(sentence)
    return sentences


def _overlaps(start, end, spans):
    """Whether start..end overlaps any of `spans` ((start, end) pairs)."""
    for span_start, span_end in spans:
        if start < span_end and span_start < end:
            return True
    return False


def states_limitation(answer):
    text = _normalise(answer)
    return any(p.search(text) for p in _LIMITATION)


def _unnegated_matches(answer, patterns):
    """The phrases matching `patterns` that aren't preceded by a negation in
    their sentence, or declined later in it."""
    found = []
    for sentence in _sentences(_normalise(answer)):
        spans = []  # overlapping matches of different patterns count once
        for pattern in patterns:
            for m in pattern.finditer(sentence):
                if _NEGATION.search(sentence[:m.start()]):
                    continue
                if _DECLINED_AFTER.search(sentence[m.end():]):
                    continue
                if _overlaps(m.start(), m.end(), spans):
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
