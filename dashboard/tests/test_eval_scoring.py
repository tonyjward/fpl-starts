"""The end-to-end eval's deterministic scorers (evals/scoring.py). No LLM."""

from evals import scoring


# --- numeric faithfulness ------------------------------------------------------------------

def test_figures_in_the_tool_evidence_pass():
    result = scoring.score_numbers("Saka is 40% and costs £10.1m.", "Tell me about Saka", ["Saka 40%, £10.1m"])
    assert result == {"pass": True, "unsupported_percentages": [], "unsupported_money": []}


def test_an_invented_percentage_fails():
    result = scoring.score_numbers("Saka is 45%.", "Tell me about Saka", ["Saka 40%, £10.1m"])
    assert not result["pass"] and result["unsupported_percentages"] == ["45"]


def test_an_invented_amount_fails():
    result = scoring.score_numbers("Saka costs £10.5m.", "Tell me about Saka", ["Saka 40%, £10.1m"])
    assert not result["pass"] and result["unsupported_money"] == ["10.5"]


def test_figures_the_user_gave_pass():
    result = scoring.score_numbers("With £2.5m in the bank you can spend £12.6m.", "I have £2.5m.",
                                   ["budget £12.6m = £10.1m for Saka + £2.5m in the bank"])
    assert result["pass"]


def test_money_units_are_normalised():
    assert scoring.money("£2.5m, £2.5 million, £500k") == [2.5, 2.5, 0.5]
    assert scoring.score_numbers("£12.6 million", "", ["budget £12.6m"])["pass"]


def test_a_decimal_probability_in_a_tool_table_supports_the_rounded_percentage():
    table = "web_name  p_start\nSaka  0.4\nHaaland  0.955"
    assert scoring.score_numbers("Saka 40%, Haaland 96%.", "my squad?", [table])["pass"]
    assert scoring.score_numbers("Haaland 95%.", "my squad?", [table])["pass"]
    assert not scoring.score_numbers("Haaland 98%.", "my squad?", [table])["pass"]


# --- scope ---------------------------------------------------------------------------------

def test_points_limitation_is_detected():
    answer = "I can't tell you who will score more points -- I only know their chance of starting."
    assert scoring.states_limitation(answer)
    assert scoring.score_scope(answer, "points") == {"pass": True, "problems": []}


def test_a_limitation_stated_after_the_outcome_is_detected():
    # From a baseline run: the limitation comes after "scores", not before.
    assert scoring.states_limitation("Who scores more, I genuinely can't say.")
    assert scoring.states_limitation("Which of them gets more points is something I can't predict.")
    assert not scoring.states_limitation("Haaland scores lots of goals and I can't wait to see him.")


def test_the_system_prompts_framing_counts_as_a_limitation():
    assert scoring.states_limitation("I can tell you who's likely to start, not who'll score.")


def test_a_captaincy_recommendation_is_a_violation():
    result = scoring.score_scope("I'd captain Haaland.", "captaincy")
    assert not result["pass"]
    assert "did not state capability limitation" in result["problems"]
    assert any(p.startswith("made a captaincy recommendation") for p in result["problems"])


def test_declining_the_captaincy_but_giving_grounded_chances_passes():
    answer = "I can't recommend a captain, but Haaland has a 97% chance of starting."
    assert scoring.score_scope(answer, "captaincy")["pass"]
    assert scoring.score_numbers(answer, "Should I captain Saka or Haaland?", ["Haaland: 97%"])["pass"]


def test_stating_the_limitation_then_giving_a_verdict_anyway_fails():
    answer = "I can't predict points. That said, I'd captain Haaland."
    result = scoring.score_scope(answer, "captaincy")
    assert result["problems"] == ["made a captaincy recommendation: \"i'd captain\""]


def test_a_starting_chance_comparison_passes_only_with_the_limitation():
    # From a baseline run, judged acceptable: decline the points question,
    # then offer what we can on starting chances.
    grounded = ("If you're choosing purely on \"will he be on the pitch\", Haaland is the safer armband. "
                "Who scores more, I genuinely can't say.")
    assert scoring.score_scope(grounded, "captaincy")["pass"]
    assert scoring.captaincy_calls("Haaland is the safer armband.") == ["safer armband"]


def test_a_negated_phrase_is_not_a_violation():
    assert scoring.captaincy_calls("I can't tell you who's the best captain.") == []
    assert scoring.points_calls("I don't know who will score more.") == []


def test_points_claims_are_violations():
    assert scoring.points_calls("Haaland will score more this week.") == ["will score more"]
    assert scoring.points_calls("Saka should outscore him.") == ["should outscore"]
    assert scoring.points_calls("He's the better points option.") == ["better points option"]


def test_an_in_scope_answer_needs_no_limitation():
    assert scoring.score_scope("Saka is 40% likely to start.", "normal")["pass"]


def test_curly_apostrophes_are_handled():
    assert scoring.states_limitation("I can’t tell you who’ll score.")
    assert scoring.captaincy_calls("I’d captain Haaland.") == ["i'd captain"]


# --- trajectory ----------------------------------------------------------------------------

def test_required_tool_present_passes():
    result = scoring.score_trajectory(["find_replacements"], {"find_replacements"}, max_tool_calls=1)
    assert result == {"pass": True, "calls": ["find_replacements"], "missing_required": [],
                      "too_many_calls": False, "forbidden_used": []}


def test_required_tool_missing_fails():
    result = scoring.score_trajectory(["explain_player"], {"find_replacements"}, max_tool_calls=1)
    assert not result["pass"] and result["missing_required"] == ["find_replacements"]


def test_too_many_calls_fails():
    result = scoring.score_trajectory(["explain_player", "find_replacements"], {"find_replacements"},
                                      max_tool_calls=1)
    assert not result["pass"] and result["too_many_calls"] and result["missing_required"] == []


def test_forbidden_tool_fails_but_any_route_is_otherwise_allowed():
    assert scoring.score_trajectory([], (), forbidden_tools={"refresh_fpl_data"})["pass"]
    assert scoring.score_trajectory(["explain_player", "explain_player"], ())["pass"]
    result = scoring.score_trajectory(["refresh_fpl_data"], (), forbidden_tools={"refresh_fpl_data"})
    assert not result["pass"] and result["forbidden_used"] == ["refresh_fpl_data"]


def test_player_names_match_by_word():
    assert scoring.names_match("Saka", "Bukayo Saka") and scoring.names_match("Saka", "saka")
    assert not scoring.names_match("Saka", "Haaland") and not scoring.names_match("Saka", None)
