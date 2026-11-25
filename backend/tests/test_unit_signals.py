
from unittest.mock import MagicMock, patch

import pytest
import transformers


# Keep unit tests lightweight: prevent large model weights from loading.
def fake_model(*args, **kwargs):
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    model.parameters.return_value = []
    return model


with (
    patch.object(
        transformers.AutoTokenizer,
        "from_pretrained",
        return_value=MagicMock(),
    ),
    patch.object(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        side_effect=fake_model,
    ),
):
    from backend import pipeline
    from backend.lgs_pipeline import local_confidence


# ---------------------------------------------------------
# UNIT TESTS: LOCAL CONFIDENCE
# ---------------------------------------------------------

@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.50, 0.0),
        (0.00, 1.0),
        (1.00, 1.0),
        (0.25, 0.5),
        (0.75, 0.5),
    ],
)
def test_local_confidence(score, expected):
    assert local_confidence(score) == pytest.approx(expected)


# ---------------------------------------------------------
# UNIT TESTS: EXPLICIT OFFENSIVE-LANGUAGE SIGNAL
# ---------------------------------------------------------

def test_offensive_signal_for_empty_message():
    assert pipeline.get_offensive_signal("") == 0.0


def test_offensive_signal_for_whitespace():
    assert pipeline.get_offensive_signal("   ") == 0.0


def test_offensive_signal_for_harmless_message():
    assert pipeline.get_offensive_signal("Have a nice day") == 0.0


def test_offensive_signal_for_one_matching_term():
    assert pipeline.get_offensive_signal("You are stupid") == 0.75


def test_offensive_signal_is_case_insensitive():
    assert pipeline.get_offensive_signal("You are STUPID") == 0.75


def test_offensive_signal_for_two_matching_terms():
    assert pipeline.get_offensive_signal("stupid idiot") == 0.90


def test_offensive_signal_for_three_matching_terms():
    assert pipeline.get_offensive_signal(
        "stupid idiot worthless"
    ) == 1.0


@pytest.mark.parametrize(
    "history",
    [
        [],
        [{"sender": "user", "text": "Hello"}],
    ],
)
def test_reciprocity_returns_neutral_score_for_short_history(history):
    score, details = pipeline.get_reciprocity_score(history)

    assert score == pytest.approx(0.5)
    assert details == {}


@pytest.mark.parametrize(
    "history",
    [
        [],
        [{"sender": "user", "text": "Hello"}],
    ],
)
def test_sentiment_trajectory_returns_neutral_score_for_short_history(history):
    score, trajectory = pipeline.get_sentiment_trajectory(history)

    assert score == pytest.approx(0.5)
    assert trajectory == []
# ---------------------------------------------------------
# STEP 2: word-boundary matching and severity tiers
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "my grandpa died last week",
        "i am on a diet",
        "we ordered a salad",
        "i failed my exam",
        "check abc and mcdonalds",
        "kuch nahi yaar, just tired",
        "i studied all night",
    ],
)
def test_offensive_signal_ignores_substrings_inside_words(text):
    assert pipeline.get_offensive_signal(text) == 0.0


@pytest.mark.parametrize(
    "text",
    ["loser", "LOSERRRRR", "hiiiii....loserrrrr....nvm...i was jk", "i hate mondays"],
)
def test_single_mild_term_is_weak_signal(text):
    assert pipeline.get_offensive_signal(text) == pytest.approx(0.35)


def test_two_mild_terms_are_still_below_standard_tier():
    assert pipeline.get_offensive_signal("loser dumb") == pytest.approx(0.50)


def test_mild_term_next_to_real_insult_adds_one_hit():
    assert pipeline.get_offensive_signal("stupid loser") == pytest.approx(0.90)


def test_elongated_standard_insult_still_detected():
    assert pipeline.get_offensive_signal("you are STUPIDDDD") == pytest.approx(0.75)


def test_hate_you_is_standard_not_double_counted():
    assert pipeline.get_offensive_signal("i hate you") == pytest.approx(0.75)


def test_threatening_phrases_remain_strong():
    assert pipeline.get_offensive_signal("nobody likes you") == pytest.approx(0.75)
    assert pipeline.get_offensive_signal("you are pathetic and worthless") >= 0.90


def test_mutual_mild_teasing_counts_as_reciprocal():
    history = [
        {"sender": "a", "text": "lol loser"},
        {"sender": "b", "text": "lol loser"},
    ]
    score, _ = pipeline.get_reciprocity_score(history)
    assert score == pytest.approx(1.0)

# ---------------------------------------------------------
# STEP 3: banter signal v2
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    ["that's my brother", "the vase is broken", "he is a dudebro", "have a nice day"],
)
def test_banter_ignores_substrings_inside_words(text):
    assert pipeline.get_banter_signal(text, []) == 0.0


@pytest.mark.parametrize(
    "text",
    ["looool", "lmaooo", "LOLLL", "hahahaha", "heheh", "xdd", "😂"],
)
def test_banter_detects_laughter_even_when_elongated(text):
    assert pipeline.get_banter_signal(text, []) >= 0.35


@pytest.mark.parametrize(
    "text", ["nvm i was jk", "just kidding", "i was joking", "jkjk"]
)
def test_banter_detects_explicit_joke_phrases(text):
    assert pipeline.get_banter_signal(text, []) >= 0.35


def test_slang_and_elongation_alone_stay_weak():
    # casual register is only weak evidence; it must not reach 0.5 by itself
    assert pipeline.get_banter_signal("supppp bro ik dw", []) < 0.5


def test_joke_phrase_in_recent_history_raises_banter():
    alone = pipeline.get_banter_signal("supppp ik dw", [])
    with_history = pipeline.get_banter_signal(
        "supppp ik dw",
        [{"sender": "a", "text": "nvm i was jk"}],
    )
    assert with_history >= alone + 0.25 - 1e-9
    assert with_history >= 0.5


def test_banter_looks_at_last_four_history_messages():
    history = [{"sender": "a", "text": "just kidding lol"}] + [
        {"sender": "b", "text": "ok"} for _ in range(3)
    ]
    assert pipeline.get_banter_signal("hey", history) >= 0.25


def test_hostile_message_without_playful_markers_has_no_banter():
    for text in [
        "you are such a loser",
        "nobody likes you, everyone laughs at you",
        "i know where you live and i will hurt you after school",
    ]:
        assert pipeline.get_banter_signal(text, []) == 0.0


def test_laughter_helper_matches_whole_words_only():
    assert pipeline._has_laughter("that was lollll")
    assert not pipeline._has_laughter("lollipop")
    assert not pipeline._has_laughter("xdr config")

# ---------------------------------------------------------
# STEP 4: sentiment trajectory
# ---------------------------------------------------------

def _traj(monkeypatch, values):
    """Run get_sentiment_trajectory with scripted VADER compound scores."""
    lookup = {f"m{i}": v for i, v in enumerate(values)}
    monkeypatch.setattr(
        pipeline.vader, "polarity_scores",
        lambda text: {"compound": lookup[text]},
    )
    history = [
        {"sender": "a" if i % 2 == 0 else "b", "text": f"m{i}"}
        for i in range(len(values))
    ]
    return pipeline.get_sentiment_trajectory(history)[0]


def test_two_messages_are_not_enough_for_a_trajectory(monkeypatch):
    assert _traj(monkeypatch, [0.0, -0.6]) == pytest.approx(0.5)


def test_flat_sentiment_is_neutral_not_declining(monkeypatch):
    assert _traj(monkeypatch, [0.0, 0.0, 0.0, 0.0]) == pytest.approx(0.5)
    assert _traj(monkeypatch, [0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_steady_decline_scores_low(monkeypatch):
    assert _traj(monkeypatch, [0.5, -0.3, -0.8]) < 0.25


def test_decline_scores_lower_than_oscillation(monkeypatch):
    decline = _traj(monkeypatch, [0.5, 0.0, -0.4, -0.8])
    oscillating = _traj(monkeypatch, [0.6, -0.4, 0.5, -0.3])
    assert decline < oscillating
    assert oscillating > 0.5


def test_sudden_hostility_after_neutral_scores_low(monkeypatch):
    assert _traj(monkeypatch, [0.0, 0.0, -0.7]) < 0.25
# ---------------------------------------------------------
# STEP 5: threat language detector (blocks the banter override)
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "i will hurt you after school",
        "i know where you live",
        "i'll come after you",
        "kill youuu",
        "lol i will hurt u jk",
        "kys",
        "go die",
        "kill yourself",
    ],
)
def test_threat_language_detected(text):
    assert pipeline.has_threat_language(text)


@pytest.mark.parametrize(
    "text",
    [
        "i will find you a pizza",
        "this homework will kill me",
        "see you after school",
        "you look great",
        "supppp loser....ik dw",
        "",
    ],
)
def test_benign_text_is_not_threat_language(text):
    assert not pipeline.has_threat_language(text)
# ---------------------------------------------------------
# STEP 7: defensive matcher and own-text adjustment
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "please stop, that really hurts",
        "i didn't do anything to you, leave me alone",
        "PLEASE STOP",
        "why are you doing this to me",
        "that’s not true",
    ],
)
def test_defensive_language_detected(text):
    assert pipeline.has_defensive_language(text)


@pytest.mark.parametrize(
    "text",
    ["haha stoppp 😂", "you are such a loser", "bus stop is near", "why men do this", ""],
)
def test_non_defensive_text_not_matched(text):
    assert not pipeline.has_defensive_language(text)


def test_reciprocity_counts_apostrophe_defensive_message():
    _, details = pipeline.get_reciprocity_score(
        [
            {"sender": "a", "text": "you are a loser"},
            {"sender": "b", "text": "i didn't do anything, leave me alone"},
        ]
    )
    assert details["b"]["defensive"] == 1


def _own(**overrides):
    args = dict(
        contextual_local=0.5, contextual_global=0.5,
        isolated_local=0.5, isolated_global=0.5,
        offensive_signal=0.0, own_banter=0.0,
        defensive_language=False, threat_language=False,
        sender_offensive_count=0,
    )
    args.update(overrides)
    return pipeline.own_text_adjustment(**args)


def test_floor_restores_threat_scored_low_by_friendly_history():
    local, glob, reason = _own(
        contextual_local=0.83, contextual_global=0.01,
        isolated_local=0.93, isolated_global=0.96, threat_language=True,
    )
    assert reason == "own_text_floor"
    assert (local, glob) == (0.93, 0.96)


def test_floor_requires_no_playful_marker_in_message_itself():
    assert _own(
        contextual_local=0.83, contextual_global=0.01,
        isolated_local=0.93, isolated_global=0.96, own_banter=0.35,
    )[2] is None


def test_floor_requires_strong_isolated_score():
    assert _own(
        contextual_local=0.2, contextual_global=0.1,
        isolated_local=0.6, isolated_global=0.6,
    )[2] is None


def test_floor_never_lowers_scores():
    local, glob, _ = _own(
        contextual_local=0.99, contextual_global=0.99,
        isolated_local=0.9, isolated_global=0.9,
    )
    assert (local, glob) == (0.99, 0.99)


def test_victim_cap_uses_own_text_scores():
    local, glob, reason = _own(
        contextual_local=0.96, contextual_global=0.97,
        isolated_local=0.48, isolated_global=0.08, defensive_language=True,
    )
    assert reason == "victim_own_text"
    assert (local, glob) == (0.48, 0.08)


@pytest.mark.parametrize(
    "change",
    [
        {"sender_offensive_count": 1},   # the defender was abusive earlier
        {"offensive_signal": 0.35},      # message has an offensive term
        {"threat_language": True},       # message contains a threat
        {"isolated_local": 0.9, "isolated_global": 0.9},  # not benign alone
    ],
)
def test_victim_cap_requires_every_guard(change):
    base = dict(
        contextual_local=0.97, contextual_global=0.98,
        isolated_local=0.01, isolated_global=0.05, defensive_language=True,
    )
    base.update(change)
    assert _own(**base)[2] != "victim_own_text"
# ---------------------------------------------------------
# STEP 6: global memory gating
# ---------------------------------------------------------

def _mem(label, similarity):
    return {"label": label, "similarity": similarity}


def test_memory_ignores_distant_neighbours():
    # moderate similarity is typical for any text and carries no evidence
    memories = [_mem(1, 0.8), _mem(1, 0.7), _mem(0, 0.6)]
    assert pipeline.global_memory_evidence(memories) is None


def test_memory_uses_close_harmful_majority():
    memories = [_mem(1, 0.95), _mem("1", 0.92), _mem(0, 0.99)]
    assert pipeline.global_memory_evidence(memories) == pytest.approx(0.95)


def test_benign_close_neighbours_count_against_memory():
    memories = [_mem(1, 0.93), _mem(0, 0.99), _mem(0, 0.95)]
    assert pipeline.global_memory_evidence(memories) is None


def test_memory_never_reinforces_in_playful_context():
    memories = [_mem(1, 0.95), _mem(1, 0.96)]
    assert pipeline.global_memory_evidence(memories, banter_signal=0.55) is None
    assert pipeline.global_memory_evidence(memories, banter_signal=0.45) == pytest.approx(0.96)


def test_memory_handles_empty_and_bad_data():
    assert pipeline.global_memory_evidence([]) is None
    assert pipeline.global_memory_evidence(None) is None
    assert pipeline.global_memory_evidence([_mem("x", 0.99), _mem(None, 0.98)]) is None
    assert pipeline.global_memory_evidence([_mem(1, 1.7)]) == pytest.approx(1.0)