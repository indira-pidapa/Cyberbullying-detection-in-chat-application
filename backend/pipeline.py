"""
pipeline.py
All classification pipeline functions ported from Colab.
Loaded once at startup by main.py — do not import heavy libraries at the top
of other files, import from here instead.
"""
import re
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.special import softmax
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent   # cyberbullying_app/
MODEL_PATH = BASE_DIR / "model" / "global_model" / "muril_cyberbullying"
MAX_LEN    = 128
THRESHOLD  = 0.7
TEMPERATURE = 1.5

# ── Device ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Global model/tokenizer (loaded once at startup) ───────────────────────
print(f"Loading MuRIL from {MODEL_PATH} ...")
muril_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
muril_model     = AutoModelForSequenceClassification.from_pretrained(
                      str(MODEL_PATH)
                  ).to(DEVICE)
muril_model.eval()
print(f"✅ MuRIL loaded on {DEVICE}")

# ── Local model ────────────────────────────────────────────────────────────
LOCAL_MODEL_PATH = (
    BASE_DIR / "model" / "local_model" / "muril_student_4layer"
)

print(f"Loading Local Model from {LOCAL_MODEL_PATH} ...")

local_tokenizer = AutoTokenizer.from_pretrained(
    str(LOCAL_MODEL_PATH)
)

local_model = AutoModelForSequenceClassification.from_pretrained(
    str(LOCAL_MODEL_PATH)
).to(DEVICE)

local_model.eval()

print(
    f"✅ Local Model loaded: "
    f"{sum(p.numel() for p in local_model.parameters()) / 1e6:.1f}M parameters"
)

vader = SentimentIntensityAnalyzer()
print("✅ VADER loaded")

# ── Banter/laughter signals ───────────────────────────────────────────────
LAUGH_EMOJIS  = ['😂', '🤣', '💀', '☠️', '😆', '😛', '😜', '🙃', '😄']

# Whole-word, elongation-tolerant patterns.
_LAUGH_RE = re.compile(
    r"\b(?:l+o+l+z*|lmf?a+o+|rofl|(?:ha+){2,}h*|(?:he+){2,}h*|x+d+)\b"
)
_JOKE_RE = re.compile(
    r"\b(?:j+k+(?:j+k+)*|kidding|joking|just playing|just teasing"
    r"|only teasing|just a joke|it was a joke|was a joke|not serious)\b"
)


def _squeeze(s: str) -> str:
    return re.sub(r"(.)\1+", r"\1", s)


# Casual / playful register markers (weak evidence; squeezed form).
_SLANG_TOKENS = {
    _squeeze(w) for w in
    ('sup', 'wassup', 'wazzup', 'nvm', 'dw', 'ik', 'ikr', 'bro', 'bruh',
     'dude', 'bestie', 'yaar')
}
_SLANG_PHRASES = [_squeeze(p) for p in
                  ('ok ok', 'fine fine', 'you win', 'u win', 'ik ik')]
_ELONGATED_RE = re.compile(r"\b\w*([a-z])\1{2,}\w*\b")


def _has_laughter(text: str) -> bool:
    """True if the text contains laughter (emoji or whole-word laughter)."""
    if any(e in text for e in LAUGH_EMOJIS):
        return True
    return bool(_LAUGH_RE.search(text.lower()))

# ── Offensive signals (English + Hinglish) ────────────────────────────────
# STANDARD: one hit is already a meaningful insult / abuse signal.
OFFENSIVE_STANDARD = [
    'stupid', 'idiot', 'pathetic', 'shut up', 'nobody likes', 'worthless',
    'ugly', 'disgusting', 'kill yourself', 'kys', 'go die', 'just die',
    'freak', 'hate you', 'hate u',
    'bc', 'mc', 'teri toh', 'chup kar', 'izzat nahi', 'koi nahi chahta',
    'tu kuch nahi', 'tum kuch nahi',
]
# MILD: very common in friendly teasing -> weak evidence only.
OFFENSIVE_MILD = [
    'loser', 'dumb', 'hate', 'fail', 'bakwas', 'bekar', 'nikal',
    'sala', 'saali', 'bewakoof', 'gadha', 'ullu',
]


def _term_regex(term: str):
    """Whole-word pattern that tolerates elongation: 'loser' matches 'loserrrr',
    'saali' matches 'saaaali', but 'salad' / 'died' / 'mcdonalds' do not match."""
    words = []
    for word in term.split():
        squeezed = re.sub(r"(.)\1+", r"\1", word)   # collapse doubled letters
        words.append("".join(re.escape(c) + "+" for c in squeezed))
    return re.compile(r"\b" + r"\s+".join(words) + r"\b")


_STANDARD_RE = [_term_regex(t) for t in OFFENSIVE_STANDARD]
_MILD_RE = [_term_regex(t) for t in OFFENSIVE_MILD]


def _offensive_hits(text: str) -> tuple:
    """Returns (standard_hits, mild_hits). Each term counts once per message.
    Matched standard phrases are removed first so 'hate you' is not
    double-counted as 'hate'."""
    residual = text.lower()
    standard = 0
    for pattern in _STANDARD_RE:
        residual, n = pattern.subn(" ", residual)
        standard += 1 if n else 0
    mild = 0
    for pattern in _MILD_RE:
        residual, n = pattern.subn(" ", residual)
        mild += 1 if n else 0
    return standard, mild


def count_offensive_terms(text: str) -> int:
    standard, mild = _offensive_hits(text)
    return standard + mild

# Threat / self-harm-directed language. Used ONLY to block the banter
# override (it never raises a score by itself).
_THREAT_RE = re.compile(
    r"\b(?:kill|hurt|beat|stab|shoot|murder|punch|slap)\s+(?:you+|u+)\b"
    r"|\bwhere\s+(?:you+|u+)\s+live\b"
    r"|\bcome\s+after\s+(?:you+|u+)\b"
    r"|\bkill\s*(?:yourself|urself)\b"
    r"|\bkys\b"
    r"|\b(?:go|just)\s+die\b"
)


def has_threat_language(text: str) -> bool:
    return bool(text) and bool(_THREAT_RE.search(text.lower()))
def get_offensive_signal(text: str) -> float:
    """
    Detect explicit offensive / abusive language.

    0.0       no offensive term
    0.35/0.50 only mild, teasing-level terms (1 / 2+)
    0.75      one standard insult
    0.90      two
    1.00      three or more
    A mild term alongside a real insult adds at most one extra hit.
    """
    if not text or not text.strip():
        return 0.0

    standard, mild = _offensive_hits(text)

    if standard == 0 and mild == 0:
        return 0.0

    if standard == 0:
        return 0.35 if mild == 1 else 0.50

    total = standard + (1 if mild else 0)

    if total == 1:
        return 0.75
    if total == 2:
        return 0.90
    return 1.0
# ─────────────────────────────────────────────────────────────────────────
# SIGNAL 1 — MuRIL harmful score
# ─────────────────────────────────────────────────────────────────────────
def get_muril_score(text: str, conversation_history: list = None) -> float:
    """
    Returns probability (0→1) that the message is harmful.
    Uses [SEP]-concatenated context if history is provided.
    """
    if conversation_history:
        recent     = [m if isinstance(m, str) else m.get('text', '')
                      for m in conversation_history[-4:]]
        input_text = f" {muril_tokenizer.sep_token} ".join(recent + [text])
    else:
        input_text = text

    encoding = muril_tokenizer(
        input_text,
        max_length     = MAX_LEN,
        padding        = 'max_length',
        truncation     = True,
        return_tensors = 'pt'
    )
    input_ids      = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        output        = muril_model(input_ids=input_ids,
                                    attention_mask=attention_mask)
        scaled_logits = output.logits / TEMPERATURE
        probs         = softmax(scaled_logits.cpu().numpy()[0])

    return float(probs[1])   # probability of HARMFUL


# ─────────────────────────────────────────────────────────────────────────
# SIGNAL 2 — Banter signal (emoji + laughter detector)
# ─────────────────────────────────────────────────────────────────────────
def get_banter_signal(text: str, conversation_history: list = None) -> float:
    """
    Detect playful banter in the current message and recent context.
    Returns 0.0 (no banter) to 1.0 (strong banter).

    Current message:
        laughter emoji 0.40 each | laughter token 0.35 (max 2)
        explicit joke phrase 0.35 (once) | slang 0.10 each (max 0.30)
        elongated word 0.10 (once)
    Recent history (last 4 messages, any sender):
        joke phrase +0.25 (once) | laughter +0.10 per message (max 0.20)
    """
    text_lower = (text or "").lower()

    emoji_hits = sum(1 for e in LAUGH_EMOJIS if e in text)
    laugh_hits = min(2, len(_LAUGH_RE.findall(text_lower)))
    joke_hit = 1 if _JOKE_RE.search(text_lower) else 0

    squeezed = _squeeze(text_lower)
    squeezed_words = set(re.findall(r"[a-z0-9']+", squeezed))
    slang_hits = len(_SLANG_TOKENS & squeezed_words)
    slang_hits += sum(
        1 for p in _SLANG_PHRASES
        if re.search(r"\b" + re.escape(p) + r"\b", squeezed)
    )
    elongated = 1 if _ELONGATED_RE.search(text_lower) else 0

    current_score = min(
        1.0,
        emoji_hits * 0.40
        + laugh_hits * 0.35
        + joke_hit * 0.35
        + min(0.30, slang_hits * 0.10)
        + elongated * 0.10,
    )

    history_joke = 0.0
    history_laugh = 0.0
    if conversation_history:
        for msg in conversation_history[-4:]:
            raw = msg.get("text", "") if isinstance(msg, dict) else str(msg)
            history_text = raw.lower()
            if _JOKE_RE.search(history_text):
                history_joke = 0.25
            if _has_laughter(raw):
                history_laugh = min(0.20, history_laugh + 0.10)

    return round(min(1.0, current_score + history_joke + history_laugh), 4)
# ─────────────────────────────────────────────────────────────────────────
# SIGNAL 3 — Reciprocity score
# ─────────────────────────────────────────────────────────────────────────
DEFENSIVE_SIGNALS = [

    # --- Core Direct Defensive ---
    'thats not true', 'thats not fair', 'just stop', 'please stop',
    'leave me alone', 'thats enough', 'i said stop', 'stop it',
    'thats harsh', 'thats mean', 'ur being mean', 'you are being mean',
    'that doesnt make it okay', 'im not weird', 'just leave me',
    'make it stop', 'seriously stop',

    # --- Strong Pushback & Boundaries ---
    'idc just stop', 'can you not', 'can u not', 'just dont',
    'dont start', 'what is your problem', 'whats your problem',
    'why are you like this', 'why are u like this',
    'stop texting me', 'dont text me',
    'this isnt funny', 'not funny', 'its not funny',
    'youre not funny', 'ur not funny',
    'stop acting like that', 'dont talk to me like that',
    'talk properly', 'be respectful', 'thats rude',
    'low blow', 'too far', 'you crossed the line',
    'this is too much',
    'why me', 'why always me',
    'leave me out of this', 'dont involve me',
    'stop targeting me', 'why are you targeting me',
    'im serious stop', 'seriously just stop',
    'stop messing with me', 'dont mess with me',
    'back off', 'just back off',
    'can you just stop', 'can you please stop',
    'what did i do', 'what did i even do',
    'how is that funny',

    # --- Gen-Z Slang (Filtered but Useful) ---
    'bro stop', 'bruh stop', 'nah stop', 'wtf stop',
    'wtf is wrong with you',
    'thats messed up', 'thats messed',
    'youre weird for that', 'ur weird for that',
    'this aint funny', 'aint funny',
    'youre doing too much', 'ur doing too much',
    'why you doing this',
    'stop bro seriously', 'im not playing', 'im not joking',
    'deadass stop', 'im fr stop', 'stop fr', 'actually stop',

    # --- Emotional & Vulnerable ---
    'why are you doing this to me', 'i didnt do anything',
    'what did i do to you', 'why are you like this with me',
    'this actually hurts', 'that hurt', 'youre hurting me',
    'stop hurting me', 'i dont like this',
    'this makes me uncomfortable', 'i dont feel okay',
    'im tired of this',

    # --- Hinglish (Strong + Natural) ---
    'band karo', 'chup karo', 'yaar band karo',
    'sach nahi hai', 'yeh sahi nahi', 'kyu bol rahe ho',
    'mujhe akela chhodo', 'please mat karo', 'aisa mat karo',
    'yeh galat hai', 'bahut hua', 'mat karo yaar',
    'bas karo', 'ab bas', 'kyu kar rahe ho',
    'mere pe kyu', 'kyu mere saath', 'mujhe kyu bol rahe ho',
    'itna bhi mat bolo', 'zyada ho raha hai',
    'limit mein raho', 'had ho gayi',
    'mazaak mat banao', 'mazaak mat udaao',
    'seriously mat karo',
    'bakwas band karo', 'faltu mat bolo',
    'samajh nahi aa raha kya', 'samajh nahi aata kya',
    'dekh bhai mat kar', 'fazool baat mat kar',
    'limit cross mat kar', 'dimaag mat khao',
    'kya masla hai', 'kya dikkat hai',
    'tamiz se baat kar', 'over mat ho',

    # --- Kannada & Mixed (Stronger Signals Only) ---
    'beda', 'sari illa',
    'yen maadtiya', 'yaake',
    'beda please', 'beda maga', 'beda guru',
    'saaku', 'saaku aaytu',
    'matad beda', 'hinge matad beda',
    'nange beda', 'nange ishta illa',
    'yake hing maadtiya', 'yake hinge',
    'nanna target maad beda',
    'please bidi', 'bittu bidi',
    'innu saaku', 'nidu jaasti aytu',
    'yako ashtu hogo', 'sumne irappa', 'vaasi bidi'
]
# Whole-word, apostrophe-insensitive matching ("don't" == "dont").
_DEFENSIVE_RE = [
    _term_regex(re.sub(r"['’]", "", s.lower()))
    for s in DEFENSIVE_SIGNALS
]


def has_defensive_language(text: str) -> bool:
    if not text:
        return False
    cleaned = re.sub(r"['’]", "", text.lower())
    return any(p.search(cleaned) for p in _DEFENSIVE_RE)

def get_reciprocity_score(conversation_history: list) -> tuple:
    """
    Updated to include victim defense detection.

    Three signals combined:
    1. Offensive reciprocity — are both sides using offensive language?
       If only one side is → bullying signal
    2. Defensive imbalance — is one person repeatedly defending themselves?
       If yes → strong bullying signal
    3. Laughter bonus — if both sides are laughing → banter signal

    Key insight: in bullying, the victim defends while the bully attacks.
    This asymmetry is detectable even when no offensive keywords are used.
    """
    if len(conversation_history) < 2:
        return 0.5, {}

    sender_scores = {}

    for msg in conversation_history:
        sender = msg.get('sender', 'unknown') if isinstance(msg, dict) \
                 else 'unknown'
        text   = (msg.get('text', '') if isinstance(msg, dict) \
                  else str(msg)).lower()

        if sender not in sender_scores:
            sender_scores[sender] = {
                'offensive' : 0,
                'defensive' : 0,
                'laughter'  : 0
            }

        if count_offensive_terms(text) > 0:
            sender_scores[sender]['offensive'] += 1
        if has_defensive_language(text):
            sender_scores[sender]['defensive'] += 1
        if _has_laughter(text):
            sender_scores[sender]['laughter'] += 1

    if not sender_scores:
        return 0.5, sender_scores

    senders = list(sender_scores.keys())
    if len(senders) < 2:
        return 0.0, sender_scores

    offensive_counts = [sender_scores[s]['offensive'] for s in senders]
    defensive_counts = [sender_scores[s]['defensive'] for s in senders]
    laughter_counts  = [sender_scores[s]['laughter']  for s in senders]

    # ── Signal 1: Offensive reciprocity ──────────────────────────────
    max_off = max(offensive_counts)
    min_off = min(offensive_counts)

    if max_off == 0:
        offensive_reciprocity = 0.5   # neutral — no offensive keywords used
    else:
        offensive_reciprocity = min_off / max_off

    # ── Signal 2: Defensive imbalance ────────────────────────────────
    # If one person has significantly more defensive messages than the other,
    # they are being targeted — this is a bullying signal
    max_def = max(defensive_counts)
    min_def = min(defensive_counts)
    total_def = sum(defensive_counts)

    if total_def == 0:
        defense_penalty = 0.0   # nobody defending — no signal either way
    elif max_def == 0:
        defense_penalty = 0.0
    else:
        # How imbalanced is the defense?
        # Perfect balance: both defend equally → penalty = 0
        # Complete imbalance: only one defends → penalty = 1.0
        defense_imbalance = 1.0 - (min_def / max_def)

        # Scale by how many defensive messages there are
        # 1 defensive message = small signal
        # 4+ defensive messages = strong signal
        defense_weight = min(1.0, total_def / 4)

        defense_penalty = defense_imbalance * defense_weight

    # ── Signal 3: Laughter bonus ──────────────────────────────────────
    both_laughing  = all(c > 0 for c in laughter_counts)
    laughter_bonus = 0.3 if both_laughing else 0.0

    # ── Combine ───────────────────────────────────────────────────────
    # Start from offensive reciprocity
    # Subtract defense penalty (imbalanced defense → lower reciprocity → more likely bullying)
    # Add laughter bonus (mutual laughter → more likely banter)
    final_score = offensive_reciprocity - defense_penalty + laughter_bonus
    final_score = round(max(0.0, min(1.0, final_score)), 4)

    return final_score, sender_scores

# ─────────────────────────────────────────────────────────────────────────
# SIGNAL 4 — Sentiment trajectory
# ─────────────────────────────────────────────────────────────────────────
def get_sentiment_trajectory(conversation_history: list) -> tuple:
    """
    Measures whether sentiment is oscillating (banter) or
    steadily declining (bullying).

    Returns:
        (trajectory_score, sentiment_scores_list)
        0.0 = declining, 1.0 = oscillating, 0.5 = not enough evidence

    Needs at least 3 messages with some sentiment movement; otherwise
    returns the neutral 0.5 (VADER cannot read most slang / Hinglish,
    so a flat reading means "no information", not "bullying").
    """
    if len(conversation_history) < 2:
        return 0.5, []

    scores = []
    for msg in conversation_history:
        text  = (msg.get('text', '') if isinstance(msg, dict) \
                 else str(msg))
        scores.append(vader.polarity_scores(text)['compound'])

    # Not enough points to judge a direction, or sentiment barely moves.
    if len(scores) < 3 or (max(scores) - min(scores)) < 0.10:
        return 0.5, scores

    # Successive changes, ignoring tiny wobbles.
    diffs = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
    diffs = [d for d in diffs if abs(d) >= 0.05]

    if len(diffs) >= 2:
        changes = sum(
            1 for i in range(1, len(diffs)) if diffs[i] * diffs[i - 1] < 0
        )
        oscillation_rate = changes / (len(diffs) - 1)
    else:
        oscillation_rate = 0.0   # one swing or a steady drop: no oscillation

    overall_trend = scores[-1] - scores[0]
    trend_score   = max(0.0, min(1.0, (overall_trend + 1) / 2))
    trajectory_score = round(
        (oscillation_rate * 0.6) + (trend_score * 0.4), 4
    )

    return trajectory_score, scores
# ─────────────────────────────────────────────────────────────────────────
# LGS-MT — MEMORY SCORES
# ─────────────────────────────────────────────────────────────────────────
MEMORY_MIN_SIMILARITY = 0.90     # only genuinely close neighbours count
MEMORY_MIN_HARMFUL_SHARE = 0.60  # harmful share among those close neighbours


def global_memory_evidence(memories, banter_signal: float = 0.0):
    """
    Returns the memory score (max similarity of harmful close neighbours),
    or None when memory gives no reliable evidence. Memory never reinforces
    a flag in a playful context, and benign close neighbours count against it.
    """
    if banter_signal >= 0.5:
        return None

    close = []
    for memory in memories or []:
        similarity = max(0.0, min(1.0, memory.get("similarity", 0.0)))
        if similarity < MEMORY_MIN_SIMILARITY:
            continue
        try:
            label = int(memory.get("label", 0))
        except (TypeError, ValueError):
            label = 0
        close.append((similarity, label))

    if not close:
        return None

    harmful = [s for s, label in close if label == 1]
    if not harmful or len(harmful) / len(close) < MEMORY_MIN_HARMFUL_SHARE:
        return None

    return max(harmful)
OWN_TEXT_FLOOR_MIN = 0.85       # message alone must look clearly harmful
OWN_BANTER_MAX = 0.30           # ...and carry no playful marker of its own
VICTIM_OWN_TEXT_MAX = 0.50      # message alone must look clearly benign


def own_text_adjustment(
    *,
    contextual_local: float,
    contextual_global,
    isolated_local: float,
    isolated_global,
    offensive_signal: float,
    own_banter: float,
    defensive_language: bool,
    threat_language: bool,
    sender_offensive_count: int,
):
    """
    Both models score the whole thread, so abusive history can make a
    victim's reply look harmful and friendly history can make a threat
    look harmless. Cross-check against the message's own text:

      victim cap  (scores can only go DOWN): defensive language, no
        offensive term / threat, the sender was not abusive in the window,
        and the message alone scores < VICTIM_OWN_TEXT_MAX.
      own-text floor (scores can only go UP): the message alone scores
        >= OWN_TEXT_FLOOR_MIN and has no playful marker of its own.

    Returns (local_score, global_score, reason_or_None).
    """
    if isolated_global is None:
        isolated = isolated_local
    else:
        isolated = 0.4 * isolated_local + 0.6 * isolated_global

    if (
        defensive_language
        and offensive_signal == 0.0
        and not threat_language
        and sender_offensive_count == 0
        and isolated < VICTIM_OWN_TEXT_MAX
    ):
        capped_global = contextual_global
        if contextual_global is not None and isolated_global is not None:
            capped_global = min(contextual_global, isolated_global)
        return (
            min(contextual_local, isolated_local),
            capped_global,
            "victim_own_text",
        )

    if isolated >= OWN_TEXT_FLOOR_MIN and own_banter < OWN_BANTER_MAX:
        floored_global = contextual_global
        if contextual_global is not None and isolated_global is not None:
            floored_global = max(contextual_global, isolated_global)
        floored_local = max(contextual_local, isolated_local)
        if (
            floored_local != contextual_local
            or floored_global != contextual_global
        ):
            return floored_local, floored_global, "own_text_floor"

    return contextual_local, contextual_global, None
# ─────────────────────────────────────────────────────────────────────────
# SCORE FUSION
# ─────────────────────────────────────────────────────────────────────────
def fuse_scores(
    muril_score: float,
    local_model_score: float,
    banter_signal: float,
    reciprocity_score: float,
    trajectory_score: float,
    local_memory_score=None,
    global_memory_score=None,
    history_length: int = 0,
    threshold: float = THRESHOLD,
    offensive_signal: float = 0.0,
    override_block_reason: str = None
) -> dict:

    # ------------------------------------------------------------
    # 1. Determine base model score
    # ------------------------------------------------------------

    if muril_score is None:
        base_score = local_model_score
    else:
        base_score = (
            0.4 * local_model_score +
            0.6 * muril_score
        )

    # ------------------------------------------------------------
    # 2. Memory evidence
    # ------------------------------------------------------------

    local_mem = (
        local_memory_score
        if local_memory_score is not None
        else 0.0
    )

    global_mem = (
        global_memory_score
        if global_memory_score is not None
        else 0.0
    )

    memory_bonus = (
        0.10 * local_mem +
        0.05 * global_mem
    )

    score_with_memory = min(
        1.0,
        base_score + memory_bonus
    )
    # ------------------------------------------------------------
    # EXPLICIT OFFENSIVE LANGUAGE BOOST
    # ------------------------------------------------------------

    offensive_boost = 0.20 * offensive_signal

    score_with_offensive = min(
        1.0,
        score_with_memory + offensive_boost
    )
    # ------------------------------------------------------------
    # 3. History-dependent contextual weights
    # ------------------------------------------------------------

    if history_length < 2:

        W_RECIP = 0.10
        W_BANTER = 0.25
        W_TRAJ = 0.15
        max_reduction = 0.20

    elif history_length < 4:

        W_RECIP = 0.15
        W_BANTER = 0.20
        W_TRAJ = 0.15
        max_reduction = 0.30

    else:

        W_RECIP = 0.20
        W_BANTER = 0.15
        W_TRAJ = 0.10
        max_reduction = 0.35

    # ------------------------------------------------------------
    # 4. Contextual reduction
    # ------------------------------------------------------------
    weighted_reduction = min(
        max_reduction,
        (
            reciprocity_score * W_RECIP
            + banter_signal * W_BANTER
            + trajectory_score * W_TRAJ
        )
    )

    # Do not allow banter/context reduction to hide
    # explicit offensive language.
    if offensive_signal >= 0.75:
        weighted_reduction = min(
            weighted_reduction,
            0.15
        )

    # ------------------------------------------------------------
    # 5. Protect strong model evidence
    # ------------------------------------------------------------

    if base_score >= 0.90:

        weighted_reduction = min(
            weighted_reduction,
            0.15
        )

    elif base_score >= 0.80:

        weighted_reduction = min(
            weighted_reduction,
            0.20
        )
        # ------------------------------------------------------------
    # 5b. Evidence-gated banter override
    #
    # Strong, mutual, explicit banter may discount even a very high
    # model score (the models were observed scoring friendly teasing
    # at 0.96-0.99). Every gate must hold; any threat language, strong
    # abuse, victim pushback, or one-sided pattern disables it.
    # ------------------------------------------------------------

    banter_override = (
        history_length >= 1
        and banter_signal >= 0.55
        and reciprocity_score >= 0.80
        and offensive_signal < 0.75
        and override_block_reason is None
    )

    if banter_override:
        weighted_reduction = max(
            weighted_reduction,
            min(0.35, 0.20 + 0.25 * banter_signal)
        )
    # ------------------------------------------------------------
    # 6. Apply reduction
    # ------------------------------------------------------------

    adjusted_score = (
        score_with_offensive *
        (1 - weighted_reduction)
    )

    adjusted_score = max(
        0.0,
        min(1.0, adjusted_score)
    )

    # ------------------------------------------------------------
    # 7. Final decision
    # ------------------------------------------------------------

    flagged = adjusted_score >= threshold

    # ------------------------------------------------------------
    # 8. RETURN RESULT
    #
    # IMPORTANT:
    # This must be aligned with the rest of the function.
    # ------------------------------------------------------------

    return {
        "muril_score": (
            round(muril_score, 4)
            if muril_score is not None
            else None
        ),

        "local_model_score":
            round(local_model_score, 4),

        "local_memory_score": (
            round(local_memory_score, 4)
            if local_memory_score is not None
            else None
        ),

        "global_memory_score": (
            round(global_memory_score, 4)
            if global_memory_score is not None
            else None
        ),

        "memory_bonus":
            round(memory_bonus, 4),

        "banter_signal":
            round(banter_signal, 4),

        "reciprocity_score":
            round(reciprocity_score, 4),

        "trajectory_score":
            round(trajectory_score, 4),

        "base_score":
            round(base_score, 4),

        "reduction_applied":
            round(weighted_reduction, 4),

        "adjusted_score":
            round(adjusted_score, 4),

        "offensive_signal":
            round(offensive_signal, 4),

                "offensive_boost":
            round(offensive_boost, 4),

        "banter_override":
            banter_override,

        "override_block_reason":
            override_block_reason,

        "flagged":
            flagged,

        "label":
            1 if flagged else 0,
    }
# ─────────────────────────────────────────────────────────────────────────
# LOCAL MODEL
# ─────────────────────────────────────────────────────────────────────────


def get_local_model_score(
    text: str,
    conversation_history: list = None
) -> float:
    """
    Returns probability that the message is harmful
    according to the local student model.
    """

    if conversation_history:

        recent = [
            m.get("text", "")
            if isinstance(m, dict)
            else str(m)
            for m in conversation_history[-4:]
        ]

        input_text = (
            f" {local_tokenizer.sep_token} "
            .join(recent + [text])
        )

    else:

        input_text = text

    encoding = local_tokenizer(
        input_text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)

    attention_mask = encoding[
        "attention_mask"
    ].to(DEVICE)

    with torch.no_grad():

        output = local_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        scaled_logits = (
            output.logits / TEMPERATURE
        )

        probs = softmax(
            scaled_logits.cpu().numpy()[0]
        )

    return float(probs[1])

# ─────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────
# EMBEDDING HELPER (for ChromaDB storage)
# ─────────────────────────────────────────────────────────────────────────
def get_message_embedding(text: str) -> list:
    """
    Returns 768-dim MuRIL [CLS] embedding for a message.
    Used by database.py to store messages in ChromaDB.
    """
    encoding = muril_tokenizer(
        text,
        max_length     = MAX_LEN,
        padding        = 'max_length',
        truncation     = True,
        return_tensors = 'pt'
    )
    input_ids      = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        base_model = getattr(muril_model, 'bert',
                     getattr(muril_model, 'albert',
                     getattr(muril_model, 'roberta', None)))

        if base_model is not None:
            outputs   = base_model(input_ids=input_ids,
                                   attention_mask=attention_mask)
            embedding = outputs.last_hidden_state[:, 0, :] \
                               .cpu().numpy()[0].tolist()
        else:
            outputs   = muril_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            embedding = outputs.hidden_states[-1][:, 0, :] \
                               .cpu().numpy()[0].tolist()

    return embedding