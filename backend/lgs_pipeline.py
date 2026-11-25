"""
lgs_pipeline.py

Local-Global Split-Memory Transformer (LGS-MT)

Architecture:

                    Current Message
                          |
              +-----------+-----------+
              |                       |
              v                       v
        LOCAL MEMORY            GLOBAL MEMORY
        Conversation            Generalized
        specific                 patterns
              |                       |
              v                       v
        LOCAL MODEL              GLOBAL MODEL
        4-layer MuRIL            12-layer MuRIL
              |                       |
              +-----------+-----------+
                          |
                          v
                    Score Fusion
                          |
                          v
                    Final Decision
"""

"""
lgs_pipeline.py — LGS-MT confidence-based router
"""
from .pipeline import (
    get_local_model_score,
    get_muril_score,
    get_banter_signal,
    get_reciprocity_score,
    get_sentiment_trajectory,
    get_message_embedding,
    get_offensive_signal,
    has_threat_language,
    has_defensive_language,
    own_text_adjustment,
    global_memory_evidence,
    fuse_scores,
    THRESHOLD,
)

from .memory import search_local_memory, search_global_memory

CONFIDENCE_THRESHOLD = 0.95   # tune this empirically once you have eval data


def local_confidence(local_score: float) -> float:
    """0.0 = totally uncertain (score near 0.5), 1.0 = fully confident."""
    return abs(local_score - 0.5) * 2


def lgs_pipeline(
    conversation_id,
    conversation_history,
    current_message,
    threshold=THRESHOLD
):
    text = current_message["text"]
    full_history = conversation_history + [current_message]
    # ------------------------------------------------------------
    # EXPLICIT OFFENSIVE LANGUAGE
    # ------------------------------------------------------------

    offensive_signal = get_offensive_signal(text)
    # ------------------------------------------------------------
    # 1. ALWAYS-ON LOCAL SIGNALS
    # ------------------------------------------------------------
    local_score = get_local_model_score(
        text,
        conversation_history
    )

    banter = get_banter_signal(
        text,
        conversation_history
    )

    recip, sender_counts = get_reciprocity_score(
        full_history
    )

    traj, sentiments = get_sentiment_trajectory(
        full_history
    )
        # ------------------------------------------------------------
    # BANTER-OVERRIDE GATES (see fuse_scores 5b)
    # ------------------------------------------------------------
    recent_texts = [text] + [
        m.get("text", "") if isinstance(m, dict) else str(m)
        for m in conversation_history[-4:]
    ]

    history_offensive_signal = max(
        (get_offensive_signal(t) for t in recent_texts[1:]),
        default=0.0,
    )

    override_block_reason = None
    if any(has_threat_language(t) for t in recent_texts):
        override_block_reason = "threat_language"
    elif any(
        isinstance(c, dict) and c.get("defensive", 0) > 0
        for c in sender_counts.values()
    ):
        override_block_reason = "defensive_message"
    elif history_offensive_signal >= 0.75:
        override_block_reason = "strong_offensive_history"

        # ------------------------------------------------------------
    # OWN-TEXT CHECK
    # The models score the whole thread. Also score the message alone.
    # ------------------------------------------------------------
    sender = current_message.get("sender")
    sender_stats = (
        sender_counts.get(sender, {})
        if isinstance(sender_counts, dict) else {}
    )
    sender_offensive_count = (
        sender_stats.get("offensive", 0)
        if isinstance(sender_stats, dict) else 0
    )

    own_banter = get_banter_signal(text, [])
    defensive_language = has_defensive_language(text)
    threat_language = has_threat_language(text)

    isolated_local = (
        get_local_model_score(text, [])
        if conversation_history
        else local_score
    )
    context_disagreement = abs(isolated_local - local_score) >= 0.4
    # ------------------------------------------------------------
    # 2. LOCAL MEMORY
    # ------------------------------------------------------------
    embedding = get_message_embedding(text)

    local_memories = (
        search_local_memory(
            conversation_id,
            embedding,
            n_results=5
        )
        if conversation_id
        else []
    )
    
# Local memory is retrieved for context, but don't let a
# previous automated flag alone reinforce another flag.
# This prevents false positives from creating a feedback loop.
    local_memory_score = None
# ------------------------------------------------------------
    # 3. LOCAL MODEL CONFIDENCE
    # ------------------------------------------------------------
    confidence = local_confidence(local_score)

    # ------------------------------------------------------------
    # 4. CONTEXT-AWARE ROUTING
    # ------------------------------------------------------------
    contextual_risk = (
        recip < 0.4
        or traj < 0.25
        or banter >= 0.5
    )

    use_global = (
        confidence < CONFIDENCE_THRESHOLD
        or contextual_risk
        or context_disagreement
    )

    # ------------------------------------------------------------
    # 5. DEFAULT VALUES
    # ------------------------------------------------------------
    global_score = None
    global_memory_score = None
    global_memories = []
    isolated_global = None
    own_text_reason = None
    # ------------------------------------------------------------
    # 6. LOCAL-ONLY PATH
    # ------------------------------------------------------------
    if not use_global:

        result = fuse_scores(
            local_model_score=local_score,
            muril_score=local_score,
            banter_signal=banter,
            reciprocity_score=recip,
            trajectory_score=traj,
            local_memory_score=local_memory_score,
            global_memory_score=None,
            offensive_signal=offensive_signal,
            history_length=len(conversation_history),
            threshold=threshold,
            override_block_reason=override_block_reason,
        )

        source = "local_only"

        router_reason = "high_local_confidence"

    # ------------------------------------------------------------
    # 7. GLOBAL FALLBACK PATH
    # ------------------------------------------------------------
    else:

        global_score = get_muril_score(
            text,
            [
                m.get("text", "")
                if isinstance(m, dict)
                else str(m)
                for m in conversation_history
            ]
        )

                # --------------------------------------------------------
        # OWN-TEXT CHECK: cross-check thread-level scores against
        # the message's own text (see own_text_adjustment).
        # --------------------------------------------------------
        victim_candidate = (
            defensive_language
            and offensive_signal == 0.0
            and not threat_language
        )
        floor_candidate = isolated_local >= 0.7 and own_banter < 0.30

        if conversation_history and (victim_candidate or floor_candidate):
            isolated_global = get_muril_score(text, [])

        if conversation_history:
            local_for_fuse, global_for_fuse, own_text_reason = (
                own_text_adjustment(
                    contextual_local=local_score,
                    contextual_global=global_score,
                    isolated_local=isolated_local,
                    isolated_global=isolated_global,
                    offensive_signal=offensive_signal,
                    own_banter=own_banter,
                    defensive_language=defensive_language,
                    threat_language=threat_language,
                    sender_offensive_count=sender_offensive_count,
                )
            )
        else:
            local_for_fuse, global_for_fuse = local_score, global_score

        global_memories = search_global_memory(
            embedding,
            n_results=5
        )

        # --------------------------------------------------------
        # GLOBAL MEMORY SCORE
        # --------------------------------------------------------
        global_memory_score = global_memory_evidence(
        global_memories, banter
        )
        # --------------------------------------------------------
        # FUSE GLOBAL SCORE + OTHER SIGNALS
        # --------------------------------------------------------
        print(
            "[LGS DEBUG]",
            {
                "local_score": local_score,
                "global_score": global_score,
                "local_memory_score": local_memory_score,
                "global_memory_score": global_memory_score,
                "offensive_signal": offensive_signal,
                "history_length": len(conversation_history),
            },
        )
        result = fuse_scores(
            muril_score=global_for_fuse,
            local_model_score=local_for_fuse,
            banter_signal=banter,
            reciprocity_score=recip,
            trajectory_score=traj,
            local_memory_score=local_memory_score,
            global_memory_score=global_memory_score,
            history_length=len(conversation_history),
            threshold=threshold,
            offensive_signal=offensive_signal,
            override_block_reason=override_block_reason,
        )

        source = "global_fallback"

        if confidence < CONFIDENCE_THRESHOLD:
            router_reason = "low_local_confidence"
        elif contextual_risk:
            router_reason = "contextual_risk"
        else:
            router_reason = "context_disagreement"

    # ------------------------------------------------------------
    # 8. COMMON RESULT FIELDS
    #
    # IMPORTANT:
    # This is OUTSIDE the if/else.
    # Therefore BOTH paths receive these fields.
    # ------------------------------------------------------------
    result.update({

        "confidence": round(confidence, 4),

        "source": source,

        "local_score": round(
            local_score,
            4
        ),

        "global_score": (
            round(global_score, 4)
            if global_score is not None
            else None
        ),

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

        "final_score": result["adjusted_score"],

        "threshold": threshold,

        "local_memories": local_memories,

        "global_memories": global_memories,

        "sender_counts": sender_counts,

        "history_length": len(conversation_history),

        "contextual_risk": contextual_risk,

        "router_reason": router_reason,

        "isolated_local": round(isolated_local, 4),

        "isolated_global": (
            round(isolated_global, 4)
            if isolated_global is not None
            else None
        ),

        "own_text_reason": own_text_reason,

        "embedding": embedding,

    })

    return result