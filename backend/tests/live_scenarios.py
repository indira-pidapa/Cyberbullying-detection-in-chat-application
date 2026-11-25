"""
Live API scenarios. Requires the backend running on localhost:8000.
Run from the project root:   python -m backend.tests.live_scenarios

- Every scenario gets a brand-new conversation_id (no history contamination).
- Senders are fixed, distinct IDs (never the Swagger placeholder "string").
"""
import json
import sys
import urllib.request
import uuid

API = "http://localhost:8000"
ALICE, BOB = "alice_test", "bob_test"

SCENARIOS = [
    {
        "name": "friendly_banter_user_example",
        "expect": "none_flagged",
        "messages": [
            (ALICE, "hiiiii....loserrrrr....nvm...i was jk..u k that right?"),
            (BOB,   "supppp loser....ik dw"),
        ],
    },
    {
        "name": "friendly_banter_variant",
        "expect": "none_flagged",
        "messages": [
            (ALICE, "lmaooo u lost again 😂"),
            (BOB,   "shhh ur just salty haha"),
            (ALICE, "ok ok you win this one jk"),
        ],
    },
    {
        "name": "one_sided_repeated_harassment",
        "expect": "flag_by_end+sender_never_flagged:" + BOB,
        "messages": [
            (ALICE, "you are such a loser"),
            (BOB,   "please stop"),
            (ALICE, "nobody likes you, everyone laughs at you"),
            (BOB,   "why are you doing this to me"),
            (ALICE, "you are pathetic and worthless, just disappear"),
        ],
    },
    {
        "name": "explicit_threat",
        "expect": "flag_by_end",
        "messages": [
            (ALICE, "i know where you live and i will hurt you after school"),
        ],
    },
    {
        "name": "defensive_victim",
        "expect": "sender_never_flagged:" + BOB,
        "messages": [
            (ALICE, "you are such a freak, nobody wants you here"),
            (BOB,   "please stop, that really hurts"),
            (BOB,   "i didn't do anything to you, leave me alone"),
        ],
    },
    {
        "name": "ambiguous_insult_informational",
        "expect": "informational",
        "messages": [
            (ALICE, "you are so dumb sometimes"),
        ],
    },
    {
        "name": "ambiguous_sarcasm_informational",
        "expect": "informational",
        "messages": [
            (ALICE, "wow nice one, genius"),
        ],
    },
           {"name": "diag_isolated_banter_reply", "expect": "informational",
        "messages": [(BOB, "supppp loser....ik dw")]},
       {"name": "diag_isolated_victim_1", "expect": "informational",
        "messages": [(BOB, "please stop, that really hurts")]},
       {"name": "diag_isolated_victim_2", "expect": "informational",
        "messages": [(BOB, "i didn't do anything to you, leave me alone")]},
        {
        "name": "joking_threat_with_lol_informational",
        "expect": "informational",
        "messages": [
            (ALICE, "lmaooo ok that was funny 😂 jk"),
            (BOB,   "haha stoppp 😂"),
            (ALICE, "lol i will hurt you after school jk"),
        ],
    },
    {
        "name": "laundered_insult_with_jk",
        "expect": "flag_by_end",
        "messages": [
            (ALICE, "lol u are so stupid and worthless jk"),
        ],
    },
        {
        "name": "threat_after_friendly_chat",
        "expect": "flag_by_end",
        "messages": [
            (ALICE, "lmaooo ok that was funny 😂"),
            (BOB,   "haha stoppp 😂"),
            (ALICE, "i know where you live and i will hurt you after school"),
        ],
    },
]


def post_classify(conv_id, sender, text):
    body = json.dumps(
        {"conversation_id": conv_id, "sender_id": sender, "message": text}
    ).encode()
    req = urllib.request.Request(
        f"{API}/classify", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def f(v):
    if v is None:
        return "  -  "
    return f"{v:5.2f}" if isinstance(v, (int, float)) else str(v)

def verdict(expect, flags):
    results = []
    for part in expect.split("+"):
        if part == "none_flagged":
            results.append(not any(fl for _, fl in flags))
        elif part == "flag_by_end":
            results.append(flags[-1][1])
        elif part.startswith("sender_never_flagged:"):
            who = part.split(":", 1)[1]
            results.append(not any(fl for s, fl in flags if s == who))
        else:
            return None  # informational
    return all(results)



def run():
    failures = 0
    passed = 0
    info = 0

    print("\n" + "=" * 65)
    print("          LGS-MT LIVE SCENARIO EVALUATION")
    print("=" * 65)

    for sc in SCENARIOS:
        conv_id = f"test-{sc['name']}-{uuid.uuid4().hex[:8]}"

        print(f"\nSCENARIO: {sc['name']}")
        print("-" * 65)

        flags = []

        for idx, (sender, text) in enumerate(sc["messages"], 1):
            r = post_classify(conv_id, sender, text)
            s = r["signals"]

            flags.append((sender, r["flagged"]))

            status = "FLAGGED" if r["flagged"] else "NOT FLAGGED"

            print(f"\n  Message {idx} | {sender}")
            print(f"  Prediction : {status}")

            print(
                f"  Scores     : Final={f(r.get('adjusted_score'))} "
                f"| Base={f(s.get('base_score'))}"
            )

            print(
                f"  Models     : Local={f(s.get('local_score'))} "
                f"| Global={f(s.get('global_score'))}"
            )

            print(
                f"  Signals    : Offensive={f(s.get('offensive_signal'))} "
                f"| Banter={f(s.get('banter_signal'))}"
            )

            print(
                f"  Context    : Reciprocity={f(s.get('reciprocity_score'))} "
                f"| Trajectory={f(s.get('trajectory_score'))}"
            )

            print(
                f"  Routing    : {s.get('router_reason')} "
                f"| Override={s.get('banter_override')}"
            )

            print(
                f"  Adjustment : Reduction={f(s.get('reduction_applied'))} "
                f"| Memory={f(s.get('memory_bonus'))}"
            )

            if s.get("override_block_reason"):
                print(f"  Block reason: {s['override_block_reason']}")

            if s.get("own_text_reason"):
                print(f"  Own-text   : {s['own_text_reason']}")

            print(f"  Message    : {text}")

        # Evaluate scenario expectations
        ok = verdict(sc["expect"], flags)

        if ok is None:
            label = "INFO"
            info += 1

        elif ok:
            label = "PASS"
            passed += 1

        else:
            label = "FAIL"
            failures += 1

        print("\n" + "-" * 65)
        print(f"  SCENARIO RESULT: {label}")
        print(f"  Expected: {sc['expect']}")
        print("-" * 65)

    # Final evaluation summary
    total = len(SCENARIOS)

    print("\n" + "=" * 65)
    print("              FINAL EVALUATION SUMMARY")
    print("=" * 65)

    print(f"  Total scenarios : {total}")
    print(f"  Passed          : {passed}")
    print(f"  Failed          : {failures}")
    print(f"  Informational   : {info}")

    print("\n  Note: INFO scenarios are not pass assertions.")
    print("=" * 65)

    # Consistency check
    assert passed + failures + info == total

    sys.exit(1 if failures else 0)



if __name__ == "__main__":
    run()