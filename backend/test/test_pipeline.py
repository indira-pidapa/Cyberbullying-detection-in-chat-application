# test_pipeline.py — run from backend/ folder
# pip install requests first if needed
import requests

API = "http://localhost:8000"

def simulate(conv_id, messages):
    requests.delete(f"{API}/db/{conv_id}")
    print(f"\n{'='*60}")
    print(f"CONVERSATION: {conv_id}")
    print('='*60)
    for msg in messages:
        r = requests.post(f"{API}/classify", json={
            "conversation_id": conv_id,
            "sender_id": msg["sender"],
            "message": msg["text"]
        }).json()
        status = "🚨 FLAGGED" if r["flagged"] else "✅ safe   "
        recip  = r["signals"]["reciprocity_score"]
        muril  = r["signals"]["muril_score"]
        red    = r["signals"]["reduction_applied"]
        print(f"{status} [{msg['sender']}] {msg['text'][:50]}")
        print(f"         muril={muril:.2f} recip={recip:.2f} -red={red:.2f} adj={r['adjusted_score']:.2f}")

simulate("test_cag", [
    {"sender":"user_A","text":"yo"},
    {"sender":"user_B","text":"hey"},
    {"sender":"user_A","text":"u saw the assignment?"},
    {"sender":"user_B","text":"yea just started it"},
    {"sender":"user_A","text":"u literally struggle with everything"},
    {"sender":"user_B","text":"thats not true"},
    {"sender":"user_A","text":"like even in class u look lost"},
    {"sender":"user_B","text":"why are you saying this"},
    {"sender":"user_A","text":"im just being honest"},
    {"sender":"user_B","text":"that doesnt make it okay"},
    {"sender":"user_A","text":"u get weird about small things"},
    {"sender":"user_B","text":"just stop"},
    {"sender":"user_A","text":"okay sorry if u feel bad"},
])

simulate("test_banter", [
    {"sender":"user_A","text":"bro ur such a nerd lol 😂"},
    {"sender":"user_B","text":"shut up you idiot 😂"},
    {"sender":"user_A","text":"haha you wish loser 💀"},
    {"sender":"user_B","text":"you're so dumb honestly"},
    {"sender":"user_A","text":"ok ok you win this time 😂"},
])