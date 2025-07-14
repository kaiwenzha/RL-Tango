import json
from pathlib import Path

in_path  = Path("./data/StrategyQA/strategyqa_train.json")
out_path = Path("./data/StrategyQA/strategyqa_train.jsonl")

def convert_facts_to_string(facts):
    assert len(facts) > 0
    string = "You are given the following facts:\n"
    for i, fact in enumerate(facts):
        string += f"{i+1}. {fact}\n"
    return string

with in_path.open("r", encoding="utf-8") as f:
    records = json.load(f)

with out_path.open("w", encoding="utf-8") as f:
    for ex in records:
        json.dump(
            {"question": ex["question"], "answer": ex["answer"], "facts": convert_facts_to_string(ex["facts"])},
            f,
            ensure_ascii=False
        )
        f.write("\n")