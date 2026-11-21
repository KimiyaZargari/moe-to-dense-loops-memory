import json, sys

def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["doc_id"]] = r
    return rows

old, new = load(sys.argv[1]), load(sys.argv[2])
print("keys in a sample:", sorted(next(iter(old.values())).keys()))
print("n old:", len(old), " n new:", len(new))

METRIC = "exact_match"  # change if the key above is named differently

common = sorted(set(old) & set(new))            # should be doc_id 0..499
same_prompt = sum(old[i]["prompt_hash"] == new[i]["prompt_hash"] for i in common)
same_score = sum(old[i][METRIC] == new[i][METRIC] for i in common)
print(f"overlap: {len(common)}")
print(f"identical prompts: {same_prompt}/{len(common)}")
print(f"identical scores:  {same_score}/{len(common)}")

def acc(rows, ids):
    return sum(rows[i][METRIC] for i in ids) / len(ids)

print("old run, acc on first 500:     ", round(acc(old, common), 4))
print("new run, acc on same first 500:", round(acc(new, common), 4))
rest = [i for i in new if i not in old]
print("new run, acc on questions 501+:", round(acc(new, rest), 4), f"(n={len(rest)})")