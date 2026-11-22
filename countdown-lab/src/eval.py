import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from .generator import read_jsonl, assert_dataset_valid
from .verifier import verify

def pass_k(n,c,k):
    if not 0 <= c <= n or not 1 <= k <= n: raise ValueError("Invalid n,c,k")
    return 1.0 if n-c < k else 1.0-math.comb(n-c,k)/math.comb(n,k)

def bootstrap(values,seed=42,repeats=2000):
    rng=random.Random(seed)
    estimates=sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(repeats))
    return [estimates[int(.025*repeats)],estimates[min(repeats-1,int(.975*repeats))]]

def evaluate(records,dataset,expected_n,k=1,selection_budget=1):
    if type(expected_n) is not int or expected_n<1 or selection_budget<1:
        raise ValueError("Expected counts must be positive")
    assert_dataset_valid(dataset)
    expected={r["id"]:r for r in dataset}
    if not records or not expected: raise ValueError("Empty data/log")
    if len({(r["method"],r["config_id"],r["split"]) for r in records}) != 1:
        raise ValueError("Mixed method/config/split")
    groups=defaultdict(list);seen=set()
    for r in records:
        ident=r["problem_id"];index=r["sample_index"]
        if ident not in expected or type(index) is not int or index<0:
            raise ValueError("Unknown ID or bad index")
        if (ident,index) in seen: raise ValueError("Duplicate sample")
        if r["split"] != expected[ident]["split"]: raise ValueError("Split mismatch")
        seen.add((ident,index));groups[ident].append(r)
        truth=verify(r["output"],expected[ident]["numbers"],expected[ident]["target"])
        if truth != r["verifier"]: raise ValueError("Stale or inconsistent verifier result")
        if r["tokens"] < 0 or r["seconds"] < 0: raise ValueError("Negative cost")
    if set(groups) != set(expected): raise ValueError("Missing problem IDs")
    sizes={len(g) for g in groups.values()}
    if len(sizes)!=1: raise ValueError("Unequal sample counts")
    n=sizes.pop()
    if n!=expected_n: raise ValueError("Sample count differs from locked inference plan")
    if not 1<=k<=n: raise ValueError("Need n >= k")
    if any({r["sample_index"] for r in group} != set(range(n)) for group in groups.values()):
        raise ValueError("Missing/noncontiguous sample indices")
    budgets={r.get("selection_budget",1) for r in records}
    if budgets!={selection_budget}: raise ValueError("Selection budget differs from locked plan")
    selected=selection_budget>1
    if selected and (n!=1 or records[0]["method"] not in {"B2","B3"}):
        raise ValueError("Selected log must have one record per problem")
    p1=[];pk=[]
    for group in groups.values():
        c=sum(r["verifier"]["valid"] and r["verifier"]["target_ok"] for r in group)
        p1.append(c/n);pk.append(pass_k(n,c,k))
    mean=lambda x:sum(x)/len(x)
    errors=Counter(r["verifier"]["error_code"] or "OK" for r in records)
    metric=dict(problem_count=len(groups),samples_per_problem=n,k=k,
        valid_output_rate=mean([int(r["verifier"]["valid"]) for r in records]),
        parse_failure_rate=mean([int(not r["verifier"]["parse_ok"]) for r in records]),
        error_counts=dict(errors),mean_tokens_per_record=mean([r["tokens"] for r in records]),
        mean_seconds_per_record=mean([r["seconds"] for r in records]),
        mean_tokens_per_problem=sum(r["tokens"] for r in records)/len(groups),
        mean_seconds_per_problem=sum(r["seconds"] for r in records)/len(groups),
        bootstrap_unit="problem ID; uncertainty across problems, not training seeds")
    if selected:
        metric.update(selected_success_rate=mean(p1),selected_success_ci95=bootstrap(p1),selection_budget=next(iter(budgets)))
    else:
        metric.update(pass_at_1=mean(p1),pass_at_k=mean(pk),pass_at_1_ci95=bootstrap(p1),pass_at_k_ci95=bootstrap(pk))
    return metric

def main():
    p=argparse.ArgumentParser();p.add_argument("--log",required=True);p.add_argument("--dataset",required=True)
    p.add_argument("--k",type=int,default=1);p.add_argument("--out",required=True)
    p.add_argument("--expected-n",type=int,required=True,help="Locked raw sample count, or 1 for selected logs")
    p.add_argument("--selection-budget",type=int,default=1,help="Original N for selected B2/B3 logs")
    a=p.parse_args();out=Path(a.out)
    if out.exists(): raise FileExistsError(out)
    metrics=evaluate(read_jsonl(a.log),read_jsonl(a.dataset),a.expected_n,a.k,a.selection_budget)
    out.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(metrics,indent=2);out.write_text(text,encoding="utf-8");print(text)
if __name__ == "__main__": main()
