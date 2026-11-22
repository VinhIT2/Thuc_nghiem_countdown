import argparse
import ast
import hashlib
import json
import random
from fractions import Fraction
from pathlib import Path
from .verifier import verify, VERSION as VERIFIER_VERSION

VERSION = "generator-v1-reference-profiles"
PROFILES = ["id", "ood-range", "ood-arity", "ood-fraction", "ood-depth"]

def canonical_key(numbers, target):
    return json.dumps([sorted(numbers), target], separators=(",", ":"))

def key_hash(numbers, target):
    return hashlib.sha256(canonical_key(numbers, target).encode()).hexdigest()

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def assert_dataset_valid(rows):
    keys, ids = set(), set()
    for row in rows:
        key = canonical_key(row["numbers"], row["target"])
        if key in keys or row["id"] in ids:
            raise ValueError("Duplicate problem/id: " + row["id"])
        keys.add(key); ids.add(row["id"])
        v = verify("FINAL: " + row["reference_expression"], row["numbers"], row["target"])
        if not (v["valid"] and v["target_ok"]):
            raise ValueError((row["id"], v))
    return keys

def features(expression):
    def walk(n):
        if isinstance(n, ast.Constant): return Fraction(n.value), 0, False
        if isinstance(n, ast.UnaryOp):
            v, d, f = walk(n.operand); return -v, d + 1, f
        a, da, fa = walk(n.left); b, db, fb = walk(n.right)
        if isinstance(n.op, ast.Add): v = a+b
        elif isinstance(n.op, ast.Sub): v = a-b
        elif isinstance(n.op, ast.Mult): v = a*b
        else: v = a/b
        return v, 1+max(da, db), fa or fb or v.denominator != 1
    return walk(ast.parse(expression, mode="eval").body)

def combine(a, b, op):
    va, ea = a; vb, eb = b
    if op == "+": v = va+vb
    elif op == "-": v = va-vb
    elif op == "*": v = va*vb
    else:
        if vb == 0: return None
        v = va/vb
    return v, f"({ea} {op} {eb})"

def random_expression(numbers, rng, deep=False):
    leaves = [(Fraction(n), str(n)) for n in numbers]
    rng.shuffle(leaves)
    def tree(items):
        if len(items) == 1: return items[0]
        mid = len(items)-1 if deep else len(items)//2
        a, b = tree(items[:mid]), tree(items[mid:])
        if a is None or b is None: return None
        return combine(a, b, rng.choice(["+", "-", "*", "/"]))
    return tree(leaves)

def solve(numbers, target, accept=None):
    # Vét cạn cặp; không memo theo giá trị vì cần lọc hình dạng reference.
    def dfs(items):
        if len(items) == 1:
            v, expr = items[0]
            return expr if v == target and (accept is None or accept(expr)) else None
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                rest = [x for k, x in enumerate(items) if k not in (i, j)]
                a, b = items[i], items[j]
                candidates = [combine(a,b,"+"), combine(a,b,"*"), combine(a,b,"-"),
                              combine(b,a,"-"), combine(a,b,"/"), combine(b,a,"/")]
                for item in candidates:
                    if item is None: continue
                    found = dfs(rest+[item])
                    if found is not None: return found
        return None
    return dfs([(Fraction(n), str(n)) for n in numbers])

def matches(expr, profile):
    _, depth, frac = features(expr)
    if profile == "ood-arity": return depth == 3 and not frac
    if profile == "ood-fraction": return depth == 2 and frac
    if profile == "ood-depth": return depth == 3 and not frac
    return depth == 2 and not frac

def generate(count, seed=42, method="expression", profile="id", exclude=()):
    if profile not in PROFILES or method not in {"expression", "solver"}:
        raise ValueError("Unknown method/profile")
    if count < 1: raise ValueError("count must be positive")
    if method == "solver" and profile == "ood-arity":
        raise ValueError("5-number solver profile omitted for CPU budget; use expression")
    rng = random.Random(seed); seen = set(exclude); rows = []
    low, high = (11,20) if profile == "ood-range" else (1,10)
    arity = 5 if profile == "ood-arity" else 4
    for attempt in range(count*2000):
        numbers = [rng.randint(low,high) for _ in range(arity)]
        if method == "expression":
            item = random_expression(numbers,rng,deep=profile=="ood-depth")
            if item is None: continue
            value, expr = item
            if value.denominator != 1 or not 1 <= value <= 200: continue
            target = int(value)
            if not matches(expr,profile): continue
        else:
            target = rng.randint(1,200)
            expr = solve(numbers,target,accept=lambda e:matches(e,profile))
            if expr is None: continue
        key = canonical_key(numbers,target)
        if key in seen: continue
        _, depth, frac = features(expr)
        seen.add(key)
        rows.append(dict(id=f"pending-{len(rows):06d}", numbers=numbers, target=target,
                         reference_expression=expr, difficulty="hard" if frac else ("medium" if depth>2 else "easy"),
                         reference_depth=depth, reference_has_fraction=frac,
                         generator_version=VERSION, split=profile))
        if len(rows) == count: break
    if len(rows) != count:
        raise RuntimeError(f"Only {len(rows)}/{count} unique problems; reduce count/change versioned profile")
    assert_dataset_valid(rows)
    return rows

def build(out, count, seed, method, profile, excludes):
    out = Path(out)
    if profile == "id" and count < 20:
        raise ValueError("ID count >= 20 so validation/test are nonempty")
    if profile in {"ood-fraction", "ood-depth"} and not excludes:
        raise ValueError("Pass all frozen ID files through --exclude")
    excluded_rows = [r for path in excludes for r in read_jsonl(path)]
    excluded = assert_dataset_valid(excluded_rows)
    rows = generate(count,seed,method,profile,excluded)
    if profile == "id":
        random.Random(seed+1).shuffle(rows)
        a,b = int(count*.8),int(count*.9)
        groups = {"train":rows[:a],"validation":rows[a:b],"test-id":rows[b:]}
    else: groups = {profile:rows}
    for split, group in groups.items():
        for i,row in enumerate(group): row.update(id=f"{split}-{i:06d}",split=split)
    assert_dataset_valid([r for group in groups.values() for r in group])
    contents = {f"{split}.jsonl":"".join(json.dumps(r,ensure_ascii=False)+"\n" for r in group) for split,group in groups.items()}
    manifest = dict(generator_version=VERSION,verifier_version=VERIFIER_VERSION,
                    seed=seed,method=method,profile=profile,count=count,target_range=[1,200],
                    difficulty_rule="reference fraction => hard; depth>2 => medium; otherwise easy",
                    reference_based_ood=True,
                    excluded_hashes={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in excludes},
                    files={name:dict(sha256=hashlib.sha256(text.encode()).hexdigest(),count=len(text.splitlines())) for name,text in contents.items()})
    contents[f"manifest-{profile}.json"] = json.dumps(manifest,indent=2,ensure_ascii=False)
    if any((out/name).exists() for name in contents):
        raise FileExistsError("Frozen outputs exist; choose a new versioned directory")
    out.mkdir(parents=True,exist_ok=True)
    for name,text in contents.items(): (out/name).write_text(text,encoding="utf-8",newline="\n")
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True);p.add_argument("--count",type=int,default=120)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--method",choices=["expression","solver"],default="expression")
    p.add_argument("--profile",choices=PROFILES,default="id")
    p.add_argument("--exclude",nargs="*",default=[])
    a=p.parse_args();print(json.dumps(build(a.out,a.count,a.seed,a.method,a.profile,a.exclude),indent=2))
if __name__ == "__main__": main()
