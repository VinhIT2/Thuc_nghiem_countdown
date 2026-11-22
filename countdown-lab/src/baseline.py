import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from .generator import read_jsonl, assert_dataset_valid
from .prompts import build_prompt
from .verifier import verify, VERSION

def select(pool, method):
    valid = [r for r in pool if r["verifier"]["valid"]]
    if method == "B3":
        correct = [r for r in valid if r["verifier"]["target_ok"]]
        return (correct or valid or pool)[0]
    if method != "B2": raise ValueError(method)
    if not valid: return pool[0]
    counts = Counter(r["verifier"]["value"] for r in valid)
    best = max(counts.values())
    # Thứ tự pool cố định: tie-break sample xuất hiện đầu tiên.
    return next(r for r in valid if counts[r["verifier"]["value"]] == best)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True);p.add_argument("--out",required=True)
    p.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--revision",required=True,help="40-character model commit SHA")
    p.add_argument("--adapter");p.add_argument("--style",choices=["direct","fewshot"],default="direct")
    p.add_argument("--n",type=int,default=1);p.add_argument("--max-new-tokens",type=int,default=128)
    p.add_argument("--temperature",type=float,default=.8);p.add_argument("--top-p",type=float,default=.95)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--config-id",required=True)
    p.add_argument("--sample",action="store_true",help="sample even if N=1; N>1 always samples")
    a=p.parse_args()
    if a.n < 1 or a.max_new_tokens < 1: raise ValueError("Positive N/length required")
    if len(a.revision)!=40 or any(c not in '0123456789abcdef' for c in a.revision):
        raise ValueError("Pin exact model/tokenizer commit, not main")
    out=Path(a.out);paths=[out,Path(str(out)+".manifest.json")]
    if a.n>1: paths += [Path(str(out)+f".{m}.selected.jsonl") for m in ("B2","B3")]
    if any(x.exists() for x in paths): raise FileExistsError("Use a fresh log path")
    rows=read_jsonl(a.dataset);assert_dataset_valid(rows)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    device="cuda" if torch.cuda.is_available() else "cpu"
    dtype=torch.float32 if device=="cpu" else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    tokenizer=AutoTokenizer.from_pretrained(a.model,revision=a.revision)
    tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,torch_dtype=dtype)
    if a.adapter: model=PeftModel.from_pretrained(model,a.adapter)
    model.to(device);model.eval()
    out.parent.mkdir(parents=True,exist_ok=True)
    manifest=vars(a)|dict(device=device,dtype=str(dtype),verifier_version=VERSION,
        dataset_sha256=hashlib.sha256(Path(a.dataset).read_bytes()).hexdigest(),
        timing="generate wall-time with GPU synchronization; excludes loading/tokenization",
        output_token_count="includes generated EOS; input tokens logged separately")
    Path(str(out)+".manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    selected={m:[] for m in ("B2","B3")}
    with out.open("x",encoding="utf-8") as f:
        for idx,row in enumerate(rows):
            prompt=tokenizer.apply_chat_template(build_prompt(row,a.style),tokenize=False,add_generation_prompt=True)
            inputs=tokenizer(prompt,return_tensors="pt",add_special_tokens=False).to(device)
            pool=[]
            for j in range(a.n):
                seed=a.seed+idx*a.n+j;torch.manual_seed(seed)
                if device=="cuda": torch.cuda.manual_seed_all(seed);torch.cuda.synchronize()
                sampled=a.sample or a.n>1
                options=dict(max_new_tokens=a.max_new_tokens,do_sample=sampled,pad_token_id=tokenizer.pad_token_id)
                if sampled: options.update(temperature=a.temperature,top_p=a.top_p)
                start=time.perf_counter()
                with torch.inference_mode(): result=model.generate(**inputs,**options)
                if device=="cuda": torch.cuda.synchronize()
                seconds=time.perf_counter()-start
                generated=result[0,inputs.input_ids.shape[1]:]
                text=tokenizer.decode(generated,skip_special_tokens=True)
                v=verify(text,row["numbers"],row["target"])
                rec=dict(problem_id=row["id"],split=row["split"],sample_index=j,seed=seed,
                    prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),output=text,expression=v["expression"],
                    verifier=v,error_code=v["error_code"],tokens=len(generated),seconds=seconds,
                    input_tokens=inputs.input_ids.shape[1],method=("B0" if a.style=="direct" else "B1") if a.n==1 else "POOL",
                    config_id=a.config_id)
                f.write(json.dumps(rec,ensure_ascii=False)+"\n");f.flush();pool.append(rec)
            if a.n>1:
                for method in selected:
                    chosen=select(pool,method)
                    selected[method].append(chosen|dict(method=method,sample_index=0,chosen_sample_index=chosen["sample_index"],
                        selection_budget=a.n,tokens=sum(r["tokens"] for r in pool),seconds=sum(r["seconds"] for r in pool),
                        input_tokens=sum(r["input_tokens"] for r in pool),config_id=a.config_id+"-"+method))
    if a.n>1:
        for method,data in selected.items():
            Path(str(out)+f".{method}.selected.jsonl").write_text("".join(json.dumps(r)+"\n" for r in data),encoding="utf-8")
if __name__ == "__main__": main()
