import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path
from .generator import read_jsonl, assert_dataset_valid, VERSION as GENERATOR_VERSION
from .verifier import VERSION as VERIFIER_VERSION

EXPECTED={"torch":"2.6.0","transformers":"4.49.0","trl":"0.15.2",
          "peft":"0.14.0","datasets":"3.3.2","accelerate":"1.4.0"}

def prepare(config):
    import torch
    from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
    for package,wanted in EXPECTED.items():
        actual=importlib.metadata.version(package).split("+")[0]
        if actual != wanted:
            raise RuntimeError(f"{package}: expected {wanted}, got {actual}; use pinned environment")
    if not torch.cuda.is_available(): raise RuntimeError("Training example requires NVIDIA CUDA; core/baseline can use CPU")
    if not re.fullmatch(r"[0-9a-f]{40}",config["revision"]): raise ValueError("Pin model revision")
    out=Path(config["output_dir"])
    if out.exists(): raise FileExistsError("Fresh output_dir required; never overwrite a run")
    train,val=read_jsonl(config["train"]),read_jsonl(config["validation"])
    # Cả keys lẫn IDs phải độc lập giữa train và validation.
    assert_dataset_valid(train+val)
    if not train or not val: raise ValueError("Empty train/validation")
    if any(r["split"]!="train" for r in train) or any(r["split"]!="validation" for r in val):
        raise ValueError("Expected frozen train and validation files, not test")
    precision=config["precision"]
    if precision not in {"bf16","fp16","fp32"}: raise ValueError(precision)
    if precision=="bf16" and not torch.cuda.is_bf16_supported(): raise RuntimeError("GPU lacks bf16; choose fp16 after smoke test")
    dtype={"bf16":torch.bfloat16,"fp16":torch.float16,"fp32":torch.float32}[precision]
    set_seed(config["seed"])
    tok=AutoTokenizer.from_pretrained(config["model"],revision=config["revision"])
    tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(config["model"],revision=config["revision"],torch_dtype=dtype)
    model.config.use_cache=False
    out.mkdir(parents=True)
    env={p:importlib.metadata.version(p) for p in EXPECTED}
    manifest=dict(config=config,packages=env,python=platform.python_version(),
        gpu=torch.cuda.get_device_name(0),generator_version=GENERATOR_VERSION,verifier_version=VERIFIER_VERSION,
        data_hashes={k:hashlib.sha256(Path(config[k]).read_bytes()).hexdigest() for k in ("train","validation")},
        checkpoint_rule="evaluate saved checkpoints on validation externally; final-adapter is NOT automatically best",
        world_size=1)
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return model,tok,train,val,out

def training_args(c):
    return dict(output_dir=c["output_dir"],learning_rate=c["learning_rate"],
        lr_scheduler_type=c["lr_scheduler_type"],warmup_ratio=c["warmup_ratio"],
        max_steps=c["max_steps"],num_train_epochs=c["num_train_epochs"],
        per_device_train_batch_size=c["per_device_train_batch_size"],
        per_device_eval_batch_size=c["per_device_eval_batch_size"],
        gradient_accumulation_steps=c["gradient_accumulation_steps"],
        bf16=c["precision"]=="bf16",fp16=c["precision"]=="fp16",seed=c["seed"],
        gradient_checkpointing=c["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant":False},
        eval_strategy="steps",eval_steps=c["eval_steps"],save_strategy="steps",save_steps=c["save_steps"],
        logging_steps=c["logging_steps"],report_to="none",load_best_model_at_end=False)

def save_run(trainer,tokenizer,out,seconds):
    import torch
    trainer.save_model(str(out/"final-adapter"))
    tokenizer.save_pretrained(out/"final-adapter")
    trainer.save_state()
    data=dict(wall_seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        warning="Final adapter is last step, not necessarily validation-best")
    (out/"runtime.json").write_text(json.dumps(data,indent=2),encoding="utf-8")
