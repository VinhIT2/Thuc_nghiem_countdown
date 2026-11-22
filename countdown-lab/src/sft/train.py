import argparse
import json
import time
import yaml
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer
from src.prompts import build_prompt
from src.training_common import prepare,training_args,save_run

class CompletionCollator:
    # Nhãn -100 cho prompt và padding; chỉ học assistant completion.
    def __init__(self,tokenizer): self.tokenizer=tokenizer
    def __call__(self,examples):
        length=max(len(e["input_ids"]) for e in examples)
        batch={"input_ids":[],"attention_mask":[],"labels":[]}
        for e in examples:
            pad=length-len(e["input_ids"])
            batch["input_ids"].append(e["input_ids"]+[self.tokenizer.pad_token_id]*pad)
            batch["attention_mask"].append([1]*len(e["input_ids"])+[0]*pad)
            batch["labels"].append(e["labels"]+[-100]*pad)
        return {k:torch.tensor(v,dtype=torch.long) for k,v in batch.items()}

def encode_rows(rows,tok,c):
    encoded=[]
    for row in rows:
        messages=build_prompt(row,c["prompt_style"])
        prefix=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        full=tok.apply_chat_template(messages+[{"role":"assistant","content":"FINAL: "+row["reference_expression"]}],
            tokenize=False,add_generation_prompt=False)
        prefix_ids=tok(prefix,add_special_tokens=False)["input_ids"]
        ids=tok(full,add_special_tokens=False)["input_ids"]
        if ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError("Chat token boundary mismatch; inspect tokenizer template")
        if len(ids)>c["max_seq_length"] or len(ids)<=len(prefix_ids):
            raise ValueError("Too long or empty target; do not silently truncate FINAL")
        encoded.append(dict(input_ids=ids,attention_mask=[1]*len(ids),
            labels=[-100]*len(prefix_ids)+ids[len(prefix_ids):]))
    return Dataset.from_list(encoded)

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args()
    c=yaml.safe_load(open(a.config,encoding="utf-8"))
    model,tok,train,val,out=prepare(c)
    tok.padding_side="right"
    lora=LoraConfig(task_type=TaskType.CAUSAL_LM,bias="none",**c["lora"])
    args=SFTConfig(**training_args(c),max_seq_length=c["max_seq_length"],packing=False,
        dataset_kwargs={"skip_prepare_dataset":True},remove_unused_columns=False)
    trainer=SFTTrainer(model=model,args=args,processing_class=tok,peft_config=lora,
        train_dataset=encode_rows(train,tok,c),eval_dataset=encode_rows(val,tok,c),
        data_collator=CompletionCollator(tok))
    trainer.model.print_trainable_parameters()
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    try: trainer.train()
    finally:
        torch.cuda.synchronize()
        save_run(trainer,tok,out,time.perf_counter()-start)
    print("Evaluate saved checkpoints on validation before selecting final test adapter.")
if __name__=="__main__":main()
