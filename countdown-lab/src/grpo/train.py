import argparse
import json
import statistics
import time
import yaml
import torch
from datasets import Dataset
from peft import PeftModel, PeftConfig
from trl import GRPOConfig, GRPOTrainer
from src.prompts import build_prompt
from src.verifier import verify
from src.reward import score
from src.training_common import prepare,training_args,save_run

class AuditedGRPOTrainer(GRPOTrainer):
    # Diagnostic cho TRL 0.15.2 single-GPU; không gọi là entropy token.
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        if not hasattr(self,"_guide_diag_step"): self._guide_diag_step=-1
        step=self.state.global_step
        if model.training and self._guide_diag_step != step:
            self._guide_diag_step=step
            with open(self.args.output_dir+"/step-events.jsonl","a",encoding="utf-8") as f:
                f.write(json.dumps(dict(step=step,event="optimization_batch"))+"\n")
        return super().compute_loss(model,inputs,return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)

def completion_text(completion):
    if isinstance(completion,str): return completion
    if isinstance(completion,list) and len(completion)==1 and isinstance(completion[0],dict):
        content=completion[0].get("content")
        if isinstance(content,str): return content
    raise ValueError("Unexpected completion schema; inspect pinned TRL")

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args()
    c=yaml.safe_load(open(a.config,encoding="utf-8"))
    if int(__import__("os").environ.get("WORLD_SIZE","1"))!=1:
        raise ValueError("Educational audit callback supports single GPU only")
    g=c["num_generations"]
    if g<2 or c["per_device_train_batch_size"]%g or c["per_device_eval_batch_size"]%g:
        raise ValueError("TRL 0.15.2 per-device batch must be divisible by G on one GPU")
    if c["reward_mode"] not in {"binary","shaped"}: raise ValueError("Reward mode")
    if c["use_vllm"]: raise ValueError("This smoke script pins use_vllm=False")
    adapter_config=PeftConfig.from_pretrained(c["sft_adapter"])
    if adapter_config.base_model_name_or_path != c["model"]:
        raise ValueError("SFT adapter/base model mismatch")
    model,tok,train,val,out=prepare(c)
    tok.padding_side="left"
    model=PeftModel.from_pretrained(model,c["sft_adapter"],is_trainable=True)
    # KHÔNG truyền peft_config vào trainer: adapter đã tồn tại.
    # Reference của TRL v0.15.2 là model khi disable adapter = original base,
    # không phải snapshot SFT. Báo cáo lựa chọn KL này trong experiment card.
    model.print_trainable_parameters()
    def dataset(rows):
        prepared=[]
        for r in rows:
            prompt=build_prompt(r,c["prompt_style"])
            ids=tok.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True)
            if len(ids)>c["max_prompt_length"]:raise ValueError("Prompt would be truncated")
            prepared.append(dict(prompt=prompt,numbers=r["numbers"],target=r["target"],problem_id=r["id"]))
        return Dataset.from_list(prepared)
    callback_index=0
    def countdown_reward(prompts,completions,numbers,target,problem_id,**kwargs):
        nonlocal callback_index
        if not len(completions)==len(numbers)==len(target)==len(problem_id):
            raise ValueError("Reward callback columns not aligned")
        texts=[completion_text(x) for x in completions]
        results=[verify(t,n,y) for t,n,y in zip(texts,numbers,target)]
        rewards=[score(r,c["reward_mode"]) for r in results]
        if len(rewards)%g:raise ValueError("Incomplete reward groups")
        grouped=[rewards[i:i+g] for i in range(0,len(rewards),g)]
        for i in range(0,len(problem_id),g):
            if len(set(problem_id[i:i+g]))!=1: raise ValueError("Unexpected group ordering")
        phase="train" if all(x.startswith("train-") for x in problem_id) else "validation"
        diag=dict(callback_index=callback_index,phase=phase,reward_mean=statistics.mean(rewards),
            reward_std=statistics.pstdev(rewards),group_variance_rate=statistics.mean(float(len(set(x))>1) for x in grouped),
            valid_rate=statistics.mean(float(r["valid"]) for r in results),
            solve_rate=statistics.mean(float(r["valid"] and r["target_ok"]) for r in results),
            entropy=None,entropy_note="not instrumented; do not substitute KL")
        with (out/"reward-diagnostics.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(diag)+"\n")
        with (out/"reward-outputs.jsonl").open("a",encoding="utf-8") as f:
            for pid,text,result,reward in zip(problem_id,texts,results,rewards):
                f.write(json.dumps(dict(callback_index=callback_index,phase=phase,problem_id=pid,
                    output=text,verifier=result,reward=reward),ensure_ascii=False)+"\n")
        callback_index+=1
        return rewards
    args=GRPOConfig(**training_args(c),num_generations=g,beta=c["beta"],temperature=c["temperature"],
        max_prompt_length=c["max_prompt_length"],max_completion_length=c["max_completion_length"],
        use_vllm=False,remove_unused_columns=False)
    trainer=AuditedGRPOTrainer(model=model,args=args,processing_class=tok,reward_funcs=countdown_reward,
        train_dataset=dataset(train),eval_dataset=dataset(val))
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    try:trainer.train()
    finally:
        torch.cuda.synchronize()
        save_run(trainer,tok,out,time.perf_counter()-start)
    print("Validation generation in trainer is diagnostic, not fixed-protocol pass@1. Use baseline.py separately.")
if __name__=="__main__":main()
