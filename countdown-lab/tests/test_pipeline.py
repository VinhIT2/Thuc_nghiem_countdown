import copy
import pytest
from src.generator import generate, assert_dataset_valid, canonical_key, build, read_jsonl
from src.eval import evaluate, pass_k
from src.verifier import verify
from src.baseline import select

@pytest.mark.parametrize("profile",["id","ood-range","ood-arity","ood-fraction","ood-depth"])
def test_generator(profile):
    rows=generate(12,42,"expression",profile)
    assert len(assert_dataset_valid(rows))==12
    assert rows==generate(12,42,"expression",profile)

def test_solver():
    assert len(assert_dataset_valid(generate(3,7,"solver","id")))==3

def test_split_freeze(tmp_path):
    build(tmp_path,40,42,"expression","id",[])
    paths=[tmp_path/f"{s}.jsonl" for s in ("train","validation","test-id")]
    rows=[r for p in paths for r in read_jsonl(p)]
    assert len(assert_dataset_valid(rows))==40
    with pytest.raises(FileExistsError): build(tmp_path,40,42,"expression","id",[])
    build(tmp_path,5,43,"expression","ood-fraction",paths)
    ood=read_jsonl(tmp_path/"ood-fraction.jsonl")
    assert not (assert_dataset_valid(rows)&assert_dataset_valid(ood))

def test_duplicate():
    rows=generate(1)
    with pytest.raises(ValueError): assert_dataset_valid(rows+rows)
    assert canonical_key([2,3,7,10],47)==canonical_key([10,7,3,2],47)

def fixture():
    row=dict(id="test-id-000000",numbers=[2,3,7,10],target=47,
        reference_expression="(10-3)*7-2",split="test-id")
    outputs=["FINAL: (10-3)*7-2"]+["FINAL: 2+3+7+10"]*3
    logs=[dict(problem_id=row["id"],split=row["split"],sample_index=i,seed=i,
        output=t,verifier=verify(t,row["numbers"],row["target"]),tokens=10,seconds=.1,
        method="POOL",config_id="t",prompt_hash="x") for i,t in enumerate(outputs)]
    return [row],logs

def test_metric():
    data,logs=fixture();r=evaluate(logs,data,expected_n=4,k=2)
    assert r["pass_at_1"]==.25 and r["pass_at_k"]==.5
    assert r["valid_output_rate"]==1
    assert pass_k(4,0,4)==0 and pass_k(4,4,4)==1

def test_vote_not_oracle():
    _,logs=fixture()
    assert select(logs,"B2")["verifier"]["value"]=="22"
    assert select(logs,"B3")["verifier"]["target_ok"]

@pytest.mark.parametrize("kind",["duplicate","missing","mixed","stale","k","gap"])
def test_bad_log(kind):
    data,logs=fixture();k=1
    if kind=="duplicate":logs.append(copy.deepcopy(logs[0]))
    elif kind=="missing":data.append(data[0]|dict(id="missing",numbers=[1,2,3,4],target=10,reference_expression="1+2+3+4"))
    elif kind=="mixed":logs[0]["config_id"]="other"
    elif kind=="stale":logs[0]["verifier"]["value"]="bad"
    elif kind=="k":k=5
    else:logs[0]["sample_index"]=8
    with pytest.raises(ValueError):evaluate(logs,data,expected_n=4,k=k)

def test_uniform_truncation_rejected():
    data,logs=fixture()
    with pytest.raises(ValueError):evaluate(logs[:1],data,expected_n=4)

def test_selected_budget():
    data,logs=fixture()
    chosen=logs[0]|dict(method="B3",selection_budget=4)
    result=evaluate([chosen],data,expected_n=1,selection_budget=4)
    assert result["selected_success_rate"]==1
    assert "pass_at_1" not in result
    with pytest.raises(ValueError):evaluate([chosen],data,expected_n=1,selection_budget=8)
