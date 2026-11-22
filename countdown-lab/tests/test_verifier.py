import pytest
from src.verifier import verify
from src.reward import score

@pytest.mark.parametrize("text,numbers,target,valid,correct,error",[
("FINAL: (10-3)*7-2",[2,3,7,10],47,True,True,None),
("FINAL: 2+3+7+10",[2,3,7,10],47,True,False,"WRONG_TARGET"),
("FINAL: 10+7+3",[2,3,7,10],47,False,False,"NUMBERS"),
("FINAL: (10-3)*7-2+0",[2,3,7,10],47,False,False,"NUMBERS"),
("FINAL: 2+2+3+7",[2,2,3,7],14,True,True,None),
("FINAL: 2+3+7",[2,2,3,7],14,False,False,"NUMBERS"),
("FINAL: 2**3+7+10",[2,3,7,10],25,False,False,"OPERATOR"),
("FINAL: sum([2,3,7,10])",[2,3,7,10],22,False,False,"TOKEN"),
("FINAL: (10+2)/(3-3)",[2,3,3,10],1,False,False,"DIV_ZERO"),
("FINAL: (6/3)/(2/4)",[2,3,4,6],4,True,True,None),
("FINAL: -3+10+7+2",[2,3,7,10],16,True,True,None),
("FINAL: "+"1"*9000,[2,3,7,10],47,False,False,"OUTPUT_LIMIT"),
("FINAL: __import__('os')",[2,3,7,10],47,False,False,"TOKEN"),
("FINAL: x+2+3+7",[2,3,7,10],47,False,False,"TOKEN"),
("FINAL: 2+3+7+10\nFINAL: 47",[2,3,7,10],47,False,False,"MARKER_COUNT"),
(" reasoning\nFINAL: \t(10-3)*7-2 \n",[2,3,7,10],47,True,True,None),
("FINAL: (10−3)*7-2",[2,3,7,10],47,False,False,"TOKEN"),
("FINAL: 1e1+3+7+2",[2,3,7,10],22,False,False,"TOKEN"),
("FINAL: 0xA+3+7+2",[2,3,7,10],22,False,False,"TOKEN"),
("FINAL: 10.0+3+7+2",[2,3,7,10],22,False,False,"TOKEN"),
("FINAL: True+3+7+2",[1,2,3,7],13,False,False,"TOKEN"),
("FINAL: 10+3+7+2\nThanks",[2,3,7,10],22,False,False,"FINAL_LINE"),
("FINAL: 10//3+7+2",[2,3,7,10],12,False,False,"OPERATOR"),
("FINAL: (10+3+7+2",[2,3,7,10],22,False,False,"SYNTAX"),
("FINAL: 1000000+3+7+2",[2,3,7,10],22,False,False,"LITERAL_LIMIT"),
])
def test_contract(text,numbers,target,valid,correct,error):
    r=verify(text,numbers,target)
    assert (r["valid"],r["target_ok"],r["error_code"]) == (valid,correct,error)
    assert score(r)==float(valid and correct)
    assert score(r,"shaped")== (1.0 if valid and correct else .1 if valid else 0)

def test_bad_problem():
    assert verify("FINAL: 1+2",[True,2],3)["error_code"]=="BAD_PROBLEM"

def test_bad_reward_mode():
    with pytest.raises(ValueError): score({},"unknown")
