import ast
from collections import Counter
from fractions import Fraction
from .parser import ParseError, parse_output

VERSION = "verifier-v1"

def verify(text, numbers, target):
    r = dict(parse_ok=False, operators_ok=False, numbers_ok=False,
             value=None, target_ok=False, valid=False, error_code=None,
             expression=None)
    if (not isinstance(numbers, list) or not 2 <= len(numbers) <= 6
        or any(type(x) is not int or not 1 <= x <= 999999 for x in numbers)
        or type(target) is not int or not 1 <= target <= 10**9):
        r["error_code"] = "BAD_PROBLEM"
        return r
    try:
        expr, tree = parse_output(text)
    except ParseError as e:
        r.update(parse_ok=e.parse_ok, operators_ok=e.operators_ok, error_code=e.code)
        return r
    r.update(parse_ok=True, operators_ok=True, expression=expr)
    used = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)]
    if Counter(used) != Counter(numbers):
        r["error_code"] = "NUMBERS"
        return r
    r["numbers_ok"] = True
    def compute(n):
        if isinstance(n, ast.Expression):
            return compute(n.body)
        if isinstance(n, ast.Constant):
            return Fraction(n.value)
        if isinstance(n, ast.UnaryOp):
            return -compute(n.operand)
        a, b = compute(n.left), compute(n.right)
        if isinstance(n.op, ast.Add): v = a + b
        elif isinstance(n.op, ast.Sub): v = a - b
        elif isinstance(n.op, ast.Mult): v = a * b
        else: v = a / b
        if v.numerator.bit_length() > 128 or v.denominator.bit_length() > 128:
            raise OverflowError
        return v
    try:
        value = compute(tree)
    except ZeroDivisionError:
        r["error_code"] = "DIV_ZERO"
        return r
    except (OverflowError, RecursionError):
        r["error_code"] = "VALUE_LIMIT"
        return r
    r.update(value=str(value), valid=True, target_ok=value == Fraction(target))
    r["error_code"] = None if r["target_ok"] else "WRONG_TARGET"
    return r
