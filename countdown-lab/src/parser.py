import ast
import re

VERSION = "parser-v1"
MAX_OUTPUT, MAX_EXPR, MAX_NODES = 8192, 512, 128
ALLOWED = {ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
           ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub}

class ParseError(ValueError):
    def __init__(self, code, parse_ok=False, operators_ok=False):
        super().__init__(code)
        self.code = code
        self.parse_ok = parse_ok
        self.operators_ok = operators_ok

def parse_output(text):
    if not isinstance(text, str) or len(text) > MAX_OUTPUT:
        raise ParseError("OUTPUT_LIMIT")
    if text.count("FINAL:") != 1:
        raise ParseError("MARKER_COUNT")
    expr = text.split("FINAL:", 1)[1].strip(" \t\r\n")
    if not expr or "\n" in expr or "\r" in expr:
        raise ParseError("FINAL_LINE")
    if len(expr) > MAX_EXPR:
        raise ParseError("EXPR_LIMIT")
    # Chỉ ký tự ASCII của biểu thức; không số khoa học/hex/float.
    if not re.fullmatch(r"[0-9+*/() \t\-]+", expr):
        raise ParseError("TOKEN")
    if any(len(x) > 6 for x in re.findall(r"[0-9]+", expr)):
        raise ParseError("LITERAL_LIMIT")
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise ParseError("SYNTAX") from None
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise ParseError("NODE_LIMIT", True)
    if any(type(n) not in ALLOWED for n in nodes):
        raise ParseError("OPERATOR", True)
    if any(type(n.value) is not int for n in nodes if isinstance(n, ast.Constant)):
        raise ParseError("LITERAL", True)
    return expr, tree
