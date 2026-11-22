def build_prompt(row, style="direct"):
    if style not in {"direct", "fewshot"}: raise ValueError(style)
    rule = ("Use every supplied positive integer exactly once (including repeats). "
            "Allowed: + - * /, parentheses, unary minus; exact rational arithmetic. "
            "No other numbers, functions, or zero division. End with exactly one line "
            "FINAL: expression. Do not write anything after it.")
    messages = [{"role":"system","content":rule}]
    if style == "fewshot":
        # Ví dụ riêng, chỉ 3 số: không trùng bài ID 4 số hoặc OOD 5 số.
        messages += [{"role":"user","content":"Numbers: [1, 2, 3]; target: 7"},
                     {"role":"assistant","content":"FINAL: 1 + 2 * 3"},
                     {"role":"user","content":"Numbers: [2, 4, 6]; target: 1"},
                     {"role":"assistant","content":"FINAL: (6 - 4) / 2"}]
    messages.append({"role":"user","content":f"Numbers: {row['numbers']}; target: {row['target']}"})
    return messages
