def score(result, mode="binary"):
    if mode not in {"binary", "shaped"}:
        raise ValueError("Unknown reward mode")
    if result["valid"] and result["target_ok"]:
        return 1.0
    return 0.1 if mode == "shaped" and result["valid"] else 0.0
