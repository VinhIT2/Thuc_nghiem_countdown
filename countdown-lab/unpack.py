import json
from pathlib import Path
root = Path.cwd().resolve()
files = json.loads(Path("countdown-code-bundle.json").read_text(encoding="utf-8"))
planned = []
for name, content in files.items():
    path = (root / name).resolve()
    if root not in path.parents or path.exists():
        raise ValueError(f"Unsafe or existing path: {name}")
    if not isinstance(content, str):
        raise TypeError(name)
    planned.append((path, content))
for path, content in planned:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
print(f"Wrote {len(planned)} files")
