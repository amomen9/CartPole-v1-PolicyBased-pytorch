import pathlib

p = pathlib.Path("functions.py")
lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()

start = 1060
end = 1200

for i in range(start - 1, min(end, len(lines))):
    print(f"{i+1}: {lines[i]}")
