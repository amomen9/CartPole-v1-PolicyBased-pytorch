import pathlib

p = pathlib.Path("functions.py")
lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()

start = 1040
end = 1165

for i in range(start - 1, min(end, len(lines))):
    print(f"{i+1}: {lines[i]}")
