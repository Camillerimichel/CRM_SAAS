import os
import re

SRC_DIR = "src"

# motifs à corriger
patterns = {
    "from models": "from src.models",
    "from services": "from src.services",
}

# motif pour vérifier
check_pattern = re.compile(r"^\s*from\s+(models|services)\b")

def fix_imports():
    for root, _, files in os.walk(SRC_DIR):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as file:
                    lines = file.readlines()

                changed = False
                new_lines = []
                for line in lines:
                    new_line = line
                    for bad, good in patterns.items():
                        if line.strip().startswith(bad):
                            new_line = line.replace(bad, good)
                            changed = True
                    new_lines.append(new_line)

                if changed:
                    with open(path, "w", encoding="utf-8") as file:
                        file.writelines(new_lines)
                    print(f"[✅ corrigé] {path}")

def check_imports():
    print("\n--- Vérification après correction ---")
    for root, _, files in os.walk(SRC_DIR):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as file:
                    for i, line in enumerate(file, start=1):
                        if check_pattern.search(line):
                            print(f"[⚠️ reste à corriger] {path}:{i} -> {line.strip()}")

if __name__ == "__main__":
    fix_imports()
    check_imports()
