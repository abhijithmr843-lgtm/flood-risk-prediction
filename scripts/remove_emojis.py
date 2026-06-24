# scripts/remove_emojis.py
# PURPOSE: One-time utility to strip emoji characters from all
# project source files (Python). Run once, review the diff in
# git, then delete or keep this script as you prefer.
#
# IMPORTANT: This version preserves leading whitespace (indentation)
# on every line. Indentation is significant in Python — it is never
# stripped or collapsed by this script.

import os
import re

# Unicode ranges that cover the vast majority of emoji characters
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"# symbols & pictographs
    "\U0001F600-\U0001F64F"# emoticons
    "\U0001F680-\U0001F6FF"# transport & map symbols
    "\U0001F700-\U0001F77F"# alchemical symbols
    "\U0001F780-\U0001F7FF"# geometric shapes extended
    "\U0001F800-\U0001F8FF"# supplemental arrows-C
    "\U0001F900-\U0001F9FF"# supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"# chess symbols
    "\U0001FA70-\U0001FAFF"# symbols & pictographs extended-A
    "\U00002700-\U000027BF"# dingbats (includes checkmarks, scissors)
    "\U00002600-\U000026FF"# miscellaneous symbols (sun, cloud, etc.)
    "\U0001F1E6-\U0001F1FF"# flags
    "\U00002B00-\U00002BFF"# arrows, stars
    "\U0001F000-\U0001F0FF"# mahjong/dominoes
    "\U0000FE0F"# variation selector (emoji presentation)
    "\U0000200D"# zero-width joiner (emoji combos)
    "]+",
    flags=re.UNICODE
)

# File types to clean
TARGET_EXTENSIONS = ('.py', '.md', '.txt')

# Folders to skip entirely (virtual env, git internals, cache, data).
# .venv is this project's actual venv folder name.
SKIP_DIRS = {
    '.git', '.venv', '.floodenv', 'flood_env', '__pycache__',
    'node_modules', '.streamlit', 'data', 'models'
}


def collapse_inline_spaces(line):
    """
    Collapse 2+ consecutive spaces/tabs WITHIN a line's content,
    but never touch leading whitespace (indentation).
    """
    stripped = line.lstrip('\t')
    leading_ws = line[:len(line) - len(stripped)]
    collapsed = re.sub(r'[ \t]{2,}', '', stripped)
    return leading_ws + collapsed


def strip_quote_leading_space(line):
    """
    After removing an emoji, a single leftover space often remains
    right after the opening quote of a string, e.g.:
        print("Loaded data") -> print("Loaded data") (after emoji removal)
    This strips that one leftover space so it becomes:
        print("Loaded data")
    Matches a quote character followed by whitespace, anywhere in
    the line. Leading indentation is handled separately and is
    never touched by this function.
    """
    stripped = line.lstrip('\t')
    leading_ws = line[:len(line) - len(stripped)]

    # Any quote character ('or ") immediately followed by one or
    # more spaces/tabs, followed by a non-space character: drop the
    # whitespace between the quote and the next character.
    content = re.sub(r'(["\'])[ \t]+(?=\S)', r'\1', stripped)

    return leading_ws + content


def clean_file(filepath):
    """Remove emoji characters from one file, return True if changed."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original = f.read()
    except (UnicodeDecodeError, PermissionError):
        return False

    cleaned = EMOJI_PATTERN.sub('', original)

    lines = cleaned.split('\n')

    # Pass 1: collapse leftover double-spaces in line content
    # (indentation untouched)
    lines = [collapse_inline_spaces(line) for line in lines]

    # Pass 2: strip a single leftover space right after an opening
    # quote, where an emoji used to sit (indentation untouched)
    lines = [strip_quote_leading_space(line) for line in lines]

    cleaned = '\n'.join(lines)

    # Remove trailing spaces before newlines (safe — does not affect
    # indentation, which is always at the START of a line, not the end)
    cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)

    if cleaned != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        return True
    return False


def find_target_files(root_dir='.'):
    """Walk the project and collect files to clean, skipping noise dirs."""
    targets = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            if filename.endswith(TARGET_EXTENSIONS):
                targets.append(os.path.join(dirpath, filename))
    return targets


def main():
    print("Scanning project for emoji characters...")
    files = find_target_files('.')
    print(f"Found {len(files)} candidate files\n")

    changed_files = []
    for filepath in files:
        if clean_file(filepath):
            changed_files.append(filepath)
            print(f"Cleaned: {filepath}")

    print(f"\nDone. {len(changed_files)} file(s) modified out of {len(files)} scanned.")
    if changed_files:
        print("\nReview changes with: git --no-pager diff")
        print('Then commit with: git add . && git commit -m "Remove emojis from codebase"')


if __name__ == "__main__":
    main()