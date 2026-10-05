"""
Why don't the rows in data_split.csv match the images?  (COS 801 Group 6)

Run on Colab after mounting Drive, or locally:

    python diagnose_paths.py                                   # Colab defaults
    python diagnose_paths.py "/path/to/COS 801 Group 6"        # local copy of the group folder
    python diagnose_paths.py "<root>" "<root>/data/data_split.csv"

In a notebook cell:  !python diagnose_paths.py   (or %run diagnose_paths.py)

It never renames, moves or deletes anything. It writes two files next to the CSV:
  * path_diagnosis.csv       - every row that did not resolve, with the reason
  * data_split_resolved.csv  - the same rows and splits, plus a `resolved_path` column
                               where the file was found under a different name.
                               The split itself is untouched.
"""
import os
import re
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

import pandas as pd

# ---- set these if you paste this code into a notebook cell; None = defaults below ----
ROOT_OVERRIDE = None   # e.g. "/content/drive/Shareddrives/COS 801 Group 6"
CSV_OVERRIDE = None    # e.g. "/content/drive/Shareddrives/COS 801 Group 6/data/data_split.csv"

IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
# In a notebook, sys.argv holds the kernel's own "-f <connection file>.json"; ignore those.
ARGS = [a for a in sys.argv[1:] if not a.startswith("-") and not a.endswith(".json")]
ROOT = Path(ROOT_OVERRIDE or (ARGS[0] if ARGS else
            ("/content/drive/Shareddrives/COS 801 Group 6" if IN_COLAB else "."))).expanduser()
CSV = Path(CSV_OVERRIDE or (ARGS[1] if len(ARGS) > 1 else ROOT / "data" / "data_split.csv"))
PATH_COL = "path"

if IN_COLAB and not Path("/content/drive").exists():
    sys.exit("Mount Drive first:  from google.colab import drive; drive.mount('/content/drive')")
if not CSV.exists():
    sys.exit(f"CSV not found: {CSV}\nPass the group folder and the CSV path as arguments.")
if not ROOT.exists():
    sys.exit(f"Folder not found: {ROOT}")

df = pd.read_csv(CSV)
paths = df[PATH_COL].astype(str)
print(f"CSV:  {CSV}  ({len(df)} rows)")
print(f"Root: {ROOT}\n")

# ---------------------------------------------------------------------------
# 1. Which folder are the CSV paths relative to?
#    They start with "data/raw/...", so the base is the folder that CONTAINS `data`.
#    Joining them onto ".../data" or ".../data/raw" doubles the folder name.
# ---------------------------------------------------------------------------
candidates = [ROOT, ROOT / "data", ROOT / "data" / "raw", ROOT.parent, CSV.parent, CSV.parent.parent]
candidates = list(dict.fromkeys(candidates))
sample = paths.sample(min(200, len(paths)), random_state=0)
print("Step 1: which base folder do the paths resolve from? (200-row sample)")
hits = {}
for b in candidates:
    hits[b] = sum((b / p).exists() for p in sample)
    print(f"  {hits[b]:4d}/{len(sample)}  {b}")
base = max(hits, key=hits.get)
if hits[base] == 0:
    print("\n  No base folder works for any sampled row, so this is not just a base-folder mistake.")
    print("  Check that the shared drive really contains:", (ROOT / Path(paths.iloc[0]).parts[0]))
    print("  Continuing with the classification below, using", ROOT)
    base = ROOT
else:
    print(f"\n  Use this as the base:  {base}")
    wrong = [b for b in candidates if b != base and hits[b] == 0 and str(b).startswith(str(base))]
    if wrong:
        print(f"  Joining onto a deeper folder ({wrong[0]}) doubles part of the path, so nothing matches.")

# ---------------------------------------------------------------------------
# 2. For every row: is the file there, and if not, why not?
#    One directory listing per folder, so this is fast even on Drive.
# ---------------------------------------------------------------------------
print("\nStep 2: checking every row ...")
listing_cache = {}


def listdir(folder: Path):
    if folder not in listing_cache:
        try:
            listing_cache[folder] = set(os.listdir(folder))
        except (FileNotFoundError, NotADirectoryError):
            listing_cache[folder] = None
    return listing_cache[folder]


def find_folder(rel_parent: Path):
    """The folder, or a folder whose name differs only in case/whitespace."""
    exact = base / rel_parent
    if listdir(exact) is not None:
        return exact, "same"
    cur = base
    for part in rel_parent.parts:
        names = listdir(cur)
        if names is None:
            return None, None
        if part in names:
            cur = cur / part
            continue
        key = re.sub(r"\s+", " ", part).strip().lower()
        alt = [n for n in names if re.sub(r"\s+", " ", n).strip().lower() == key]
        if len(alt) != 1:
            return None, None
        cur = cur / alt[0]
    return cur, "renamed"


WINDOWS_BAD = re.compile(r'[?*:"<>|]')


def variants(name: str):
    """Names the same file commonly ends up with after a download, unzip, sync or upload."""
    dec = urllib.parse.unquote(name)
    yield "url_decoded", dec
    yield "url_decoded_twice", urllib.parse.unquote(dec)
    yield "windows_chars_replaced", WINDOWS_BAD.sub("_", name)
    yield "windows_chars_removed", WINDOWS_BAD.sub("", name)
    yield "query_string_cut", name.split("?")[0]
    stem, dot, ext = name.rpartition(".")
    if dot:
        yield "extension_case", stem + "." + (ext.lower() if ext != ext.lower() else ext.upper())


basenames_index = None


def find_elsewhere(name):
    global basenames_index
    if basenames_index is None:
        basenames_index = {}
        for folder, names in list(listing_cache.items()):
            for n in names or ():
                basenames_index.setdefault(n, []).append(folder / n)
    return basenames_index.get(name, [])


status, resolved = [], []
for rel in paths:
    rp = Path(rel)
    folder, how = find_folder(rp.parent)
    if folder is None:
        status.append("folder_missing")
        resolved.append(None)
        continue
    names = listdir(folder)
    if rp.name in names:
        status.append("ok" if how == "same" else "folder_renamed")
        resolved.append(str((folder / rp.name).relative_to(base)))
        continue
    lower = {n.lower(): n for n in names}
    hit = None
    for why, cand in variants(rp.name):
        if cand in names:
            hit = (why, cand)
            break
    if hit is None and rp.name.lower() in lower:
        hit = ("name_case", lower[rp.name.lower()])
    if hit:
        status.append(hit[0])
        resolved.append(str((folder / hit[1]).relative_to(base)))
    else:
        status.append("file_missing")
        resolved.append(None)

# second pass: missing files that exist under another folder of the dataset
for i, s in enumerate(status):
    if s == "file_missing":
        other = find_elsewhere(Path(paths.iloc[i]).name)
        if len(other) == 1:
            status[i] = "in_other_folder"
            resolved[i] = str(other[0].relative_to(base))

df["diagnosis"] = status
df["resolved_path"] = resolved
counts = Counter(status)

EXPLAIN = {
    "ok": "found exactly as written",
    "folder_renamed": "folder name differs (case or spaces); file found",
    "url_decoded": "file exists with %20 etc. decoded (e.g. spaces); the CSV keeps the literal %20",
    "url_decoded_twice": "file exists with %25xx decoded twice",
    "windows_chars_replaced": "file exists with ? * : \" < > | replaced by _ (Windows / zip tools)",
    "windows_chars_removed": "file exists with ? * : \" < > | removed",
    "query_string_cut": "file exists with everything after '?' cut off",
    "extension_case": "file exists with .JPG/.jpg case flipped",
    "name_case": "file exists with different letter case",
    "in_other_folder": "file exists, but in a different folder",
    "folder_missing": "the folder itself is not on the drive",
    "file_missing": "not found under any common renaming: not uploaded, or upload failed",
}
print("\nResult per row:")
for k in EXPLAIN:
    if counts.get(k):
        print(f"  {counts[k]:6d}  {k:24s} {EXPLAIN[k]}")

top = paths.str.split("/").str[:4].str.join("/")
bad = df.diagnosis != "ok"
if bad.any():
    print("\nWhere the problems are:")
    print(pd.crosstab(top[bad], df.diagnosis[bad]).to_string())
    print("\nExamples:")
    for k in EXPLAIN:
        ex = df[df.diagnosis == k]
        if k != "ok" and len(ex):
            r = ex.iloc[0]
            print(f"  [{k}]\n     CSV:  {r[PATH_COL]}\n     disk: {r.resolved_path or '-'}")

# ---------------------------------------------------------------------------
# 3. What to do
# ---------------------------------------------------------------------------
ok = counts.get("ok", 0)
fixable = sum(v for k, v in counts.items() if k not in ("ok", "file_missing", "folder_missing"))
lost = counts.get("file_missing", 0) + counts.get("folder_missing", 0)
print(f"\nSummary: {ok} found as written, {fixable} found under another name, {lost} not on the drive.")
if base != ROOT:
    print(f"* Base folder: join the CSV paths onto {base}, not {ROOT}.")
if fixable:
    print("* Renamed files: use the `resolved_path` column of data_split_resolved.csv. Rows and splits are unchanged.")
    print("  Do NOT rename the files on Drive to match: other members' copies would then stop matching.")
if lost:
    print("* Missing files: re-upload those folders, ideally from the same archive that produced the CSV.")
    print("  If the PlantDoc files containing ? were downloaded or unzipped on Windows, they cannot exist there;")
    print("  copy them from a Mac or Linux machine, or from Colab directly (git clone / unzip on Colab keeps them).")
if ok == len(df):
    print("Every row resolves. If a member's code still fails, the problem is in that code, not the data:")
    print("  check that it joins onto the base above and does not split paths on spaces.")

out1 = CSV.with_name("path_diagnosis.csv")
out2 = CSV.with_name("data_split_resolved.csv")
try:
    df[bad].to_csv(out1, index=False)
    df.drop(columns="diagnosis").to_csv(out2, index=False)
    print(f"\nWrote {out1}\n      {out2}")
except OSError as e:
    print(f"\nCould not write next to the CSV ({e}); writing to the current folder instead.")
    df[bad].to_csv("path_diagnosis.csv", index=False)
    df.drop(columns="diagnosis").to_csv("data_split_resolved.csv", index=False)
