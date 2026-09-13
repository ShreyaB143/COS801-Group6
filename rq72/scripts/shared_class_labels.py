#python shared_class_labels.py
#helper script for mapping shared classes between the plantvillage and plantdoc datasets
import json
import unicodedata
from pathlib import Path

# canonical -> (PlantVillage folder, PlantDoc folder)
SHARED_CLASSES: dict[str, tuple[str, str]] = {
    "apple_scab": ("Apple___Apple_scab", "Apple Scab Leaf"),
    "apple_rust": ("Apple___Cedar_apple_rust", "Apple rust leaf"),
    "apple_healthy": ("Apple___healthy", "Apple leaf"),
    "blueberry_healthy": ("Blueberry___healthy", "Blueberry leaf"),
    "cherry_healthy": ("Cherry_(including_sour)___healthy", "Cherry leaf"),
    "corn_gray_spot": ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn Gray leaf spot"),
    "corn_rust": ("Corn_(maize)___Common_rust_", "Corn rust leaf"),
    "corn_blight": ("Corn_(maize)___Northern_Leaf_Blight", "Corn leaf blight"),
    "grape_black_rot": ("Grape___Black_rot", "grape leaf black rot"),
    "grape_healthy": ("Grape___healthy", "grape leaf"),
    "peach_healthy": ("Peach___healthy", "Peach leaf"),
    "pepper_healthy": ("Pepper,_bell___healthy", "Bell_pepper leaf"),
    "potato_early_blight": ("Potato___Early_blight", "Potato leaf early blight"),
    "potato_late_blight": ("Potato___Late_blight", "Potato leaf late blight"),
    "raspberry_healthy": ("Raspberry___healthy", "Raspberry leaf"),
    "soybean_healthy": ("Soybean___healthy", "Soyabean leaf"),
    "squash_mildew": ("Squash___Powdery_mildew", "Squash Powdery mildew leaf"),
    "strawberry_healthy": ("Strawberry___healthy", "Strawberry leaf"),
    "tomato_bacterial_spot": ("Tomato___Bacterial_spot", "Tomato leaf bacterial spot"),
    "tomato_early_blight": ("Tomato___Early_blight", "Tomato Early blight leaf"),
    "tomato_late_blight": ("Tomato___Late_blight", "Tomato leaf late blight"),
    "tomato_leaf_mold": ("Tomato___Leaf_Mold", "Tomato mold leaf"),
    "tomato_septoria": ("Tomato___Septoria_leaf_spot", "Tomato Septoria leaf spot"),
    "tomato_yellow_virus": ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato leaf yellow virus"),
    "tomato_mosaic_virus": ("Tomato___Tomato_mosaic_virus", "Tomato leaf mosaic virus"),
    "tomato_healthy": ("Tomato___healthy", "Tomato leaf"),
}

# Excluded classes that are not present in both datasets

CLASS_NAMES = tuple(SHARED_CLASSES)
N_CLASSES = len(CLASS_NAMES)
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


def _norm(name):
    return " ".join(unicodedata.normalize("NFKC", name).split()).casefold()


_LOOKUP = {
    "lab":   {_norm(pv): c for c, (pv, _) in SHARED_CLASSES.items()},
    "field": {_norm(pd): c for c, (_, pd) in SHARED_CLASSES.items()},
}
assert len(_LOOKUP["lab"]) == N_CLASSES, "a PlantVillage folder is mapped twice"
assert len(_LOOKUP["field"]) == N_CLASSES, "a PlantDoc folder is mapped twice"

def to_canonical(folder_name, domain):
    return _LOOKUP[domain].get(_norm(folder_name))

def to_index(folder_name, domain):
    name = to_canonical(folder_name, domain)
    return None if name is None else CLASS_INDEX[name]

try:
    PROJECT = Path(__file__).resolve().parents[2]   
except NameError:                                   
    PROJECT = Path.cwd()                            

DATA = PROJECT / "data" / "raw"
LAB_ROOT = DATA / "plantvillage" / "plantvillage dataset" / "color"
FIELD_ROOTS = [DATA / "plantdoc" / "train", DATA / "plantdoc" / "test"]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _subfolders(path):
    return {_norm(p.name) for p in path.iterdir() if p.is_dir()}

def _check(root, domain):
    root = Path(root)
    if not root.is_dir():
        existing = root
        while not existing.is_dir() and existing != existing.parent:
            existing = existing.parent
        found = sorted(p.name for p in existing.iterdir() if p.is_dir())[:20]
        raise FileNotFoundError(
            f"{domain} path does not exist:\n  {root}\n"
            f"Deepest folder that does exist:\n  {existing}\n"
            f"It contains: {found or 'no subfolders'}\n"
            f"Fix the paths at the top of section 2."
        )
    if domain == "lab" and any(word in str(root).casefold()
                               for word in ("grayscale", "greyscale", "segmented")):
        raise ValueError(
            f"LAB_ROOT points at a greyscale or segmented copy:\n  {root}\n"
            f"Use the colour copy — several experiments manipulate colour directly."
        )
    return root

def _collect(root, domain):
    pairs, skipped = [], {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        images = [p for p in sorted(folder.rglob("*"))
                  if p.suffix.lower() in IMAGE_SUFFIXES]
        label = to_index(folder.name, domain)
        if label is None:
            skipped[folder.name] = skipped.get(folder.name, 0) + len(images)
        else:
            pairs += [(p, label) for p in images]
    return pairs, skipped

def build_index(lab_root=LAB_ROOT, field_roots=FIELD_ROOTS):
    lab_root = _check(lab_root, "lab")
    field_roots = [_check(root, "field") for root in field_roots]

    lab, skipped = _collect(lab_root, "lab")
    field = []
    for root in field_roots:
        pairs, also_skipped = _collect(root, "field")
        field += pairs
        for name, n in also_skipped.items():
            skipped[name] = skipped.get(name, 0) + n

    lab_folders = _subfolders(lab_root)
    field_folders = set()
    for root in field_roots:
        field_folders |= _subfolders(root)

    missing = []
    for pv, pd in SHARED_CLASSES.values():
        if _norm(pv) not in lab_folders:
            missing.append(pv)
        if _norm(pd) not in field_folders:
            missing.append(pd)

    return {
        "lab_root": lab_root,
        "field_roots": field_roots,
        "lab": lab,
        "field": field,
        "skipped": skipped,
        "missing": missing,
    }

def count_per_class(pairs):
    counts = [0] * N_CLASSES
    for _, label in pairs:
        counts[label] += 1
    return counts


def report(index, min_field=15, min_lab=50):
    """Print the counts and the warnings. Returns the number of problems."""
    print(f"lab   root    {index['lab_root']}")
    for n, root in enumerate(index["field_roots"]):
        print(f"{'field root    ' if n == 0 else '              '}{root}")
    print()

    lab = count_per_class(index["lab"])
    field = count_per_class(index["field"])
    w = max(len(name) for name in CLASS_NAMES)

    print(f"{'class':<{w}}  {'lab':>7}  {'field':>6}")
    print("-" * (w + 17))
    for i, name in enumerate(CLASS_NAMES):
        if lab[i] == 0 or field[i] == 0:
            note = "  <- EMPTY"
        elif field[i] < min_field:
            note = "  <- thin field class"
        elif lab[i] < min_lab:
            note = "  <- thin lab class"
        else:
            note = ""
        print(f"{name:<{w}}  {lab[i]:>7}  {field[i]:>6}{note}")
    print("-" * (w + 17))
    print(f"{'total':<{w}}  {sum(lab):>7}  {sum(field):>6}")
    print(f"\n{N_CLASSES} classes.")

    skipped = index["skipped"]
    if skipped:
        total = sum(skipped.values())
        print(f"\nNot in the map — {total} images in {len(skipped)} folders:")
        for name, n in sorted(skipped.items(), key=lambda item: -item[1]):
            print(f"  {name}  ({n})")
        print("  Any of these that could pair across the two datasets needs a")
        print("  reason recorded in the exclusions comment above.")

    thin = [CLASS_NAMES[i] for i in range(N_CLASSES) if 0 < field[i] < min_field]
    if thin:
        print(f"\nUnder {min_field} field images: {', '.join(thin)}")
        print("  Macro-F1 weights these the same as the largest class. Either drop")
        print("  them and say so, or keep them and state the per-class noise floor.")

    problems = [f"folder in the map but not on disk: {name!r}"
                for name in index["missing"]]
    problems += [f"{CLASS_NAMES[i]}: no images on one side"
                 for i in range(N_CLASSES) if lab[i] == 0 or field[i] == 0]
    if problems:
        print(f"\n{len(problems)} PROBLEM(S) — do not train through these:")
        for line in problems:
            print(f"  - {line}")
    else:
        print("\nChecks passed.")
    return len(problems)


def save_classes(path="classes.json"):
    Path(path).write_text(json.dumps({
        "n_classes": N_CLASSES,
        "class_names": list(CLASS_NAMES),          # index i == logit i
        "plantvillage": {k: v[0] for k, v in SHARED_CLASSES.items()},
        "plantdoc": {k: v[1] for k, v in SHARED_CLASSES.items()},
    }, indent=2))


if __name__ == "__main__":
    try:
        problems = report(build_index())
    except BrokenPipeError:
        raise SystemExit(0)  
    raise SystemExit(1 if problems else 0)
