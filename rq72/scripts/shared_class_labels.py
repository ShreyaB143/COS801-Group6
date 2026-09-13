from __future__ import annotations
import unicodedata

# canonical -> (PlantVillage folder, PlantDoc folder)
SHARED_CLASSES: dict[str, tuple[str, str]] = {
    "apple_scab":            ("Apple___Apple_scab", "Apple Scab Leaf"),
    "apple_rust":            ("Apple___Cedar_apple_rust", "Apple rust leaf"),
    "apple_healthy":         ("Apple___healthy", "Apple leaf"),
    "blueberry_healthy":     ("Blueberry___healthy", "Blueberry leaf"),
    "cherry_healthy":        ("Cherry_(including_sour)___healthy", "Cherry leaf"),
    "corn_gray_spot":        ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn Gray leaf spot"),
    "corn_rust":             ("Corn_(maize)___Common_rust_", "Corn rust leaf"),
    "corn_blight":           ("Corn_(maize)___Northern_Leaf_Blight", "Corn leaf blight"),
    "grape_black_rot":       ("Grape___Black_rot", "grape leaf black rot"),
    "grape_healthy":         ("Grape___healthy", "grape leaf"),
    "peach_healthy":         ("Peach___healthy", "Peach leaf"),
    "pepper_healthy":        ("Pepper,_bell___healthy", "Bell_pepper leaf"),
    "potato_early_blight":   ("Potato___Early_blight", "Potato leaf early blight"),
    "potato_late_blight":    ("Potato___Late_blight", "Potato leaf late blight"),
    "raspberry_healthy":     ("Raspberry___healthy", "Raspberry leaf"),
    "soybean_healthy":       ("Soybean___healthy", "Soyabean leaf"),
    "squash_mildew":         ("Squash___Powdery_mildew", "Squash Powdery mildew leaf"),
    "strawberry_healthy":    ("Strawberry___healthy", "Strawberry leaf"),
    "tomato_bacterial_spot": ("Tomato___Bacterial_spot", "Tomato leaf bacterial spot"),
    "tomato_early_blight":   ("Tomato___Early_blight", "Tomato Early blight leaf"),
    "tomato_late_blight":    ("Tomato___Late_blight", "Tomato leaf late blight"),
    "tomato_leaf_mold":      ("Tomato___Leaf_Mold", "Tomato mold leaf"),
    "tomato_septoria":       ("Tomato___Septoria_leaf_spot", "Tomato Septoria leaf spot"),
    "tomato_yellow_virus":   ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato leaf yellow virus"),
    "tomato_mosaic_virus":   ("Tomato___Tomato_mosaic_virus", "Tomato leaf mosaic virus"),
    "tomato_healthy":        ("Tomato___healthy", "Tomato leaf"),
}

# Excluded classes that are not present in both datasets
