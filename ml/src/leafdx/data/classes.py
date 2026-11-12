"""The 38 PlantVillage classes (folder names exactly as published upstream)."""

from __future__ import annotations

PLANTVILLAGE_CLASSES: tuple[str, ...] = (
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
)

CLASS_TO_INDEX: dict[str, int] = {c: i for i, c in enumerate(PLANTVILLAGE_CLASSES)}


def split_class(name: str) -> tuple[str, str]:
    """Return (crop, condition) for a PlantVillage folder name, e.g. ('Tomato', 'Leaf_Mold')."""
    if "___" not in name:
        raise ValueError(f"Not a PlantVillage class name: {name!r}")
    crop, condition = name.split("___", 1)
    return crop, condition


def is_healthy(name: str) -> bool:
    return split_class(name)[1] == "healthy"


def display_name(name: str) -> str:
    """Human-readable label, e.g. 'Corn (maize): Common rust'."""
    crop, condition = split_class(name)
    crop_s = crop.replace(",_", ", ").replace("_", " ")
    cond_s = condition.replace("_", " ").strip()
    cond_s = cond_s[:1].upper() + cond_s[1:]
    return f"{crop_s}: {cond_s}"
