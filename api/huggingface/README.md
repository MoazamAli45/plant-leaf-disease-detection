---
title: LeafDx API
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Plant leaf disease classification and Grad-CAM heat maps
---

# LeafDx API

Classifies a photo of a plant leaf into one of 38 classes (14 crops, healthy or diseased) with
MobileNetV1 fine-tuned on field photos, and draws a Grad-CAM heat map showing where the model looked. Interactive docs at
`/docs`.

Accuracy on held-out field photos (PlantDoc) is 54%, so treat results as a research
demonstration, not agronomic advice. Source code, evaluation and model card:
<https://github.com/MoazamAli45/plant-leaf-disease-detection>
