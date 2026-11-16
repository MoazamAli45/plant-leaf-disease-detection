---
title: LeafDx API
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Plant leaf disease classification and Grad-CAM (research demo)
---

# LeafDx API

FastAPI service for the LeafDx research demo: 38-class plant leaf disease classification with
MobileNetV1 and Grad-CAM heat maps. Research use only; not agronomic advice. Interactive docs
at `/docs`.

Source, evaluation and model card: <https://github.com/MoazamAli45/plant-leaf-disease-detection>.
Field accuracy (PlantDoc) is about 16%; see the model card before relying on any prediction.
