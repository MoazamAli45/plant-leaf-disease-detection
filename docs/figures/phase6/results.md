| Model | Method | k / class | Seeds | Field-val acc. | Field-test acc. | Field-test macro-F1 | Field-test crop acc. | Field-test ECE | Lab check acc. |
|---|---|---|---|---|---|---|---|---|---|
| Custom CNN | baseline | - | 1 | 7.61 | 8.35 | 7.76 | 23.94 | 67.48 | 88.20 |
| Custom CNN | test-time augmentation | - | 1 | 6.82 | 9.24 | 8.36 | 24.28 | 57.06 | 88.61 |
| EfficientNetB0 | baseline | - | 1 | 15.49 | 19.27 | 18.86 | 40.53 | 59.10 | 99.60 |
| EfficientNetB0 | test-time augmentation | - | 1 | 18.37 | 19.38 | 19.23 | 42.09 | 54.70 | 99.73 |
| EfficientNetB0 | AdaBN (unlabelled field images) | - | 3 | 19.60 ± 0.15 | 20.56 ± 0.26 | 19.09 ± 0.21 | 42.84 ± 0.17 | 51.00 ± 0.32 | 91.87 ± 0.08 |
| EfficientNetB0 | fine-tuning | 5 | 3 | 32.63 ± 0.15 | 31.48 ± 0.82 | 29.94 ± 0.68 | 53.38 ± 0.74 | 21.65 ± 3.89 | 99.55 ± 0.08 |
| MobileNetV1 | baseline | - | 1 | 14.96 | 18.15 | 16.62 | 39.76 | 59.73 | 99.06 |
| MobileNetV1 | test-time augmentation | - | 1 | 17.06 | 20.04 | 18.88 | 41.31 | 51.06 | 99.20 |
| MobileNetV1 | AdaBN (unlabelled field images) | - | 3 | 19.07 ± 0.40 | 19.78 ± 0.13 | 19.25 ± 0.12 | 43.76 ± 0.00 | 56.20 ± 0.16 | 93.16 ± 0.00 |
| MobileNetV1 | fine-tuning | 5 | 3 | 23.53 ± 1.71 | 26.58 ± 1.36 | 24.86 ± 0.89 | 48.48 ± 3.45 | 14.72 ± 1.63 | 98.53 ± 0.82 |
| MobileNetV1 | fine-tuning | 10 | 3 | 33.16 ± 1.71 | 34.19 ± 0.59 | 30.96 ± 0.74 | 56.94 ± 1.00 | 14.27 ± 2.26 | 97.94 ± 0.76 |
| MobileNetV1 | fine-tuning | 20 | 3 | 39.55 ± 2.26 | 40.76 ± 0.70 | 37.36 ± 0.86 | 62.47 ± 1.28 | 13.01 ± 0.40 | 98.17 ± 0.43 |
| MobileNetV1 | fine-tuning | all | 3 | 48.38 ± 1.97 | 49.33 ± 1.78 | 43.25 ± 1.82 | 68.15 ± 1.93 | 9.24 ± 2.86 | 96.38 ± 0.48 |
| MobileNetV1 | fine-tuning (lr 0.0001) | all | 1 | 43.31 | 44.21 | 38.72 | 61.02 | 5.30 | 91.42 |
| MobileNetV1 | fine-tuning (50 lab images/class replayed) | all | 3 | 47.86 ± 2.12 | 50.00 ± 0.68 | 45.76 ± 1.22 | 69.34 ± 0.72 | 9.91 ± 0.85 | 97.94 ± 0.47 |
| MobileNetV1 | fine-tuning + test-time augmentation (50 lab images/class replayed) | all | 3 | 51.71 ± 2.59 | 53.08 ± 1.28 | 48.96 ± 1.04 | 72.16 ± 0.95 | 7.24 ± 0.21 | 98.26 ± 0.71 |
| MobileNetV2 | baseline | - | 1 | 12.34 | 12.92 | 12.15 | 28.51 | 68.00 | 97.72 |
| MobileNetV2 | test-time augmentation | - | 1 | 12.86 | 15.26 | 14.52 | 30.62 | 56.54 | 98.26 |
| MobileNetV2 | AdaBN (unlabelled field images) | - | 3 | 16.45 ± 0.15 | 18.41 ± 0.23 | 18.21 ± 0.21 | 41.05 ± 0.36 | 57.61 ± 0.23 | 93.57 ± 0.00 |
| MobileNetV3-Small | baseline | - | 1 | 12.86 | 14.48 | 13.59 | 33.52 | 58.24 | 98.93 |
| MobileNetV3-Small | test-time augmentation | - | 1 | 15.22 | 16.70 | 15.84 | 36.64 | 51.78 | 99.20 |
| MobileNetV3-Small | AdaBN (unlabelled field images) | - | 3 | 14.70 ± 0.00 | 17.04 ± 0.29 | 16.55 ± 0.38 | 40.79 ± 0.13 | 55.67 ± 0.19 | 86.55 ± 0.08 |
