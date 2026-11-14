| model | format | params | size_mb | cpu_latency_ms_median | cpu_latency_ms_p90 | accuracy | agreement_with_keras | xnnpack |
|---|---|---|---|---|---|---|---|---|
| p3_cnn_softmax | Keras (float32) | 2426534 | 9.76 | 2.64 | 2.66 | 88.20 | 100.00 | nan |
| p3_cnn_softmax | TFLite float32 | 2426534 | 9.71 | 1.39 | 1.52 | 88.20 | 100.00 | True |
| p3_cnn_softmax | TFLite dynamic-range | 2426534 | 2.44 | 0.74 | 0.90 | 88.07 | 99.46 | True |
| p3_cnn_softmax | TFLite INT8 | 2426534 | 2.45 | 0.76 | 1.15 | 87.80 | 98.66 | True |
| p3_cnn_softmax | TF.js float32 | 2426534 | 9.72 |  |  |  |  | nan |
| p3_cnn_softmax | TF.js uint8 weights | 2426534 | 2.44 |  |  |  |  | nan |
| p3_efficientnet_b0 | Keras (float32) | 4098249 | 17.24 | 11.61 | 11.98 | 99.60 | 100.00 | nan |
| p3_efficientnet_b0 | TFLite float32 | 4098249 | 16.23 | 5.42 | 6.18 | 99.60 | 100.00 | True |
| p3_efficientnet_b0 | TFLite dynamic-range | 4098249 | 4.60 | 6.71 | 6.84 | 99.46 | 99.60 | True |
| p3_efficientnet_b0 | TFLite INT8 | 4098249 | 4.96 | 2.38 | 3.00 | 69.84 | 69.97 | True |
| p3_efficientnet_b0 | TF.js float32 | 4098249 | 16.48 |  |  |  |  | nan |
| p3_efficientnet_b0 | TF.js uint8 weights | 4098249 | 4.37 |  |  |  |  | nan |
| p3_mobilenet_v1 | Keras (float32) | 3267814 | 13.40 | 7.40 | 7.54 | 99.06 | 100.00 | nan |
| p3_mobilenet_v1 | TFLite float32 | 3267814 | 12.96 | 3.95 | 4.07 | 99.06 | 100.00 | True |
| p3_mobilenet_v1 | TFLite dynamic-range | 3267814 | 3.42 | 2.78 | 2.91 | 99.06 | 99.87 | True |
| p3_mobilenet_v1 | TFLite INT8 | 3267814 | 3.55 | 1.39 | 1.41 | 98.53 | 99.20 | True |
| p3_mobilenet_v1 | TF.js float32 | 3267814 | 13.03 |  |  |  |  | nan |
| p3_mobilenet_v1 | TF.js uint8 weights | 3267814 | 3.32 |  |  |  |  | nan |
| p3_mobilenet_v2 | Keras (float32) | 2306662 | 9.82 | 7.68 | 8.06 | 97.72 | 100.00 | nan |
| p3_mobilenet_v2 | TFLite float32 | 2306662 | 9.05 | 2.77 | 3.83 | 97.72 | 100.00 | True |
| p3_mobilenet_v2 | TFLite dynamic-range | 2306662 | 2.55 | 3.20 | 3.35 | 97.32 | 99.33 | True |
| p3_mobilenet_v2 | TFLite INT8 | 2306662 | 2.76 | 1.35 | 1.37 | 95.58 | 97.05 | True |
| p3_mobilenet_v2 | TF.js float32 | 2306662 | 9.18 |  |  |  |  | nan |
| p3_mobilenet_v2 | TF.js uint8 weights | 2306662 | 2.41 |  |  |  |  | nan |
| p3_mobilenet_v3_small | Keras (float32) | 961046 | 4.42 | 3.75 | 3.81 | 98.93 | 100.00 | nan |
| p3_mobilenet_v3_small | TFLite float32 | 961046 | 3.82 | 0.97 | 1.01 | 98.93 | 100.00 | True |
| p3_mobilenet_v3_small | TFLite dynamic-range | 961046 | 1.13 | 1.22 | 1.53 | 99.20 | 99.20 | True |
| p3_mobilenet_v3_small | TFLite INT8 | 961046 | 1.24 | 0.91 | 0.94 | 60.86 | 60.72 | False |
| p3_mobilenet_v3_small | TF.js float32 | 961046 | 3.97 |  |  |  |  | nan |
| p3_mobilenet_v3_small | TF.js uint8 weights | 961046 | 1.15 |  |  |  |  | nan |
