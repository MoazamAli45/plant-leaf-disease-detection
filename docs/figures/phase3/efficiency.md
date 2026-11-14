| model | format | params | size_mb | cpu_latency_ms_median | cpu_latency_ms_p90 | accuracy | agreement_with_keras |
|---|---|---|---|---|---|---|---|
| p3_cnn_softmax | Keras (float32) | 2426534 | 9.76 | 2.68 | 3.15 | 88.20 | 100.00 |
| p3_cnn_softmax | TFLite float32 | 2426534 | 9.71 | 1.36 | 1.37 | 88.20 | 100.00 |
| p3_cnn_softmax | TFLite dynamic-range | 2426534 | 2.44 | 0.71 | 0.73 | 88.07 | 99.46 |
| p3_cnn_softmax | TFLite INT8 | 2426534 | 2.45 | 0.57 | 0.58 | 87.80 | 98.66 |
| p3_cnn_softmax | TF.js float32 | 2426534 | 9.72 |  |  |  |  |
| p3_cnn_softmax | TF.js uint8 weights | 2426534 | 2.44 |  |  |  |  |
| p3_efficientnet_b0 | Keras (float32) | 4098249 | 17.24 | 11.18 | 11.31 | 99.60 | 100.00 |
| p3_efficientnet_b0 | TFLite float32 | 4098249 | 16.23 | 5.41 | 5.82 | 99.60 | 100.00 |
| p3_efficientnet_b0 | TFLite dynamic-range | 4098249 | 4.60 | 6.71 | 7.16 | 99.46 | 99.60 |
| p3_efficientnet_b0 | TFLite INT8 | 4098249 | 4.96 | 2.27 | 2.46 | 69.84 | 69.97 |
| p3_efficientnet_b0 | TF.js float32 | 4098249 | 16.48 |  |  |  |  |
| p3_efficientnet_b0 | TF.js uint8 weights | 4098249 | 4.37 |  |  |  |  |
| p3_mobilenet_v1 | Keras (float32) | 3267814 | 13.40 | 7.30 | 7.56 | 99.06 | 100.00 |
| p3_mobilenet_v1 | TFLite float32 | 3267814 | 12.96 | 3.93 | 4.11 | 99.06 | 100.00 |
| p3_mobilenet_v1 | TFLite dynamic-range | 3267814 | 3.42 | 2.84 | 2.89 | 99.06 | 99.87 |
| p3_mobilenet_v1 | TFLite INT8 | 3267814 | 3.55 | 1.38 | 1.40 | 98.53 | 99.20 |
| p3_mobilenet_v1 | TF.js float32 | 3267814 | 13.03 |  |  |  |  |
| p3_mobilenet_v1 | TF.js uint8 weights | 3267814 | 3.32 |  |  |  |  |
| p3_mobilenet_v2 | Keras (float32) | 2306662 | 9.82 | 7.47 | 7.80 | 97.72 | 100.00 |
| p3_mobilenet_v2 | TFLite float32 | 2306662 | 9.05 | 2.67 | 2.71 | 97.72 | 100.00 |
| p3_mobilenet_v2 | TFLite dynamic-range | 2306662 | 2.55 | 3.30 | 3.37 | 97.32 | 99.33 |
| p3_mobilenet_v2 | TFLite INT8 | 2306662 | 2.76 | 1.34 | 1.36 | 95.58 | 97.05 |
| p3_mobilenet_v2 | TF.js float32 | 2306662 | 9.18 |  |  |  |  |
| p3_mobilenet_v2 | TF.js uint8 weights | 2306662 | 2.41 |  |  |  |  |
