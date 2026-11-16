| model | format | params | size_mb | cpu_latency_ms_median | cpu_latency_ms_p90 | accuracy | agreement_with_keras | xnnpack |
|---|---|---|---|---|---|---|---|---|
| p3_abl_mnv1_alpha050 | Keras (float32) | 849030 | 3.72 | 3.85 | 4.04 | 98.39 | 100.00 | nan |
| p3_abl_mnv1_alpha050 | TFLite float32 | 849030 | 3.35 | 1.33 | 1.35 | 98.39 | 100.00 | True |
| p3_abl_mnv1_alpha050 | TFLite dynamic-range | 849030 | 0.94 | 1.34 | 1.41 | 98.26 | 99.87 | True |
| p3_abl_mnv1_alpha050 | TFLite INT8 | 849030 | 1.00 | 0.66 | 0.69 | 98.12 | 99.46 | True |
| p3_abl_mnv1_alpha050 | TF.js float32 | 849030 | 3.41 |  |  |  |  | nan |
| p3_abl_mnv1_alpha050 | TF.js uint8 weights | 849030 | 0.91 |  |  |  |  | nan |
| p3_abl_mnv1_alpha075 | Keras (float32) | 1862198 | 7.78 | 5.46 | 5.90 | 99.20 | 100.00 | nan |
| p3_abl_mnv1_alpha075 | TFLite float32 | 1862198 | 7.37 | 2.43 | 2.50 | 99.20 | 100.00 | True |
| p3_abl_mnv1_alpha075 | TFLite dynamic-range | 1862198 | 1.98 | 2.05 | 2.12 | 98.79 | 99.60 | True |
| p3_abl_mnv1_alpha075 | TFLite INT8 | 1862198 | 2.08 | 1.03 | 1.04 | 99.06 | 99.60 | True |
| p3_abl_mnv1_alpha075 | TF.js float32 | 1862198 | 7.43 |  |  |  |  | nan |
| p3_abl_mnv1_alpha075 | TF.js uint8 weights | 1862198 | 1.92 |  |  |  |  | nan |
| p3_abl_mnv1_frozen | Keras (float32) | 3267814 | 13.40 | 7.16 | 7.24 | 95.71 | 100.00 | nan |
| p3_abl_mnv1_frozen | TFLite float32 | 3267814 | 12.96 | 3.90 | 3.92 | 95.71 | 100.00 | True |
| p3_abl_mnv1_frozen | TFLite dynamic-range | 3267814 | 3.42 | 2.86 | 2.98 | 95.04 | 98.93 | True |
| p3_abl_mnv1_frozen | TFLite INT8 | 3267814 | 3.55 | 1.39 | 1.45 | 94.50 | 96.92 | True |
| p3_abl_mnv1_frozen | TF.js float32 | 3267814 | 13.03 |  |  |  |  | nan |
| p3_abl_mnv1_frozen | TF.js uint8 weights | 3267814 | 3.32 |  |  |  |  | nan |
| p3_cnn_softmax | Keras (float32) | 2426534 | 9.76 | 2.61 | 2.63 | 88.20 | 100.00 | nan |
| p3_cnn_softmax | TFLite float32 | 2426534 | 9.71 | 1.36 | 1.38 | 88.20 | 100.00 | True |
| p3_cnn_softmax | TFLite dynamic-range | 2426534 | 2.44 | 0.72 | 0.76 | 88.07 | 99.46 | True |
| p3_cnn_softmax | TFLite INT8 | 2426534 | 2.45 | 0.57 | 0.58 | 87.80 | 98.66 | True |
| p3_cnn_softmax | TF.js float32 | 2426534 | 9.72 |  |  |  |  | nan |
| p3_cnn_softmax | TF.js uint8 weights | 2426534 | 2.44 |  |  |  |  | nan |
| p3_efficientnet_b0 | Keras (float32) | 4098249 | 17.24 | 11.79 | 12.13 | 99.60 | 100.00 | nan |
| p3_efficientnet_b0 | TFLite float32 | 4098249 | 16.23 | 5.40 | 5.75 | 99.60 | 100.00 | True |
| p3_efficientnet_b0 | TFLite dynamic-range | 4098249 | 4.60 | 7.00 | 8.16 | 99.46 | 99.60 | True |
| p3_efficientnet_b0 | TFLite INT8 | 4098249 | 4.96 | 2.19 | 2.28 | 69.84 | 69.97 | True |
| p3_efficientnet_b0 | TF.js float32 | 4098249 | 16.48 |  |  |  |  | nan |
| p3_efficientnet_b0 | TF.js uint8 weights | 4098249 | 4.37 |  |  |  |  | nan |
| p3_mobilenet_v1 | Keras (float32) | 3267814 | 13.40 | 7.34 | 7.56 | 99.06 | 100.00 | nan |
| p3_mobilenet_v1 | TFLite float32 | 3267814 | 12.96 | 3.92 | 4.48 | 99.06 | 100.00 | True |
| p3_mobilenet_v1 | TFLite dynamic-range | 3267814 | 3.42 | 2.90 | 3.71 | 99.06 | 99.87 | True |
| p3_mobilenet_v1 | TFLite INT8 | 3267814 | 3.55 | 1.39 | 1.41 | 98.53 | 99.20 | True |
| p3_mobilenet_v1 | TF.js float32 | 3267814 | 13.03 |  |  |  |  | nan |
| p3_mobilenet_v1 | TF.js uint8 weights | 3267814 | 3.32 |  |  |  |  | nan |
| p3_mobilenet_v2 | Keras (float32) | 2306662 | 9.82 | 7.29 | 7.69 | 97.72 | 100.00 | nan |
| p3_mobilenet_v2 | TFLite float32 | 2306662 | 9.05 | 2.68 | 2.72 | 97.72 | 100.00 | True |
| p3_mobilenet_v2 | TFLite dynamic-range | 2306662 | 2.55 | 3.28 | 3.35 | 97.32 | 99.33 | True |
| p3_mobilenet_v2 | TFLite INT8 | 2306662 | 2.76 | 1.33 | 1.36 | 95.58 | 97.05 | True |
| p3_mobilenet_v2 | TF.js float32 | 2306662 | 9.18 |  |  |  |  | nan |
| p3_mobilenet_v2 | TF.js uint8 weights | 2306662 | 2.41 |  |  |  |  | nan |
| p3_mobilenet_v3_small | Keras (float32) | 961046 | 4.42 | 3.66 | 4.03 | 98.93 | 100.00 | nan |
| p3_mobilenet_v3_small | TFLite float32 | 961046 | 3.82 | 0.96 | 1.09 | 98.93 | 100.00 | True |
| p3_mobilenet_v3_small | TFLite dynamic-range | 961046 | 1.13 | 1.25 | 1.37 | 99.20 | 99.20 | True |
| p3_mobilenet_v3_small | TFLite INT8 | 961046 | 1.24 | 0.90 | 0.93 | 60.86 | 60.72 | False |
| p3_mobilenet_v3_small | TF.js float32 | 961046 | 3.97 |  |  |  |  | nan |
| p3_mobilenet_v3_small | TF.js uint8 weights | 961046 | 1.15 |  |  |  |  | nan |
