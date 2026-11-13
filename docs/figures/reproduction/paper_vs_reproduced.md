| model | schedule | seed | metric | paper | reproduced | difference |
|---|---|---|---|---|---|---|
| cnn | fullepochs | 42 | train accuracy (last epoch, as Keras reports) | 97.11 | 92.49 | -4.62 pp |
| cnn | fullepochs | 42 | train loss (last epoch, as Keras reports) | 0.0898 | 0.2170 | +0.1272 |
| cnn | fullepochs | 42 | validation accuracy (last epoch, as Keras reports) | 90.25 | 84.58 | -5.67 pp |
| cnn | fullepochs | 42 | validation loss (last epoch, as Keras reports) | 0.3605 | 0.5527 | +0.1922 |
| cnn | fullepochs | 42 | validation accuracy, full 17,572 images (argmax) | – | 84.58 | – |
| cnn | fullepochs | 42 | validation macro-F1, full set | – | 84.27 | – |
| cnn | steps100 | 42 | train accuracy (last epoch, as Keras reports) | 97.11 | 52.91 | -44.20 pp |
| cnn | steps100 | 42 | train loss (last epoch, as Keras reports) | 0.0898 | 1.5226 | +1.4328 |
| cnn | steps100 | 42 | validation accuracy (last epoch, as Keras reports) | 90.25 | 45.67 | -44.58 pp |
| cnn | steps100 | 42 | validation loss (last epoch, as Keras reports) | 0.3605 | 2.0493 | +1.6888 |
| cnn | steps100 | 42 | validation accuracy, full 17,572 images (argmax) | – | 45.17 | – |
| cnn | steps100 | 42 | validation macro-F1, full set | – | 43.57 | – |
| mobilenet | fullepochs | 42 | train accuracy (last epoch, as Keras reports) | 93.48 | 92.15 | -1.33 pp |
| mobilenet | fullepochs | 42 | train loss (last epoch, as Keras reports) | 0.2060 | 0.2494 | +0.0434 |
| mobilenet | fullepochs | 42 | validation accuracy (last epoch, as Keras reports) | 95.47 | 94.88 | -0.59 pp |
| mobilenet | fullepochs | 42 | validation loss (last epoch, as Keras reports) | 0.1419 | 0.1537 | +0.0118 |
| mobilenet | fullepochs | 42 | validation accuracy, full 17,572 images (argmax) | 95.00 | 94.88 | – |
| mobilenet | fullepochs | 42 | validation macro-F1, full set | 95.00 | 94.89 | – |
| mobilenet | steps100 | 42 | train accuracy (last epoch, as Keras reports) | 93.48 | 87.66 | -5.82 pp |
| mobilenet | steps100 | 42 | train loss (last epoch, as Keras reports) | 0.2060 | 0.3627 | +0.1567 |
| mobilenet | steps100 | 42 | validation accuracy (last epoch, as Keras reports) | 95.47 | 91.37 | -4.10 pp |
| mobilenet | steps100 | 42 | validation loss (last epoch, as Keras reports) | 0.1419 | 0.2648 | +0.1229 |
| mobilenet | steps100 | 42 | validation accuracy, full 17,572 images (argmax) | 95.00 | 91.62 | – |
| mobilenet | steps100 | 42 | validation macro-F1, full set | 95.00 | 91.54 | – |
