# Deep Age Estimator

Human age estimation from facial images using a CNN regressor with GAN-based data augmentation, trained on the [UTKFace](https://susanqq.github.io/UTKFace/) dataset.

## Overview
- **Data:** UTKFace images (ages 0–90), resized to 128×128 grayscale, pixels normalized to [0, 1], ages normalized by 90. 80/20 train/test split.
- **Model:** 5-block CNN (Conv2D → BatchNorm → MaxPool, 32→512 filters) followed by Dense(128) + Dropout and a single regression output. Trained with MSE loss and Adam with exponential learning-rate decay.
- **Augmentation:** a DCGAN (generator + discriminator) is trained on the training images and 500 synthetic faces are added to the training set.

## Results
| Metric | Normalized | In years |
|---|---|---|
| Test MAE | 0.0697 | ~6.3 |
| Test MSE | 0.0087 | — |

## Usage
1. Download UTKFace and update `image_dir` in the notebook.
2. Install dependencies: `pip install tensorflow opencv-python scikit-learn matplotlib numpy`
3. Run `gradute_preoject.ipynb`. Use `predict_custom_image(cnn_model, "path/to/image.jpg")` to predict the age of a new image.

## Files
- `gradute_preoject.ipynb` — full pipeline (data loading, CNN, GAN, training, evaluation, inference)
- `التنبؤ بعمر االنسان باستخدام التعلم العميق.pdf` — project report
- `Screenshot *.png` — sample results
