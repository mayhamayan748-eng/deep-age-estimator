---
title: Deep Age Estimator
emoji: 🎞️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Estimate age from a face photo and generate faces with a GAN
---

# Deep Age Estimator

Upload a photo and three neural networks trained on UTKFace estimate the age of every face in it
(EfficientNetB0: 4.97 years mean error on the test set). A conditional GAN generates faces for any requested age.

Graduation project by Ayham Ayan · source code: https://github.com/mayhamayan748-eng/deep-age-estimator

Photos are processed in memory and never stored.
