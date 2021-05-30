# Plant Classification
## Using Color Histograms and Random Forests

We will extract color histograms (RGB) from flower-17 dataset and train a Random Forest classifier to predict flower species:

<div align="center">
<img src=mylib/misc/dataset.png?raw=true "dataset" width=520 >
</div>

---

## Simple Theory

- Masks (binary) were extracted from the dataset i.e., only pixels associated with the masked region will be used in constructing the color histogram.
- This allows us to describe only the petals of the image while simultaneously ignoring the background and other clutter that would otherwise distort the resulting feature vector and insert unwanted noise.

## Inference

- To extract color histogram features and train an RF on top of them: ```python run.py --images dataset/images --masks dataset/masks```.
- Results:

```
precision    recall  f1-score   support
crocus       0.92      1.00      0.96        12
daisy       0.88      0.93      0.90        15
pansy       1.00      0.85      0.92        20
sunflower       0.96      1.00      0.98        24
avg / total       0.95      0.94      0.94        71
```
> Our classifier was able to obtain an impressive 95% accuracy!

## References

- Dataset: https://www.robots.ox.ac.uk/~vgg/data/flowers/17/
- Histograms (OpenCV): https://docs.opencv.org/master/d1/db7/tutorial_py_histogram_begins.html

---

saimj7/ 30-05-2021 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
