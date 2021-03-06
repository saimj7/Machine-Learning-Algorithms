# Face recognition with logistic regression

Training/Testing a face recognition classifier with simple logistic regression (LR).

> We will be using a subset of 'Labeled Faces in the Wild' dataset:


<div align="left">
<img src=misc/dataset.png?raw=true "demo" width=600 >
</div>

---

## Simple Theory
- In general, LR performs classification using a sigmoid function (refer or run ```python plots.py```). It extracts the feature vectors from an image and multiplies them with a weight vector (w) prior to passing into the sigmoid function for classification.
- You might ask how do we compute the weights? Our goal is to find the values of 'w' that make our classifier as accurate as possible; so we will need to apply gradient ascent/descent.
- Gradient is the error on the training data w.r.t the feature vectors. Based on this error, we update the 'w' with a factor called learning rate.

> We do this 'n' times in a loop or until convergence. In other words, at each subsequent loop, our algorithm moves closer and closer to the optimal 'w' values.

---

## Inference:

- To train and test: ```python run.py```. Results:

```
                   precision    recall  f1-score   support

  Donald Rumsfeld       0.72      0.72      0.72        32
    George W Bush       0.87      0.90      0.88       129
Gerhard Schroeder       0.83      0.66      0.73        29
       Tony Blair       0.84      0.86      0.85        36

         accuracy                           0.84       226
        macro avg       0.81      0.78      0.80       226
     weighted avg       0.84      0.84      0.83       226
```

```
[PREDICTION] predicted: George W Bush, actual: George W Bush
[PREDICTION] predicted: George W Bush, actual: George W Bush
[PREDICTION] predicted: George W Bush, actual: Donald Rumsfeld
[PREDICTION] predicted: Tony Blair, actual: Gerhard Schroeder
[PREDICTION] predicted: Tony Blair, actual: Tony Blair
```
- 84% accuracy is not too bad. Note that we have only trained on the raw pixel intensities of images.


## References

- Logistic regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Labeled faces in the wild: http://vis-www.cs.umass.edu/lfw/
- Gradient descent: https://en.wikipedia.org/wiki/Gradient_descent
- In-depth theory: https://www.pyimagesearch.com/pyimagesearch-gurus/ & https://www.coursera.org/learn/machine-learning

---

saimj7/ 04-11-2020 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
