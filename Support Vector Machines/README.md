# Support Vector Machines (SVMs)

Training/Testing a SVM classifier in terms of the theory. The paper 'Support Vector Networks' has over 20,600 citations and is one of the most cited papers in computer science history!

> We will be solving the XOR problem by utilizing kernels to achieve non-linear separation.


<div align="center">
<img src=misc/plots.png?raw=true "demo" width=600 >
</div>

---

## Simple Theory
## Linear Separability
- SVMs work on the principle of 'linear separability'. As seen from the image above, data points in the plots A & B can be linearly separable by drawing a straight line through them. Whereas plot C has a rather different grouping (the XOR problem).
- This line is often referred as separating hyperplane. In general, we are interested in finding a maximum margin hyperplane (i.e., the separable line is at its maximum distance between the two datapoint classes):

<div align="center">
<img src=misc/maxmargin.png?raw=true "demo" width=500 >
</div>

> The hyperplane is at its maximum when it is equidistant from the two classes of data, hence our hyperplane has a width of 2M.

- Having a max. margin gives us some flexibility when classifying testing data, mitigating the risk of accidentally misclassifying an instance.
---

## The Kernel Trick

- Is it possible to classify non-linear data, where the datapoints cannot be separated with a straight line (as in plot C)? Turns out we with the Kernel matrix.
- It is computed from the dot product of the original vectors (i.e. the pairwise similarity between all possible pairs of feature vectors).
- Assume we have 4 datapoints A,B,C,D. We then compute a feature vector A', simply a representation of how similar A is to itself and all other data points over applying some K function: ```A' = [K(A, A), K(A, B), K(A, C), K(A, D)]```.

> The classes in A' are now ideally linearly separable.

- Several types of Kernels exist (Linear, Polynomial, Sigmoid), but they must satisfy these [**properties**](http://web.iitd.ac.in/~sumeet/CLT2008S-lecture18.pdf).

## Inference

- To train and test a SVM in order to solve the XOR problem (plot C): ```python run.py```.

- With linear kernel:

```
[RESULTS] SVM w/ Linear Kernel
              precision    recall  f1-score   support

          -1       0.59      1.00      0.74        44
           1       1.00      0.45      0.62        56

    accuracy                           0.69       100
   macro avg       0.79      0.72      0.68       100
weighted avg       0.82      0.69      0.67       100
```

- With polynomial kernel:

```
[RESULTS] SVM w/ Polynomial Kernel
              precision    recall  f1-score   support

          -1       1.00      1.00      1.00        44
           1       1.00      1.00      1.00        56

    accuracy                           1.00       100
   macro avg       1.00      1.00      1.00       100
weighted avg       1.00      1.00      1.00       100
```
> Linear kernel (82% acc) and polynomial kernel (100% acc).

- By applying a polynomial kernel, our SVM was able to classify each data point without a problem.


## References

- SVM Introduction Paper (Kernel trick): https://dl.acm.org/doi/abs/10.1145/130385.130401
- Support Vector Networks Paper (Soft margins): https://link.springer.com/article/10.1007/BF00994018
- SVM and Kernels Implementation in Sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- XOR Problem: https://cs.nyu.edu/~mohri/mls/lecture_5.pdf
- In-depth Theory: https://www.coursera.org/learn/machine-learning

---

saimj7/ 05-11-2020 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
