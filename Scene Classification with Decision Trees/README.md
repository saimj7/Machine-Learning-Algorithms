# Decision Trees

Training/Testing a decision tree in terms of the theory and image classification.

> We will be classifying a 4 scene dataset consisting of coasts, forests, highways and streets images:


<div align="center">
<img src=misc/dataset.png?raw=true "dataset" width=520 >
</div>

---

## Simple Theory

- The basic idea behind a decision tree is to break classification down into a set of choices about each entry (i.e. column) in our feature vector. We start at the root of the tree and then progress down to the leaves where the actual classification is made.
- For instance, we might solve our urge of going to the movies or a sunny beach by constructing the following simple tree:

<div align="center">
<img src=misc/tree.png?raw=true "tree" width=500 >
</div>


## Information Theory (IT)
- The roots of decision trees lie under Claude Shannon's Entropy (in the field of IT).
- Let’s pretend we have a set of positive (+1) and negative (-1) values for some arbitrary feature 'x'. If all the examples in 'x' are either positive or negative, we don't learn anything new.
- What if we have a mixture of both values? That implies 'x' is an useful feature because it is able to separate the values into 50% positive and 50% negative probabilities. Here, entropy is at its maximum.

> The entropy is at its maximum when both the probabilities are equal.

## Construction

- Decision tree construction is built upon 'decision/informative splits' using IT.
- To decide which feature to split on (root of the tree), we try every feature (i.e. column) and measure which split gives us the most 'information' — this is called our most informative split. It is normally the [**Information gain**](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees) or the [**Gini ratio/coefficient**](https://en.wikipedia.org/wiki/Gini_coefficient).
- Then the data is split into ```subsets > branches > leaf node/terminating blocks```. This process is repeated until we can classify all instances correctly.

---

## Inference

- First up, we extract the HSV color channel statistics/feature vectors from the images.
- Then, we extract texture features using 'Haralick texture features' (```pip install mahotas```) and concatenate both the color and texture features into a 'single' feature vector.

> Why? Because we can see that color alone is not enough to discriminate between the four classes in our dataset and that texture adds up as a better feature for good accuracy.

- Each class exhibits considerable variation in color distribution (hard to rule out solely based on color), and the texture/patterns of each class are somewhat similar (a bonus parameter to consider). Thus, we prefer the combined approach.

- To train and test a decision tree classifier: ```python run.py --dataset 4scenes```. Results:

```
[INFO] evaluating...
                 precision    recall  f1-score   support

  4scenes\coast       0.75      0.71      0.73        93
 4scenes\forest       0.92      0.87      0.89        87
4scenes\highway       0.68      0.71      0.70        63
 4scenes\street       0.77      0.84      0.80        67

       accuracy                           0.78       310
      macro avg       0.78      0.78      0.78       310
   weighted avg       0.79      0.78      0.78       310
```

> Final accuracy is 79%.

<div align="center">
<img src=misc/result.png?raw=true "result" width=800 >
</div>

## References

- Binary tree: https://en.wikipedia.org/wiki/Binary_tree
- Information Theory Paper (A mathematical theory of communication): https://ieeexplore.ieee.org/abstract/document/6773024
- Tree construction: https://engineering.purdue.edu/kak/Tutorials/DecisionTreeClassifiers.pdf
- Sklearn implementation: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
- Haralick texture features: https://mahotas.readthedocs.io/en/latest/features.html
- For In-depth Theory: https://www.pyimagesearch.com/pyimagesearch-gurus/ & https://www.coursera.org/learn/machine-learning

---

saimj7/ 06-11-2020 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
