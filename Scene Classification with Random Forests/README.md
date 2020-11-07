# Random Forests (RF)

Training/Testing a RF classifier in terms of the theory and image classification.

> We will be classifying a 4 scene dataset consisting of coasts, forests, highways and streets images:


<div align="center">
<img src=misc/dataset.png?raw=true "dataset" width=520 >
</div>

---

## Simple Theory

- RF consists of multiple decision trees aggregated together (ensemble classification method). Each decision tree votes on what it thinks the final classification is (k1, k2,...kg).
- These votes are tabulated by the meta-classifier, and the category with the most votes is chosen as the final classification (k):

<div align="center">
<img src=misc/rf.png?raw=true "rf" width=550 >
</div>


## Jensen’s Inequality
- The formal definition of Jensen’s Inequality states that the convex combined (average) ensemble will have error less than or equal to the average error of the individual models.
- To put it simply, imagine 10 people are to guess a football game win. The approach where you can improve your own prediction accuracy is by averaging all of the 10 guesses, ensuring you predict/guess no worse than the individual 10 people.

> The reason RF ensemble method works is because of Jensen's Inequality.

## Injecting Randomness
## Bootstrapping:

- RF train each individual decision tree on a bootstrapped sample (```sampling with replacement D times```) from the original training data. 'D' is normally the no. of training points in our training set.
- Consider the following 10-d feature vector before and after Bootstrapping:

Before         |  After
:-------------------------:|:-------------------------:
![Before](misc/before.png?raw=true "before")  |  ![After](misc/after.png?raw=true "after")

> Randomness is typically applied to improve the accuracy of ML algorithms while reducing the risk of overfitting.

## Node Splits:
- After performing row wise sampling, we move on to the column wise sampling.
- The node split criterion such as [**Information gain**](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees) or the [**Gini ratio/coefficient**](https://en.wikipedia.org/wiki/Gini_coefficient) attempts to find the optimal split (with the most 'information') at each node in the decision tree by examining each column of the feature vector.
- In practice, we normally take the square-root or the log of the dimensionality of the feature vector.
- The end result is a sampled feature vector with both replaced rows and columns.

---

## Inference

- First up, we extract the HSV color channel statistics/feature vectors from the images.
- Then, we extract texture features using 'Haralick texture features' (```pip install mahotas```) and concatenate both the color and texture features into a 'single' feature vector.

> Why? Because we can see that color alone is not enough to discriminate between the four classes in our dataset and that texture adds up as a better feature for good accuracy.

- Each class exhibits considerable variation in color distribution (hard to rule out solely based on color), and the texture/patterns of each class are somewhat similar (a bonus parameter to consider). Thus, we prefer the combined approach.

- To train and test a RF classifier: ```python run.py --dataset 4scenes --forest 1```. Results:

```
[INFO] evaluating...
                 precision    recall  f1-score   support

  4scenes\coast       0.83      0.85      0.84        95
 4scenes\forest       0.92      0.98      0.95        81
4scenes\highway       0.84      0.64      0.72        66
 4scenes\street       0.79      0.88      0.84        69

       accuracy                           0.85       311
      macro avg       0.84      0.84      0.84       311
   weighted avg       0.85      0.85      0.84       311
```

- To train and test a Decision Tree: ```python run.py --dataset 4scenes```.

> Final accuracy is 85%, an improvement over the [**DecisionTreeClassifier**](https://github.com/saimj7/Machine-Learning-Algorithms/tree/main/Scene%20Classification%20with%20Decision%20Trees) (79%).

<div align="center">
<img src=misc/result.png?raw=true "result" width=800 >
</div>

## References

- Random Forests Paper: https://link.springer.com/article/10.1023/A:1010933404324
- Sklearn implementation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Haralick texture features: https://mahotas.readthedocs.io/en/latest/features.html
- For In-depth Theory: https://www.pyimagesearch.com/pyimagesearch-gurus/ & https://www.coursera.org/learn/machine-learning

---

saimj7/ 07-11-2020 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
