# k-means clustering

Training/Testing a k-means clustering algorithm to automatically cluster blobs in an image based on their color.

> The k-means algorithm is a type of unsupervised learning algorithm (i.e. no label/category information associated with the images/feature vectors) that automatically forms clusters of similar 'things'.

<div align="center">
<img src=misc/centroid.png?raw=true "centroid" width=540 >
</div>

---

## Simple Theory

- The k-means algorithm is used to find 'k' clusters in a dataset, where the number of clusters 'k' is a user supplied value.
- Each cluster is represented by a single data point called the 'centroid'.
- The centroid is defined as the mean (average) of all data points belonging to the cluster and is thus simply the center of the cluster.

> Centroid 'X' is the center of 'k=3' clusters (red, green, blue) as seen from the image above.

## The k-means process

- Step 1: We start off by selecting 'ki' random data points from our dataset — these 'ki' are our initial centroids.
- Step 2: Assign each data point 'k' in the dataset to the nearest centroid.

> So we compute the distance from each 'k' to each centroid (using a distance metric such as the Euclidean distance) and assigning the 'k' to the cluster with the smallest distance.

- Step 3: Recalculate the position of all centroids by computing the average of all data points in the cluster.
- Step 4: Repeat Steps 2 and 3 until all cluster assignments are stable or some stopping criterion has been met (such as a maximum number of iterations).

---

## Inference

- Let us apply k-means by generating a canvas of blobs, where each blob is a shade of red, green, or blue (left image):

Generated blobs         |  After thresholding
:-------------------------:|:-------------------------:
![Blob](misc/blob.png?raw=true "blob")  |  ![After](misc/after.png?raw=true "after")

- We will cluster these RGB blobs together (```lines 1-32 in 'run.py' handle this```) and then separate from each other.
- After generating the blobs, we need to detect each one of them and extract color features to characterize each of them. Thus, we perform thresholding.

> Thresholding leaves us with binary circles image of the blobs (as seen from right image above).

- We then detect contours of these circles (to find and access them), store features and extract the average RGB values — to characterize the color of the circle.
- Now it's time to train and test our k-means to perform the clustering of colors: ```python run.py```.
- Results: We can see that our blobs have been automatically separated into the shades/clusters of red, green, and blue.

> Here, we can see the shades of red:

<div align="center">
<img src=misc/red.png?raw=true "result" width=480 >
</div>

> Shades of green and blue:

Green         |  Blue
:-------------------------:|:-------------------------:
![Green](misc/green.png?raw=true "green")  |  ![Blue](misc/blue.png?raw=true "blue")


## References

- Sklearn implementation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- Contours: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
- For In-depth Theory: https://www.pyimagesearch.com/pyimagesearch-gurus/ & https://www.coursera.org/learn/machine-learning

---

saimj7/ 10-11-2020 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
