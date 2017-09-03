# Semantic Segmentation

### Video Results
https://youtu.be/uvFBhRQCAcw

### Introduction
The goal in this project was to label the pixels in images using a Fully Convolutional Network (FCN), we used a pre-trained VGG 16 in this case. The process for building the FCN involved using 1x1 convolutions with tranpose convolutions to up-sample the image, also 2 skip layers were used to help give the FCN finer resolution. 

Once the FCN could effectively label road pixels from non-road pixels the network was modified to handle 4 different classes, this code was modified in helper.py. Also a data set was downloaded that included a much larger variety of labels. This dataset was only 208 images however and 192 of those were used to train. Still waiting to get access to the much larger city scapes dataset, it will be interesting what the results look like after training on that.

The main.py file was also modified so that it could process videos using a technique very similar to what was done in Term 1 with Advance Lane Finding. 4 different labels were used, which included road(green), car(blue), sidewalk(yellow), and other(no color). Important consideration for this project included initalizing the weights so that a regularizer was used to avoid over fitting. Also a good GPU was important for training large amounts of epochs, the results in the video used 110 epochs and on a Nvidia 1080 GTX and it only took about 40 minutes. 

### Conclusion

The FCN was able to generate nice results and mostly correctly classifed cars, and road, and even sidewalks. The network still needs work though, and should be trained on a much larger dataset. Only 192 images for training is barely scratching the surface, so it will be great to see how it performs once a larger dataset is aquired. 

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder
