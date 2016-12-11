# FacialKeyPointsRecognition
Kaggle Facial Key Points Recognition Competition 
The objective of the project is to determine the locations of key features on a face. 
The problem relates to the field of computer vision and deep learning.We aim to develop a model that predicts positions of 
15 key facial points - such as nose tip, centres and corners of eyes, mouth, lips, eyebrows, etc. - given images of faces,
assuming the images are head-shots of the person and are in gray-scale. 
For this project, R and Python was used. Python libraries include pandas, matplotlib, Theano and Lasagne.
While R libraries include ggplot, hmisc, IM.

The following is the list of the purpose of ech script:

1.img_extract.py : Python script can be use to convert each image matrix into png image files.
2.preprocess_missing.py :Python script to handle missing value.
3.ColMeansMethod.R : R script to implement the basic Column Means model.
4.PCA_script.R :R script to implement PCA on Training data: 7049 images.
5.MeanPatchScript.R :R script to implement the basic mean patch searching algorithm with
                      -a) Image flipping
                      -b) Contrast Stretching/Histogram Stretching
                      -c) Histogram equalization
                      -d) Gaussian Blurring
6.model1_NeuralNetwork.py : Python script for Neural Networks
7.model2_CNN.py :Python script for Convolutional Neural Networks
8.model3_CNN.py :Python script for Convolutional Neural Networks with Image flipping.
               
