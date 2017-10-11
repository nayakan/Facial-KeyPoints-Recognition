# Facial KeyPoints Recognition
Facial Key Points Recognition Data Science Competition 
The objective of the project is to determine the locations of key features on a face. 

Objective is to develop a model that predicts positions of 15 key facial points - such as nose tip, centres and corners of eyes, mouth, lips, eyebrows, etc. - given images of faces,
assuming the images are head-shots of the person and are in gray-scale. 

For this project, R and Python was used. Python libraries include pandas, matplotlib, Theano and Lasagne;R libraries include ggplot, hmisc, IM.

The following is the list of the purpose of each script:

1.img_extract.py : Convert each image matrix into png image files.

2.preprocess_missing.py : Handle missing value.

3.ColMeansMethod.R : Implement the basic Column Means model.

4.PCA_script.R : Implement PCA on Training data: 7049 images.

5.MeanPatchScript.R :Implement the basic mean patch searching algorithm with
                      -a) Image flipping
                      -b) Contrast Stretching/Histogram Stretching
                      -c) Histogram equalization
                      -d) Gaussian Blurring
                      
6.model1_NeuralNetwork.py :Python script for Neural Network.

7.model2_CNN.py :Python script for Convolutional Neural Network.

8.model3_CNN.py :Python script for Convolutional Neural Networks with Image flipping.

![screen shot 2017-10-11 at 09 11 04](https://user-images.githubusercontent.com/17459420/31452566-24989c1e-ae64-11e7-81c2-de73b0c5e028.png)

               
