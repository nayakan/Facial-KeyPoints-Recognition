import numpy as np
import pandas as pd
from pandas import Series,DataFrame

import matplotlib.pyplot as plt 

facial_recognition_df=pd.read_csv("training.csv")

#has 15 values. 
all_features=pd.DataFrame(facial_recognition_df[:2285]) #0-2284

#has 4 values. 
some_features=pd.DataFrame(facial_recognition_df[2285:]) #2285-

#left_eye_center_x
#filled na with median. mean~=median
plt.hist(all_features['left_eye_center_x'].dropna(), bins=100)
plt.boxplot(all_features['left_eye_center_x'].dropna())
all_features['left_eye_center_x'].fillna(all_features['left_eye_center_x'].median(), inplace=True)

#left_eye_center_y
#filled na with median. mean~=median
plt.hist(all_features['left_eye_center_y'].dropna(), bins=100)
plt.boxplot(all_features['left_eye_center_y'].dropna())
all_features['left_eye_center_y'].fillna(all_features['left_eye_center_y'].median(), inplace=True)

#right_eye_center_x
#filled na with median. mean~=median
plt.hist(all_features['right_eye_center_x'].dropna(), bins=100)
all_features['right_eye_center_x'].fillna(all_features['right_eye_center_x'].median(), inplace=True)

#right_eye_center_y
#filled na with median. mean~=median
plt.hist(all_features['right_eye_center_y'].dropna(), bins=100)
all_features['right_eye_center_y'].fillna(all_features['right_eye_center_y'].median(), inplace=True)

#left_eye_inner_corner_x
#filled na with median. mean~=median
plt.hist(all_features['left_eye_inner_corner_x'].dropna(), bins=100)
all_features['left_eye_inner_corner_x'].fillna(all_features['left_eye_inner_corner_x'].median(), inplace=True)

#left_eye_inner_corner_y
#filled na with median. mean~=median
plt.hist(all_features['left_eye_inner_corner_y'].dropna(), bins=100)
all_features['left_eye_inner_corner_y'].fillna(all_features['left_eye_inner_corner_y'].median(), inplace=True)

#left_eye_outer_corner_x
#filled na with median. mean~=median
plt.hist(all_features['left_eye_outer_corner_x'].dropna(), bins=100)
all_features['left_eye_outer_corner_x'].fillna(all_features['left_eye_outer_corner_x'].median(), inplace=True)

#left_eye_outer_corner_y
#filled na with median. mean~=median
plt.hist(all_features['left_eye_outer_corner_y'].dropna(), bins=100)
all_features['left_eye_outer_corner_y'].fillna(all_features['left_eye_outer_corner_y'].median(), inplace=True)

#'right_eye_inner_corner_x'
#filled na with median. mean~=median
plt.hist(all_features['right_eye_inner_corner_x'].dropna(), bins=100)
all_features['right_eye_inner_corner_x'].fillna(all_features['right_eye_inner_corner_x'].median(), inplace=True)

#'right_eye_inner_corner_y'
#filled na with median. mean~=median
plt.hist(all_features['right_eye_inner_corner_y'].dropna(), bins=100)
all_features['right_eye_inner_corner_y'].fillna(all_features['right_eye_inner_corner_y'].median(), inplace=True)

#'right_eye_outer_corner_x'
#filled na with median. mean~=median
plt.hist(all_features['right_eye_outer_corner_x'].dropna(), bins=100)
all_features['right_eye_outer_corner_x'].fillna(all_features['right_eye_outer_corner_x'].median(), inplace=True)

#'right_eye_outer_corner_y'
#filled na with median. mean~=median
plt.hist(all_features['right_eye_outer_corner_y'].dropna(), bins=100)
all_features['right_eye_outer_corner_y'].fillna(all_features['right_eye_outer_corner_y'].median(), inplace=True)

#'left_eyebrow_inner_end_x'
#filled na with median. mean~=median
plt.hist(all_features['left_eyebrow_inner_end_x'].dropna(), bins=100)
all_features['left_eyebrow_inner_end_x'].fillna(all_features['left_eyebrow_inner_end_x'].median(), inplace=True)

#'left_eyebrow_inner_end_y'
#filled na with median. mean~=median
plt.hist(all_features['left_eyebrow_inner_end_y'].dropna(), bins=100)
all_features['left_eyebrow_inner_end_y'].fillna(all_features['left_eyebrow_inner_end_y'].median(), inplace=True)

#'right_eyebrow_inner_end_x'
#filled na with median. mean~=median
plt.hist(all_features['right_eyebrow_inner_end_x'].dropna(), bins=100)
all_features['right_eyebrow_inner_end_x'].fillna(all_features['right_eyebrow_inner_end_x'].median(), inplace=True)

#'right_eyebrow_inner_end_y'
#filled na with median. mean~=median
plt.hist(all_features['right_eyebrow_inner_end_y'].dropna(), bins=100)
all_features['right_eyebrow_inner_end_y'].fillna(all_features['right_eyebrow_inner_end_y'].median(), inplace=True)

#'mouth_left_corner_x'
#filled na with median. mean~=median
plt.hist(all_features['mouth_left_corner_x'].dropna(), bins=100)
all_features['mouth_left_corner_x'].fillna(all_features['mouth_left_corner_x'].median(), inplace=True)

#'mouth_left_corner_y'
#filled na with median. mean~=median
plt.hist(all_features['mouth_left_corner_y'].dropna(), bins=100)
all_features['mouth_left_corner_y'].fillna(all_features['mouth_left_corner_y'].median(), inplace=True)

#'mouth_right_corner_x'
#filled na with median. mean~=median
plt.hist(all_features['mouth_right_corner_x'].dropna(), bins=100)
all_features['mouth_right_corner_x'].fillna(all_features['mouth_right_corner_x'].median(), inplace=True)

#'mouth_right_corner_y'
#filled na with median. mean~=median
plt.hist(all_features['mouth_right_corner_y'].dropna(), bins=100)
all_features['mouth_right_corner_y'].fillna(all_features['mouth_right_corner_y'].median(), inplace=True)

#'mouth_center_top_lip_x'
#filled na with median. mean~=median
plt.hist(all_features['mouth_center_top_lip_x'].dropna(), bins=100)
all_features['mouth_center_top_lip_x'].fillna(all_features['mouth_center_top_lip_x'].median(), inplace=True)

#'mouth_center_top_lip_y'
#filled na with median. mean~=median
plt.hist(all_features['mouth_center_top_lip_y'].dropna(), bins=100)
all_features['mouth_center_top_lip_y'].fillna(all_features['mouth_center_top_lip_y'].median(), inplace=True)

#'mouth_center_bottom_lip_x'
#filled na with median. mean~=median
plt.hist(all_features['mouth_center_bottom_lip_x'].dropna(), bins=100)
all_features['mouth_center_bottom_lip_x'].fillna(all_features['mouth_center_bottom_lip_x'].median(), inplace=True)

#'mouth_center_bottom_lip_y'
#filled na with median. mean~=median
plt.hist(all_features['mouth_center_bottom_lip_y'].dropna(), bins=100)
all_features['mouth_center_bottom_lip_y'].fillna(all_features['mouth_center_bottom_lip_y'].median(), inplace=True)

#'left_eyebrow_outer_end_x'
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original values')
axis2.set_title('New values')

all_features['left_eyebrow_outer_end_x'].dropna().astype(int).hist(bins=100, ax=axis1)

average_1 = all_features["left_eyebrow_outer_end_x"].mean()
std_1 = all_features["left_eyebrow_outer_end_x"].std()
count_1 = len(all_features)- all_features["left_eyebrow_outer_end_x"].count()
rand_1 = np.random.randint(average_1 - std_1, average_1 + std_1, size = count_1)
all_features["left_eyebrow_outer_end_x"][np.isnan(all_features["left_eyebrow_outer_end_x"])] = rand_1

all_features['left_eyebrow_outer_end_x'].hist(bins=70, ax=axis2)

#'left_eyebrow_outer_end_y',
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original values')
axis2.set_title('New values')

all_features['left_eyebrow_outer_end_y'].dropna().astype(int).hist(bins=100, ax=axis1)

average_1 = all_features["left_eyebrow_outer_end_y"].mean()
std_1 = all_features["left_eyebrow_outer_end_y"].std()
count_1 = len(all_features)- all_features["left_eyebrow_outer_end_y"].count()
rand_1 = np.random.randint(average_1 - std_1, average_1 + std_1, size = count_1)
all_features["left_eyebrow_outer_end_y"][np.isnan(all_features["left_eyebrow_outer_end_y"])] = rand_1

all_features['left_eyebrow_outer_end_y'].hist(bins=70, ax=axis2)


#'right_eyebrow_outer_end_x'
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original values')
axis2.set_title('New values')

all_features['right_eyebrow_outer_end_x'].dropna().astype(int).hist(bins=100, ax=axis1)

average_1 = all_features["right_eyebrow_outer_end_x"].mean()
std_1 = all_features["right_eyebrow_outer_end_x"].std()
count_1 = len(all_features)- all_features["right_eyebrow_outer_end_x"].count()
rand_1 = np.random.randint(average_1 - std_1, average_1 + std_1, size = count_1)
all_features["right_eyebrow_outer_end_x"][np.isnan(all_features["right_eyebrow_outer_end_x"])] = rand_1

all_features['right_eyebrow_outer_end_x'].hist(bins=70, ax=axis2)


#'right_eyebrow_outer_end_y',
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original values')
axis2.set_title('New values')

all_features['right_eyebrow_outer_end_y'].dropna().astype(int).hist(bins=100, ax=axis1)

average_1 = all_features["right_eyebrow_outer_end_y"].mean()
std_1 = all_features["right_eyebrow_outer_end_y"].std()
count_1 = len(all_features)- all_features["right_eyebrow_outer_end_y"].count()
rand_1 = np.random.randint(average_1 - std_1, average_1 + std_1, size = count_1)
all_features["right_eyebrow_outer_end_y"][np.isnan(all_features["right_eyebrow_outer_end_y"])] = rand_1

all_features['right_eyebrow_outer_end_y'].hist(bins=70, ax=axis2)


all_features.to_csv("cleaned_training_all_features.csv")
