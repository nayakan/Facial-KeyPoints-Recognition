## Load the lobraries
library(ggplot2)
library(Hmisc)
library(doMC)
registerDoMC()

## Load test and train data
test_data <- read.csv(file = file.choose(),stringsAsFactors = F)
str(test_data)
train_data <- read.csv(file = file.choose(),stringsAsFactors = F)
str(train_data)
nrow(train_data)
head(train_data,n=1)
head(test_data,n=1)
sample_submission <- read.csv(file = file.choose())
head(sample_submission,n=100)
train_data[,c(2)]
head(train_data$left_eye_center_x,n = 10)

############ Missing Values in the dataset #######################

############# Training Dataset
for(i in 1:ncol(train_data)){
  MissingValues <- (sum(is.na(train_final_data[,i])))
  print((c(names(train_final_data[i]),MissingValues)))
}

#### Dividing the dataset into 2 sub-datasets:
#### one with 4 key points and another with 15 keypoints

fifteen_dataset <- subset(train_data[1:2284,]) ## dividing the existing train data
str(fifteen_dataset)
head(fifteen_dataset)
fifteen_dataset<-read.csv(file = file.choose()) ## loading Preethi's clean data
fifteen_dataset<- fifteen_dataset[-c(1)]
train_final_data <- rbind(fifteen_dataset,four_dataset)

for(i in 1:ncol(fifteen_dataset)){
  MissingValues <- (sum(is.na(fifteen_dataset[,i])))
  print((c(names(fifteen_dataset[i]),MissingValues)))
}

## The dataset with four keypoints
four_dataset <- subset(train_data[2885:nrow(train_data),])
str(four_dataset)
head(four_dataset)
## Removing all the Null columns
four_dataset <- four_dataset[-c(5:20,23:28)]
## checking for Missing values in the dataset
for(i in 1:ncol(four_dataset)){
  MissingValues <- (sum(is.na(four_dataset[,i])))
  print((c(names(four_dataset[i]),MissingValues)))
}
## Imputing the missing values with mean of the data
four_dataset$mouth_center_bottom_lip_x<-impute(four_dataset$mouth_center_bottom_lip_x,mean)
four_dataset$mouth_center_bottom_lip_y<-impute(four_dataset$mouth_center_bottom_lip_y,mean)
## Checking the columns for missing values
sum(is.na(four_dataset$mouth_center_bottom_lip_x))
sum(is.na(four_dataset$mouth_center_bottom_lip_y))

########## Test dataset

for(i in 1:ncol(test_data)){
  MissingValues <- (sum(is.na(test_data[,i])))
  print((c(names(test_data[i]),MissingValues)))
}


################  Exploratory Data Analysis (EDA)  #############
boxplot(four_dataset$mouth_center_bottom_lip_x)
hist(four_dataset$mouth_center_bottom_lip_x)
boxplot(four_dataset$left_eye_center_x)
str(four_dataset)
nrow(four_dataset)



############### Principal Component Analysis (PCA) ###################

### Splitting the Image column into a matrix
image_dataset <- four_dataset$Image
image_final_dataset<- as.character(train_final_data$Image)
image_final_dataset <- foreach(im = image_final_dataset, .combine=rbind) %dopar% {
  as.integer(unlist(strsplit(im, " ")))
}
str(image_final_dataset)
### Display the image
im <-matrix(data=rev(image_final_dataset[2,]),nrow = 96,ncol = 96)
image(1:96, 1:96, im, col=gray((0:255)/255))

############# PCA
#### Training Dataset
dim(image_final_dataset)
image_dataset_transpose <- t(image_final_dataset)
dim(image_dataset_transpose)
apply(image_dataset_transpose,2,var)
pca = prcomp(image_dataset_transpose,scale. = T)
plot(pca,type = "l")
pca$rotation<--pca$rotation
pca$x<--pca$x
biplot (pca , scale =0)
std_dev<-pca$sdev
variance <- std_dev^2
variance[1:120]

##proportion of variance
prop_varex <- variance/sum(variance)
prop_varex[1:150]
plot(prop_varex, 
     xlab="Principal Component",
     ylab="Proportion of Variance Explained",
     type = "b")

#cumulative scree plot
 plot(cumsum(prop_varex), xlab = "Principal Component",
       ylab = "Cumulative Proportion of Variance Explained",
       type = "b")
 
 
 train_final_data <- data.frame(train_final_data,pca$x)
 new_data_transpose<-pca$x[,1:100]
 dim(new_data_transpose)
 new_data<-t(new_data_transpose)
 dim(new_data)
 
 im <-matrix(data=rev(new_data[1,]),nrow = 96,ncol = 96)
 image(1:96, 1:96, im, col=gray((0:255)/255))

 
 ###### Test dataset
 ##image_test_dataset <- as.character(test_data$Image)
 image_test_dataset <- foreach(im = test_data$Image, .combine=rbind) %dopar% {
   as.integer(unlist(strsplit(im, " ")))
 }
 str(image_test_dataset)
 
 ### Display the images after PCA
 im <-matrix(data=rev(image_test_dataset[1,]),nrow = 96,ncol = 96)
 image(1:96, 1:96, im, col=gray((0:255)/255))
 
 test_data<- subset(test_data,select = c("ImageId"))
 test_data <- merge(test_data,image_test_dataset)
 dim(image_test_dataset)
 image_test_transpose <- t(image_test_dataset)
 dim(image_test_transpose)
 ##apply(image_test_transpose,2,var)
 pca_test = prcomp(image_test_transpose,scale. = T)
 plot(pca_test,type = "l")
 pca_test$rotation<--pca_test$rotation
 pca_test$x<--pca_test$x
 biplot (pca_test , scale =0)
 std_dev<-pca_test$sdev
 variance <- std_dev^2
 variance[1:120]
 
 ##proportion of variance
 prop_varex <- variance/sum(variance)
 prop_varex[1:150]
 plot(prop_varex, 
      xlab="Principal Component",
      ylab="Proportion of Variance Explained",
      type = "b")
 
 #cumulative scree plot
 plot(cumsum(prop_varex), xlab = "Principal Component",
      ylab = "Cumulative Proportion of Variance Explained",
      type = "b")
 
 
 test_final_data <- data.frame(test_data,pca_test$x)
 test_new_data_transpose<-pca_test$x[,1:15]
 dim(test_new_data_transpose)
 test_new_data<-t(test_new_data_transpose)
 dim(test_new_data)
 
 ### Display the images after PCA
 im <-matrix(data=rev(test_new_data[2,]),nrow = 96,ncol = 96)
 image(1:96, 1:96, im, col=gray((0:255)/255))

#### Mean Patching
coord <- "left_eye_center"
patch_size <- 10
coord_x <- paste(coord, "x", sep="_")
coord_y <- paste(coord, "y", sep="_")
patches <- foreach (i = 1:nrow(train_data), .combine=rbind) %do% {
  im  <- matrix(data = im.train[i,], nrow=96, ncol=96)
  x   <- d.train[i, coord_x]
  y   <- d.train[i, coord_y]
  x1  <- (x-patch_size)
  x2  <- (x+patch_size)
  y1  <- (y-patch_size)
  y2  <- (y+patch_size)
  if ( (!is.na(x)) && (!is.na(y)) && (x1>=1) && (x2<=96) && (y1>=1) && (y2<=96) )
  {
    as.vector(im[x1:x2, y1:y2])
  }
  else
  {
    NULL
  }
}
mean.patch <- matrix(data = colMeans(patches), nrow=2*patch_size+1, ncol=2*patch_size+1)


#### Outlier Detection #######

boxplot(four_dataset$left_eye_center_x)
boxplot(four_dataset$left_eye_center_y)
