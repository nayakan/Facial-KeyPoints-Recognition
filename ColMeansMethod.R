#create variables to store the path to the files
data.dir <- '~/Documents/Notes/PatternRecDataMining/Project_Facial_KeyPoint/Data/'
train.file <- paste0(data.dir, 'training.csv')
test.file <- paste0(data.dir, 'test.csv')

# Read training data
d.train <- read.csv(train.file, stringsAsFactors = F)

# Move imagecolumn into a different frame
im.train <- d.train$Image
d.train$Image <- NULL

# load library
library(doMC)
registerDoMC()
#Convert strings to integers by splitting them and converting the result to integer. 
#Run parallelization using doMC library.
im.train <- foreach(im = im.train, .combine=rbind) %dopar% {
  as.integer(unlist(strsplit(im, " ")))
}
# Image matrix with 7049 rows for each image and 9216 columns for each pixel.
str(im.train)

# Read test data
d.test <- read.csv(test.file, stringsAsFactors = F)
im.test <- foreach(im = d.test$Image, .combine=rbind) %dopar% {
  as.integer(unlist(strsplit(im, " ")))
}
d.test$Image <- NULL

# Save the modified data files for later retrieval
save(d.train, im.train, d.test, im.test, file = "data.Rd")

# To view the image, convert 9216 integers into 96 X 96 matrix.
# im.train[1,] returns the first row of im.train, which corresponds 
# to the first training image. rev reverse the resulting vector to 
# match the interpretation of R's image function (which expects the 
# origin to be in the lower left corner)
im <- matrix(data = rev(im.train[1,]), nrow = 96, ncol = 96)

# To visualize the image
image(1:96, 1:96, im, col = gray((0:255)/255))

# To add keypoints
# Nose tip
points(96 - d.train$nose_tip_x[1], 96 - d.train$nose_tip_y[1], col = "red")
# Left eye center
points(96 - d.train$left_eye_center_x[1], 96 - d.train$left_eye_center_y[1], col = "blue")
# Right eye center
points(96 - d.train$right_eye_center_x[1], 96 - d.train$right_eye_center_y[1], col = "green")

# To check the variability of nose points in the training data
for(i in 1:nrow(d.train)){
  points(96 - d.train$nose_tip_x[i], 96 - d.train$nose_tip_y[i], col = "red")
}

# Check one of the outliers
idx <- which.max(d.train$nose_tip_x)
im <- matrix(data = rev(im.train[idx,]), nrow = 96, ncol = 96)
image(1:96, 1:96, im, col = gray((0:255)/255))
points(96 - d.train$nose_tip_x[idx], 96 - d.train$nose_tip_y[idx], col = "red")

# Simple benchmark is to compute the mean of the cooridnates of each keypoint 
# in the training set and use that as the prediction for all images.
# Computing mean for all columns ignoring missing values
colMeans(d.train, na.rm = T)

# Build a submission file
p <- matrix(data = colMeans(d.train, na.rm = T), nrow = nrow(d.test), ncol = ncol(d.train), byrow = T)
colnames(p) <- names(d.train)
predictions <- data.frame(ImageId = 1:nrow(d.test), p)

# Expected submission format has one key point per row.
install.packages('reshape2')
library(reshape2)
submission <- melt(predictions, id.vars = "ImageId", variable.name = "FeatureName", 
                   value.name = "Location")

# Join this ssubmission file with sample submission file to preserve the same order of entries
example.submission <- read.csv(paste0(data.dir, 'IdLookupTable.csv'))
sub.col.names      <- names(example.submission)
example.submission$Location <- NULL
submission <- merge(example.submission, submission, all.x = T, sort = F)
submission <- submission[, sub.col.names]
write.csv(submission, file="submission_means.csv", quote=F, row.names=F)
SampleSubmission <- cbind(submission[,1], submission[,4])
write.csv(SampleSubmission, file="SampleSubmission.csv", quote=F, row.names=F)








