install.packages('doMC')
install.packages('IM')

library(reshape2)
library(doMC)
library(foreach)
library(IM)
library(spatstat)

registerDoMC()
# parameters
data.dir    <- '~/Desktop/Shilpita/COEN281_PatternRecognitionDataMining/Project/FacialKeyPoints/'
patch_size  <- 15 #11   #10
search_size <- 2

################ DATAFRAME INTIALIZATION #################################################
# read data and convert image strings to arrays
train.file <- paste0(data.dir, 'training.csv')
test.file  <- paste0(data.dir, 'test.csv')
data.file  <- paste0(data.dir, 'data.Rd')
d.train    <- read.csv(train.file, stringsAsFactors=F)
d.test     <- read.csv(test.file,  stringsAsFactors=F)

im.train   <- foreach(im = d.train$Image, .combine=rbind) %dopar% {
  as.integer(unlist(strsplit(im, " ")))
}

im.test    <- foreach(im = d.test$Image, .combine=rbind) %dopar% {
  as.integer(unlist(strsplit(im, " ")))
}

d.train$Image <- NULL
d.test$Image  <- NULL

# list the coordinates we have to predict
coordinate.names <- gsub("_x", "", names(d.train)[grep("_x", names(d.train))])

################ HISTOGRAM STRETCHING #################################################

#A non-vectorized function
myNormalizeFun <- function(x,y,M){
  # M[x,y] + 1
  a <- quantile(M,c(.05)) #min(M)   #quantile(M,c(.05))
  b <- quantile(M,c(.95)) #max(M)   #quantile(M,c(.95))
  u<- 255
  l<- 0
  ((u-l)/(b-a)) *(M[x,y]- a) +l
  #M[x,y] *2
}

myVecFun <- Vectorize(myNormalizeFun,vectorize.args = c('x','y'))
    #function(r,c) r+c +1 )

image.temp <- foreach(i = 1: nrow(im.train),.combine = rbind)%do%{
  im <- matrix(data = im.train[i,], nrow=96, ncol=96)
  as.vector(outer(1:nrow(im), 1:ncol(im) , myVecFun,im))
}

################ HISTOGRAM EQUALIZATION #################################################

image.histeq <- foreach(i = 1: nrow(im.train),.combine = rbind)%do%{
  histeq(im.train[i,])
}

################ GAUSSIAN BLURRING  #################################################

image.blur <-foreach(i = 1:nrow(image.histeq) ,.combine = rbind)%do%{ 
#image.blur <-foreach(i = 1:nrow(im.train) ,.combine = rbind)%do%{
    im <- matrix(data = image.histeq[i,], nrow=96, ncol=96)
    im.blur <- blur(as.im(im) , sigma = 2,bleed = TRUE)
    as.vector(im.blur$v)
}
  
################ FLIPPING IMAGE  #################################################

# Flip the im.train data set and store the flipped images in im.flip
image.flip <- foreach( i = 1:nrow(im.train), .combine=rbind) %do% {
  im <- matrix(data = (im.train[i,]), nrow = 96, ncol = 96)
  as.vector(apply(im, 2, rev))
}

d.train1 <- d.train[,c(3, 4, 1, 2, 9, 10, 11, 12,
                       5, 6, 7, 8, 17, 18, 19, 20,
                       13, 14, 15, 16, 21, 22, 25,
                       26, 23, 24, 27, 28, 29, 30)]
colnames(d.train1) <- c("left_eye_center_x", "left_eye_center_y", "right_eye_center_x",
                        "right_eye_center_y", "left_eye_inner_corner_x", "left_eye_inner_corner_y",
                        "left_eye_outer_corner_x", "left_eye_outer_corner_y", "right_eye_inner_corner_x",
                        "right_eye_inner_corner_y", "right_eye_outer_corner_x", "right_eye_outer_corner_y",
                        "left_eyebrow_inner_end_x", "left_eyebrow_inner_end_y", "left_eyebrow_outer_end_x",
                        "left_eyebrow_outer_end_y", "right_eyebrow_inner_end_x", "right_eyebrow_inner_end_y",
                        "right_eyebrow_outer_end_x", "right_eyebrow_outer_end_y", "nose_tip_x",
                        "nose_tip_y", "mouth_left_corner_x", "mouth_left_corner_y",
                        "mouth_right_corner_x", "mouth_right_corner_y", "mouth_center_top_lip_x",
                        "mouth_center_top_lip_y", "mouth_center_bottom_lip_x",
                        "mouth_center_bottom_lip_y")

# Combine the flipped data set with d.train and im.train
#d.trainL <- rbind(d.train, d.train1)
d.trainL <- rbind(d.train,d.train1)

# Bind im.train and im.flip data 
im.trainL <- rbind(im.train, image.flip)

####################### TRAIN MODEL: COMPUTE AVERAGE PATCH ###############################
# for each one, compute the average patch
mean.patches <- foreach(coord = coordinate.names) %dopar% {
  cat(sprintf("computing mean patch for %s\n", coord))
  coord_x <- paste(coord, "x", sep="_")
  coord_y <- paste(coord, "y", sep="_")
  
  # compute average patch
  patches <- foreach (i = 1:nrow(d.train), .combine=rbind) %do% {
    im  <- matrix(data = im.train[i,], nrow=96, ncol=96)
   # im  <- matrix(data = image.histeq[i,], nrow=96, ncol=96)
   # im  <- matrix(data = image.temp[i,], nrow=96, ncol=96)
   # im  <- matrix(data = image.blur[i,], nrow=96, ncol=96)
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
  matrix(data = colMeans(patches), nrow=2*patch_size+1, ncol=2*patch_size+1)
}

################ TEST MODEL: FIND PATCH FOR TEST DATA #################################################
# for each coordinate and for each test image, find the position that best correlates with the average patch
p <- foreach(coord_i = 1:length(coordinate.names), .combine=cbind) %dopar% {
  # the coordinates we want to predict
  coord   <- coordinate.names[coord_i]
  coord_x <- paste(coord, "x", sep="_")
  coord_y <- paste(coord, "y", sep="_")
  
  # the average of them in the training set (our starting point)
  mean_x  <- mean(d.train[, coord_x], na.rm=T)
  mean_y  <- mean(d.train[, coord_y], na.rm=T)
  
  # search space: 'search_size' pixels centered on the average coordinates 
  x1 <- as.integer(mean_x)-search_size
  x2 <- as.integer(mean_x)+search_size
  y1 <- as.integer(mean_y)-search_size
  y2 <- as.integer(mean_y)+search_size
  
  # ensure we only consider patches completely inside the image
  x1 <- ifelse(x1-patch_size<1,  patch_size+1,  x1)
  y1 <- ifelse(y1-patch_size<1,  patch_size+1,  y1)
  x2 <- ifelse(x2+patch_size>96, 96-patch_size, x2)
  y2 <- ifelse(y2+patch_size>96, 96-patch_size, y2)
  
  # build a list of all positions to be tested
  params <- expand.grid(x = x1:x2, y = y1:y2)
  
  # for each image...
  r <- foreach(i = 1:nrow(d.test), .combine=rbind) %do% {
    if ((coord_i==1)&&((i %% 100)==0)) { cat(sprintf("%d/%d\n", i, nrow(d.test))) }
    im <- matrix(data = im.test[i,], nrow=96, ncol=96)
    
    # ... compute a score for each position ...
    r  <- foreach(j = 1:nrow(params), .combine=rbind) %do% {
      x     <- params$x[j]
      y     <- params$y[j]
      p     <- im[(x-patch_size):(x+patch_size), (y-patch_size):(y+patch_size)]
      score <- cor(as.vector(p), as.vector(mean.patches[[coord_i]]))
      score <- ifelse(is.na(score), 0, score)
      data.frame(x, y, score)
    }
    
    # ... and return the best
    best <- r[which.max(r$score), c("x", "y")]
  }
  names(r) <- c(coord_x, coord_y)
  r
}


################ PREPARE SUBMISSION FILE #################################################

# prepare file for submission
predictions        <- data.frame(ImageId = 1:nrow(d.test), p)
submission         <- melt(predictions, id.vars="ImageId", variable.name="FeatureName", value.name="Location")
example.submission <- read.csv(paste0(data.dir, 'IdLookupTable.csv'))
sub.col.names      <- names(example.submission)
example.submission$Location <- NULL

submission <- merge(example.submission, submission, all.x=T, sort=F)
submission <- submission[, sub.col.names]

write.csv(submission, file="submission_search.csv", quote=F, row.names=F)
SampleSubmission2 <- cbind(submission[, 1], submission[ ,4])
colnames(SampleSubmission2) <- c("RowId", "Location")
write.csv(SampleSubmission2, file="SampleSubmission_base15.csv", quote=F, row.names=F)
#write.csv(SampleSubmission2, file="SampleSubmission_histstretch.csv", quote=F, row.names=F)
#write.csv(SampleSubmission2, file="SampleSubmission_histeq.csv", quote=F, row.names=F)
#write.csv(SampleSubmission2, file="SampleSubmission_gaussblurr.csv", quote=F, row.names=F)
#write.csv(SampleSubmission2, file="SampleSubmission_histeqblurr.csv", quote=F, row.names=F)

################ SAVE/LOAD SESSION DATA #################################################

save(d.train, im.train, d.test , im.test, image.temp, image.histeq, image.blur, image.flip ,
     d.trainL, im.trainL,  coordinate.names, patch_size, search_size, file='data.Rd')

load('data.Rd')

d.train[1,]
