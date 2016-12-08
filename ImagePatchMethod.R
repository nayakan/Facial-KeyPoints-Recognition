load('data.Rd')

# Pick a single keypoint: left_eye_center
coord <- "left_eye_center"

# patch_size is the number of pixels we 
# are going to extract in each direction 
# around the center of the keypoint.
patch_size <- 10

library('foreach')

coord_x <- paste(coord, "x", sep="_")
coord_y <- paste(coord, "y", sep="_")
patches <- foreach( i = 1:nrow(d.train), .combine = rbind) %do% {
  im <- matrix(data = im.train[i, ], nrow = 96, ncol = 96)
  x <- d.train[i, coord_x]
  y <- d.train[i, coord_y]
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
# average left eye computed across our 7049 images.
image(1:21, 1:21, mean.patch[21:1,21:1], col=gray((0:255)/255))

# use this average_patch to search for the same keypoint in the test images. 
search_size <- 2
# center the search on the average keypoint location, and go search_size pixels in each direction
mean_x <- mean(d.train[, coord_x], na.rm=T)
mean_y <- mean(d.train[, coord_y], na.rm=T)
x1     <- as.integer(mean_x)-search_size
x2     <- as.integer(mean_x)+search_size
y1     <- as.integer(mean_y)-search_size
y2     <- as.integer(mean_y)+search_size

# expand.grid to build a data frame with all combinations of x's and y's
params <- expand.grid(x = x1:x2, y = y1:y2)

# Given a test image we need to try all these combinations, 
# and see which one best matches the average_patch
im <- matrix(data = im.test[1,], nrow=96, ncol=96)

r  <- foreach(j = 1:nrow(params), .combine=rbind) %dopar% {
  x     <- params$x[j]
  y     <- params$y[j]
  p     <- im[(x-patch_size):(x+patch_size), (y-patch_size):(y+patch_size)]
  score <- cor(as.vector(p), as.vector(mean.patch))
  score <- ifelse(is.na(score), 0, score)
  data.frame(x, y, score)
}

# return the coordinate with the highest score
best <- r[which.max(r$score), c("x", "y")]
best

