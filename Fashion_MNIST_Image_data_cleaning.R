# Author: Ray Liu
# Reference: "Fashion MNIST | Image Classification | Keras | R" by Devashree Madhugiri
###############################
#### install/load library
###############################
library(keras)
library(tidyverse)
library(caret)

###############################
#### settings
###############################
#following (Fan et al., 2021), we set (n,m)=(1200, 10), (480, 25) and (240, 50)
N_train <- 12000
N_test <- 2000
Ms <- c(5,10,25,50)
ns <- N_train/Ms

###############################
#### load the fashion_mnist data
###############################
fashion_mnist <- dataset_fashion_mnist()
# 'fashion_mnist' contaisn two lists, and we use '%<-%' to split it into two lists
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test
class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot') # Fan selects 'Sneaker' and 'Ankle boot'
# Training data contain 60000 samples and each picture is made of 28 x 28 pixels.
# Test data contains 10000 samples.
dim(train_images)
dim(test_images)
two_class_name <- c('Sneaker','Ankle boot')
two_class_indices <- c(7,9)

###############################
#### standardization and partition
###############################
# Create vectors for row and column indices
rows <- 1:28
cols <- 1:28
# Create a data frame of all combinations of row and column indices
grid <- expand.grid(rows, cols)
# Combine the row and column indices to label the features
colname_pixel <- paste(paste('r',grid[,2],sep = '_'), paste('c',grid[,1],sep = '_'), sep = "_")
# a toy example to figure out the meanings of each column (row first, column second)
#a <- train_images[1:2,,]
#a1 <- a %>% array_reshape(c(2,28*28))

# normalize (0,255) to (0,1) interval and standardize
train_images <- train_images / 255
test_images <- test_images / 255

# find the indices of sneaker and boot
train_sneaker_boot_ind=which(train_labels==7|train_labels==9)
test_sneaker_boot_ind=which(test_labels==7|test_labels==9)

################################
### form the data for nn
################################
x_train_nn <- train_images %>%
  array_reshape(c(60000, 28, 28, 1))
x_test_nn <- test_images %>%
  array_reshape(c(10000, 28, 28, 1))
train_x_nn <- x_train_nn[train_sneaker_boot_ind,,,]
test_x_nn <- x_test_nn[test_sneaker_boot_ind,,,]

################################
### form the data for regression
################################
x_train <- train_images %>%
  array_reshape(c(60000, 28*28)) %>% 
  as.data.frame()
x_test <- test_images %>%
  array_reshape(c(10000, 28*28)) %>% 
  as.data.frame()
train_x <- cbind.data.frame(y=train_labels,x_train)[train_sneaker_boot_ind,] %>% 
  select(!y) %>%  scale() 
test_x <- cbind.data.frame(y=test_labels,x_test)[test_sneaker_boot_ind,] %>% 
  select(!y) %>%  scale() 
colnames(train_x) <- colnames(test_x) <- colname_pixel

# there are some features all 0, and after scaling it produces NA.
# 'NA_column' aims to these columns out.
NA_columns1 <- apply(train_x, 2, function(col) all(is.nan(col)))
train_nan_index <- colnames(train_x)[which(NA_columns1==1)]
NA_columns2 <- apply(test_x, 2, function(col) all(is.nan(col)))
test_nan_index <- colnames(test_x)[which(NA_columns2==1)]
NA_columns <- NA_columns1|NA_columns2
nan_index <- colnames(train_x)[which(NA_columns==1)]

train_x <- train_x[,!NA_columns]
train_x <- cbind(intercept=rep(1,nrow(train_x)),train_x)
test_x <- test_x[,!NA_columns]
test_x <- cbind(intercept=rep(1,nrow(test_x)),test_x)


# remove the same columns from 'train_x' to guarantee full rank,
# and we just remove these same columns in 'test_x'
dup_columns <- duplicated(train_x,MARGIN = 2)
dup_index <- colnames(train_x)[which(dup_columns==1)]
train_x <- train_x[,!dup_columns]
test_x <- test_x[,!dup_columns]

train_y <- train_labels[train_sneaker_boot_ind] %>% unlist() %>% unname() %>% 
  factor(levels=two_class_indices,labels=c(0,1)) %>% as.numeric()-1
test_y <- test_labels[test_sneaker_boot_ind] %>% unlist() %>% unname() %>% 
  factor(levels=two_class_indices,labels=c(0,1)) %>% as.numeric()-1



rm(train_images,test_images,train_labels,test_labels,x_train,x_test,x_train_nn,x_test_nn)
save.image("Fashion_MNIST_Image.Rdata")
