# Load necessary libraries
library(caret)
library(rsample)
library(party)
library(caTools) 
library(randomForest) 
# Read data
traindata <- read.csv('/home/deniz/Masaüstü/yükseklisans/ayz/xaifinal/binarydataset/ionosphere.data.csv', sep=',')

# Rename columns
colnames(traindata)[1:34] <- paste0("prop", 1:35)

# Replace 'g' with 1 and 'b' with 0
traindata$g[traindata$g == "g"] <- 1
traindata$g[traindata$g == "b"] <- 0

# Create a data split
set.seed(123)
data_split <- initial_split(traindata, prop = 0.70)
train_data <- training(data_split)
#train_data$g <- as.numeric(as.character(train_data$g))
test_data <- testing(data_split)

# Define predictor and response variables
train_x <- train_data[, -which(names(train_data) == 'g')]

train_y <- train_data[['g']]

test_x <- test_data[, -which(names(test_data) == 'g')]
test_y <- test_data[['g']]

########################### decision treees ######################################
# Fit the ctree model
output.tree <- ctree(g ~ ., data = train_data)

# Plot the tree
plot(output.tree)

########################### decision treees ######################################
#################### Random classifier ###############33


classifier_RF <- randomForest(x = train_x, 
                              y = as.factor(train_y),  # Ensure train_y is a factor
                              ntree = 500)

# Predicting the Test set results
y_pred <- predict(classifier_RF, newdata = test_x)

# Confusion Matrix
confusion_mtx <- table(test_y, y_pred)
confusion_mtx

# Plotting model
plot(classifier_RF)

# Importance plot
importance(classifier_RF)


############################### SVM #############################
#https://www.geeksforgeeks.org/classifying-data-using-support-vector-machinessvms-in-r/
install.packages('e1071') 
library(e1071) 

dftrain<-data.frame(train_data)
dftrain <- dftrain[, sapply(dftrain, sd) != 0]

classifier <- svm(g ~ ., data = dftrain, type = 'C-classification', kernel = 'radial', cost = 10, scale = FALSE)


test_y <- as.factor(test_y)

# Predicting the Test set results 
y_pred <- predict(classifier, newdata = test_x)

# Confusion Matrix 
confusion_mtx <- table(test_y, y_pred)
print(confusion_mtx)

# Plotting the SVM classification
plot(classifier, dftrain)

############################### SVM #############################


library(kernlab)
classifier <- ksvm(g ~ ., data = dftrain, type = 'C-svc', kernel = 'rbfdot', C = 10)
plot(classifier, data = dftrain)


############################### GBM CLASİFİER #############################
install.packages("lightgbm")
library(caret)
library(lightgbm)

# Assuming train_x and test_x are data frames, and train_y and test_y are numeric or factor vectors

# Convert train_y and test_y to factor (assuming it's a classification problem)
train_y <- as.factor(train_y)
test_y <- as.factor(test_y)

# Create lgb.Dataset objects
dtrain <- lgb.Dataset(data = as.matrix(train_x), label = as.numeric(train_y))
dtest <- lgb.Dataset.create.valid(dtrain, data = as.matrix(test_x), label = as.numeric(test_y))

# Define parameters
params <- list(
  objective = 'multiclass',
  metric = 'multi_error',
  num_class = 3
)

# Validation data
valids <- list(test = dtest)

# Train the model
model <- lgb.train(params, dtrain, nrounds = 100, valids)

# Print best scores
print(model$best_score)

# Prediction
pred <- predict(model, as.matrix(test_x))
pred_y <- max.col(pred) - 1