# Load necessary libraries
library(caret)
library(xgboost)
library(caret)
library(lightgbm)
library(rsample)
library(party)
library(caTools) 
library(randomForest) 
library(corrplot)
library(dplyr)

df <- read.csv('/home/deniz/Masaüstü/yükseklisans/ayz/xaifinal/forestfires.csv', sep=',')

# Get unique values for each column
unique_values <- sapply(df, function(y) sum(length(unique(y))))
head(unique_values)
# Display unique values
na_count <-sapply(df, function(y) sum(length(which(is.na(y)))))
head(na_count)

categorical_cols <- df %>% select_if(is.character)
head(categorical_cols)

# Select numerical columns
numerical_cols <- df %>% select_if(is.numeric)


corr_matrix <- cor(numerical_cols )

# Open a new graphics device
dev.new()

# Plot the correlation matrix
corrplot::corrplot(corr_matrix, method = "color")



reg_model <- lm(area ~ temp + RH + wind + rain + FFMC + DMC + DC + ISI, data = df)
summary(reg_model)
# Print summary of the regression model




# Split into training (70%) and testing set (30%)
parts <- createDataPartition(df$area, p = 0.7, list = FALSE)


train <- df[parts, ]
test <- df[-parts, ]

# Define predictor and response variables in the training set
train_x <- data.matrix(train[, -which(names(train) == 'area')])
train_y <- train[['area']]

# Define predictor and response variables in the testing set
test_x <- data.matrix(test[, -which(names(test) == 'area')])
test_y <- test[['area']]


# Define final training and testing sets
xgb_train <- xgb.DMatrix(data = train_x, label = train_y)
xgb_test <- xgb.DMatrix(data = test_x, label = test_y)

# Define watchlist
watchlist <- list(train = xgb_train, test = xgb_test)

# Fit XGBoost model and display training and testing data at each round
model <- xgb.train(data = xgb_train, max.depth = 3, watchlist = watchlist, nrounds = 70)

# Make predictions on the testing set
pred_y <- predict(model, xgb_test)

# Calculate MSE, MAE, and RMSE
mse <- mean((test_y - pred_y)^2)
mae <- caret::MAE(test_y, pred_y)
rmse <- caret::RMSE(test_y, pred_y)

print(paste("MSE:", mse))
print(paste("MAE:", mae))
print(paste("RMSE:", rmse))





dtrain = lgb.Dataset(train_x, label = train_y)
dtest = lgb.Dataset.create.valid(dtrain, test_x, label = test_y)

# define parameters
params = list(
  objective = "regression"
  , metric = "l2"
  , min_data = 1L
  , learning_rate = .3
)

# validataion data
valids = list(test = dtest)


# train model 
model1 = lgb.train(
  params = params
  , data = dtrain
  , nrounds = 5L
  , valids = valids
)
lgb.get.eval.result(model1, "test", "l2")
# prediction
pred_y1 = predict(model1, test_x) 


# accuracy check
mse1 = mean((test_y - pred_y1)^2)
mae1 = caret::MAE(test_y, pred_y1)
rmse1 = caret::RMSE(test_y, pred_y1)

cat("MSE: ", mse1, "\nMAE: ", mae1, "\nRMSE: ", rmse1)





## gbm model
#https://datascienceplus.com/gradient-boosting-in-r/
library(gbm)


train$month <- as.factor(train$month)
train$day <- as.factor(train$day)

# Assuming you have a data frame named 'train' with the response variable 'Diabetes_012'
gbmmodel <- gbm(area ~ ., data = train, distribution = "gaussian", n.trees = 20,
                shrinkage = 0.01, interaction.depth = 4)

# Assuming your 'test_x' is a data frame
# Convert it to a data frame if it's a matrix
test_x_df <- as.data.frame(test_x)

# Predict using the gbm model
pred_y2 <- predict(gbmmodel, newdata = test_x_df, n.trees = 20, type = "response")

# Assuming 'test_y' is the true response variable
# accuracy check
mse2 <- mean((test_y - pred_y2)^2)
mae2 <- caret::MAE(test_y, pred_y2)
rmse2 <- caret::RMSE(test_y, pred_y2)

cat("MSE: ", mse2, "\nMAE: ", mae2, "\nRMSE: ", rmse2)


####################### XAI #############################
####################### breakdown #############################
library(DALEX) 
explain_xboost_breakdown <- DALEX::explain(model = model,
                                           data = train_x,
                                           y = train_y,
                                           label = "xboost")

# Assuming train_x[[1]] is a numeric vector or matrix
new_observation <- as.data.frame(train_x[1, , drop = FALSE])

# Extract feature names from the XGBoost model
model_feature_names <- colnames(model$feature_names)

# Check and set correct feature names in the new observation
if (!identical(colnames(new_observation), model_feature_names)) {
  colnames(new_observation) <- model_feature_names
}

# Convert the new observation to a matrix
new_observation_matrix <- as.matrix(new_observation)

# Predict using the XGBoost model
prediction <- predict(model, newdata = new_observation_matrix)

# Calculate breakdown values
bd_xboost <- predict_parts(explain_xboost_breakdown, new_observation = new_observation_matrix, type = "break_down")

# Plot breakdown values
plot(bd_xboost)


####################### breakdown #############################
####################### shap #############################

explain_xboost_shap <- DALEX::explain(model = model,
                                           data = train_x,
                                           y = train_y,
                                           label = "xboost shapley değerleri")

prediction_xboost <- predict(model, newdata = new_observation_matrix)

sh_xboost <- predict_parts(explain_xboost_shap, new_observation = new_observation_matrix, type = "shap")

plot(sh_xboost)
####################### shap #############################