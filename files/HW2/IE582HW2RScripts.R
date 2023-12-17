library(data.table)
library(tidyverse)
library(ggplot2)
library(rpart.plot)
library(skimr)
library(GGally)
library(caret) # for machine learning and predictive modeling
library(class) # for kNN
library(kknn) # for kNN with Manhattan Distance
library(rpart) # for rpart
library(gbm) # for Boosted Decision Trees
library(randomForest) # for random forests
library(reshape2) # for graphs
library(FNN)
library(class)
require(rattle)
library(pROC)
library(pdp)
library(MLmetrics)
library(magick)

data_path <- "C:/Users/anil.turgut/Desktop/582HW2/Dataset/"


### 1) DIABETES CASE

diabetes_data_path <- paste0(data_path,"Diabetes/diabetes.csv")

diabetes_data <- read.csv(diabetes_data_path)

dim(diabetes_data)

# Data Analysis & Preprocessing

str(diabetes_data)

summary(diabetes_data)

colSums(sapply(diabetes_data, is.na))

skim(diabetes_data)

diabetes_data$Diabetes_binary <- as.factor(diabetes_data$Diabetes_binary)
diabetes_data <- diabetes_data[!duplicated(diabetes_data), ]

ggplot(diabetes_data, aes(x = Diabetes_binary )) +
  geom_bar(stat = "count") +
  labs(title = "Count of Diabetes", x = "Having Diabetes", y = "Count")

ggcorr(diabetes_data,
       method = c("pairwise"),
       nbreaks = 6,
       hjust = 0.8,
       label = TRUE,
       label_size = 3,
       color = "grey20")

scaled_diabetes_data <- diabetes_data
scaled_diabetes_data[,-1] <- scale(scaled_diabetes_data[,-1])

validationIndex <- createDataPartition(scaled_diabetes_data$Diabetes_binary, p=0.70, list=FALSE)

knn_train <- scaled_diabetes_data[validationIndex,] # 70% of data to training
knn_test <- scaled_diabetes_data[-validationIndex,] 


cat("Dimension of the main dataset:",dim(diabetes_data))
cat("Dimension of the train dataset:",dim(knn_train))
cat("Dimension of the test dataset:",dim(knn_test))


trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "Accuracy"
set.seed(7)

# kNN Part
fit.knn <- train(Diabetes_binary~., data=knn_train, method="knn",
                 metric=metric ,trControl=trainControl)
knn.optimal.k <- fit.knn$bestTune # keep this Initial k for testing with knn() function in next section

print(fit.knn)

plot(fit.knn)

set.seed(7)

prediction <- predict(fit.knn, newdata = knn_test)
cm_knn <- confusionMatrix(prediction, knn_test$Diabetes_binary)
print(cm_knn)


initial_k <- sqrt(NROW(scaled_diabetes_data))
initial_k

knn.floor <- knn(train=knn_train[,-1], test=knn_test[,-1], 
                 cl=knn_train$Diabetes_binary, k=floor(initial_k))

# use confusion matrix to calculate accuracy
cm.floor <- confusionMatrix(knn_test$Diabetes_binary,knn.floor)
cm.floor

# optimal k
knn.best <- knn(train=knn_train[,-1], test=knn_test[,-1], cl=knn_train$Diabetes_binary, k= 9)
cf <- confusionMatrix(knn_test$Diabetes_binary,knn.best)
print(cf)

# Train the KNN model
knn.manhattan <- train.kknn(Diabetes_binary ~ ., data = knn_train, ks = 9, scale = TRUE, distance = 1)
predictions.manhattan <- predict(knn.manhattan, newdata = knn_test)

cm.manhattan <- table(predictions.manhattan, knn_test$Diabetes_binary)
print(cm.manhattan)
accuracy.manhattan <- sum(diag(cm.manhattan)) / sum(cm.manhattan)
print(paste("Accuracy for kNN with Manhattan Distance:", round(accuracy.manhattan,2)))


# Decision Tree Part
diabetes_data_dt <- data.table(diabetes_data)

head(diabetes_data_dt)
dim(diabetes_data_dt)
# Remove rows with any NA values
diabetes_data_dt <- na.omit(diabetes_data_dt)
validationIndex <- createDataPartition(diabetes_data_dt$Diabetes_binary, p=0.70, list=FALSE)

dt_train <- diabetes_data_dt[validationIndex,] # 70% of data to training
dt_test <- diabetes_data_dt[-validationIndex,] 

# Create a data frame to store cross-validation results
cv_results <- data.frame(minbucket = numeric(), accuracy = numeric())

set.seed(15)

# Define a range of minsplit values to try
minbucket_values <- c(1, 5, 10, 15, 20) 

# Perform cross-validation for each minbucket value
for (minbucket_val in minbucket_values) {
  # Create a decision tree model with the current minbucket value
  tree_model <- rpart(Diabetes_binary~.,diabetes_data_dt,method='class',
                      control=rpart.control(cp=0, minbucket = minbucket_val, minsplit = 2*minbucket_val))
  
  predictions <- predict(tree_model, newdata = dt_train, type = "class")
  correct_predictions <- sum(predictions == dt_train$Diabetes_binary)
  # Calculate accuracy
  accuracy <- round(correct_predictions / nrow(dt_train),5)
  
  # Store the results in the data frame
  cv_results <- rbind(cv_results, data.frame(minbucket = minbucket_val, accuracy = accuracy))
}

# Print the cross-validation results
print(cv_results)

minbucket_val <- 1
best_tree_model <- rpart(Diabetes_binary~.,diabetes_data_dt,method='class',
                         control=rpart.control(cp=0, minbucket = minbucket_val, minsplit = 2*minbucket_val))

fancyRpartPlot(best_tree_model)
predictions_model <- predict(best_tree_model, newdata = dt_train, type = "class")
cm_train <- confusionMatrix(predictions_model, dt_train$Diabetes_binary, positive="0")
cm_train
predictions_model.test <- predict(best_tree_model, newdata = dt_test, type = "class")
cm_test <- confusionMatrix(predictions_model.test, dt_test$Diabetes_binary, positive="0")
cm_test



p1 <- predict(best_tree_model, dt_test, type = 'prob')
p1 <- p1[,2]
r <- multiclass.roc(dt_test$Diabetes_binary, p1, percent = TRUE)
roc <- r[['rocs']]
r1 <- roc[[1]]
plot.roc(r1,
         print.auc=TRUE,
         auc.polygon=TRUE,
         grid=c(0.1, 0.2),
         grid.col=c("green", "red"),
         max.auc.polygon=TRUE,
         auc.polygon.col="lightblue",
         print.thres=TRUE,
         main= 'ROC Curve')


# Random Forest Part

diabetes_data_rf <- data.table(diabetes_data)

head(diabetes_data_rf)
dim(diabetes_data_rf)
# Remove rows with any NA values
diabetes_data_rf <- na.omit(diabetes_data_rf)
validationIndex <- createDataPartition(diabetes_data_rf$Diabetes_binary, p=0.70, list=FALSE)

rf_train <- diabetes_data_rf[validationIndex,] # 70% of data to training
rf_test <- diabetes_data_rf[-validationIndex,] 


# Set the number of trees and nodesize
num_trees <- 500
min_obs_per_leaf <- 5

# Create a grid of mtry values to explore
mtry_values <- c(2, 4, 6, 8, 10, 15, 20)  # Add more values as needed

# Create an empty data frame to store results
rf_results <- data.frame(mtry = numeric(0), error_rate = numeric(0))

# Perform grid search
for (m in mtry_values) {
  # Train the Random Forest model
  rf_model <- randomForest(Diabetes_binary ~ ., data = rf_train, 
                           ntree = num_trees, nodesize = min_obs_per_leaf, mtry = m)
  
  # Make predictions on the training set
  predictions <- predict(rf_model, dt_train)
  
  # Calculate the error rate (you may want to use a more appropriate metric)
  error_rate <- mean(predictions != rf_train$Diabetes_binary)
  
  # Store the results
  rf_results <- rbind(rf_results, data.frame(mtry = m, error_rate = error_rate))
}

# Print the results
print(rf_results)

# Create a line plot
ggplot(rf_results, aes(x = mtry, y = error_rate)) +
  geom_line() +
  geom_point() +
  labs(title = "Error Rate vs. mtry",
       x = "mtry",
       y = "Error Rate") +
  theme_minimal()

best_rf_model <- randomForest(Diabetes_binary ~ ., data = rf_train, 
                              ntree = num_trees, nodesize = min_obs_per_leaf, mtry = 2)

# Make predictions on the train set
rf_predictions.train <- predict(best_rf_model, newdata = rf_train)

# Create the confusion matrix
rf_confusion_matrix.train <- confusionMatrix(rf_predictions.train, rf_train$Diabetes_binary)

# Print the confusion matrix
print(rf_confusion_matrix.train)

# Make predictions on the test set
rf_predictions.test <- predict(best_rf_model, newdata = rf_test)

# Create the confusion matrix
rf_confusion_matrix.test <- confusionMatrix(rf_predictions.test, rf_test$Diabetes_binary)

# Print the confusion matrix
print(rf_confusion_matrix.test)

# Plot variable importance
varImpPlot(best_rf_model)

# Create partial dependence plot
partial_plot <- partial(best_rf_model, pred.var = 'BMI', data = dt_test)

# Plot the partial dependence plot
plot(partial_plot)


# Gradient Boosting Machines (GBM) Part

diabetes_data_gbm <- data.table(diabetes_data)
# Making the target value numeric to use in gbm
diabetes_data_gbm$Diabetes_binary <- as.numeric(diabetes_data_gbm$Diabetes_binary)
diabetes_data_gbm <- transform(diabetes_data_gbm, Diabetes_binary=Diabetes_binary-1)

head(diabetes_data_gbm)
dim(diabetes_data_gbm)

# Remove rows with any NA values
diabetes_data_gbm <- na.omit(diabetes_data_gbm)
validationIndex <- createDataPartition(diabetes_data_gbm$Diabetes_binary, p=0.70, list=FALSE)

gbm_train <- diabetes_data_gbm[validationIndex,] # 70% of data to training
gbm_test <- diabetes_data_gbm[-validationIndex,] 

# To use in cross validation I have replace "0" with "no" else "yes"
gbm_train_revised <- gbm_train
gbm_test_revised <- gbm_test 

gbm_train_revised$Diabetes_binary <- ifelse(gbm_train_revised$Diabetes_binary == "0", "no", "yes")
gbm_train_revised$Diabetes_binary <- as.factor(gbm_train_revised$Diabetes_binary )

gbm_test_revised$Diabetes_binary <- ifelse(gbm_test_revised$Diabetes_binary == "0", "no", "yes")
gbm_test_revised$Diabetes_binary <- as.factor(gbm_test_revised$Diabetes_binary )

cat("Dimension of the main dataset:",dim(diabetes_data_gbm))
cat("Dimension of the train dataset:",dim(gbm_train))
cat("Dimension of the test dataset:",dim(gbm_test))

set.seed(10)

n_repeats=5
n_folds=10

fitControl=trainControl(method = "repeatedcv",
                        number = n_folds,
                        repeats = n_repeats,
                        classProbs=TRUE, summaryFunction=twoClassSummary)
## gradient boosting
gbmGrid=expand.grid(interaction.depth = c(3, 5), 
                    n.trees = c(1:5)*100, 
                    shrinkage = c(0.05,0.1),
                    n.minobsinnode = 10)
set.seed(1)                        
gbm_fit=train(Diabetes_binary ~ ., data = gbm_train_revised, 
              method = "gbm", 
              trControl = fitControl, metric='ROC',
              tuneGrid = gbmGrid,
              verbose=F) #verbose is an argument from gbm, prints to screen
gbm_fit

plot(gbm_fit)

selected_gbm=tolerance(gbm_fit$results, metric = "ROC", tol = 2, maximize = TRUE)  
gbm_fit$results[selected_gbm,]

noftrees=100
depth=3
learning_rate=0.05

boosting_model=gbm(Diabetes_binary~., data=gbm_train,distribution = 'bernoulli', n.trees = noftrees,
                   interaction.depth = depth, n.minobsinnode = 10, shrinkage =learning_rate, cv.folds = 10)

summary(boosting_model)

gbm.perf(boosting_model, method = "cv")

prediction.train <- predict(boosting_model, newdata = gbm_train, type = "response")

confusion_data.train <- data.frame(actual = gbm_train$Diabetes_binary,
                                   predicted = ifelse(prediction.train > 0.5,1,0))  # Assuming a threshold of 0.5 for binary classification

# Create a confusion matrix
conf_matrix.train <- confusionMatrix(as.factor(confusion_data.train$predicted), as.factor(confusion_data.train$actual))

print(conf_matrix.train)


prediction.test <- predict(boosting_model, newdata = gbm_test, type = "response")

confusion_data.test <- data.frame(actual = gbm_test$Diabetes_binary,
                                  predicted = ifelse(prediction.test > 0.5,1,0))  # Assuming a threshold of 0.5 for binary classification

# Create a confusion matrix
conf_matrix.test <- confusionMatrix(as.factor(confusion_data.test$predicted), as.factor(confusion_data.test$actual))

print(conf_matrix.test)


### 2) STUDENT SUCCESS CASE

students_data_path <- paste0(data_path,"StudentSuccess/students.csv")

students_data <- read.csv(students_data_path, header = TRUE, sep = ";", quote = "\"")

dim(students_data)

# Data Analysis & Preprocessing

str(students_data)

summary(students_data)

colSums(sapply(students_data, is.na))

skim(students_data)


students_data$Target <- as.factor(students_data$Target)
students_data <- students_data[!duplicated(students_data), ]

ggplot(students_data, aes(x = Target )) +
  geom_bar(stat = "count") +
  labs(title = "Count of Student Success", x = "Student Success", y = "Count")

ggcorr(students_data,
       method = c("pairwise"),
       nbreaks = 6,
       hjust = 0.8,
       label = TRUE,
       label_size = 3,
       color = "grey20")

# knn Part 

scaled_students_data <- students_data
scaled_students_data[,-37] <- scale(scaled_students_data[,-37])

validationIndex <- createDataPartition(scaled_students_data$Target, p=0.70, list=FALSE)

knn_train <- scaled_students_data[validationIndex,] # 70% of data to training
knn_test <- scaled_students_data[-validationIndex,] 


cat("Dimension of the main dataset:",dim(scaled_students_data))
cat("Dimension of the train dataset:",dim(knn_train))
cat("Dimension of the test dataset:",dim(knn_test))


# knn CV

trainControl <- trainControl(method="repeatedcv", number=10, repeats=5)
metric <- "Accuracy"
set.seed(8)

# kNN Part
fit.knn <- train(Target~., data=knn_train, method="knn",
                 metric=metric ,trControl=trainControl)
knn.optimal.k <- fit.knn$bestTune # keep this Initial k for testing with knn() function in next section

print(fit.knn)

plot(fit.knn)

prediction <- predict(fit.knn, newdata = knn_test)
cm_knn <- confusionMatrix(prediction, knn_test$Target)
print(cm_knn)


initial_k <- sqrt(NROW(scaled_students_data))
initial_k

knn.floor <- knn(train=knn_train[,-37], test=knn_test[,-37], 
                 cl=knn_train$Target, k=floor(initial_k))

# use confusion matrix to calculate accuracy
cm.floor <- confusionMatrix(knn_test$Target,knn.floor)
cm.floor

# optimal k
knn.best <- knn(train=knn_train[,-37], test=knn_test[,-37], cl=knn_train$Target, k= 9)
cf <- confusionMatrix(knn_test$Target,knn.best)
print(cf)

# Train the KNN model
knn.manhattan <- train.kknn(Target ~ ., data = knn_train, ks = 9, scale = TRUE, distance = 1)
predictions.manhattan <- predict(knn.manhattan, newdata = knn_test)

cm.manhattan <- table(predictions.manhattan, knn_test$Target)
print(cm.manhattan)
accuracy.manhattan <- sum(diag(cm.manhattan)) / sum(cm.manhattan)
print(paste("Accuracy for kNN with Manhattan Distance:", round(accuracy.manhattan,2)))



# Decision Tree Part

students_data_dt <- data.table(students_data)

head(students_data_dt)
dim(students_data_dt)
# Remove rows with any NA values
students_data_dt <- na.omit(students_data_dt)
validationIndex <- createDataPartition(students_data_dt$Target, p=0.70, list=FALSE)

dt_train <- students_data_dt[validationIndex,] # 70% of data to training
dt_test <- students_data_dt[-validationIndex,] 

# Create a data frame to store cross-validation results
cv_results <- data.frame(minbucket = numeric(), accuracy = numeric())

set.seed(10)

# Define a range of minsplit values to try
minbucket_values <- c(1, 5, 10, 15, 20) 

# Perform cross-validation for each minbucket value
for (minbucket_val in minbucket_values) {
  # Create a decision tree model with the current minbucket value
  tree_model <- rpart(Target~.,dt_train,method='class',
                      control=rpart.control(cp=0, minbucket = minbucket_val, minsplit = 2*minbucket_val))
  
  predictions <- predict(tree_model, newdata = dt_train, type = "class")
  correct_predictions <- sum(predictions == dt_train$Target)
  # Calculate accuracy
  accuracy <- round(correct_predictions / nrow(dt_train),5)
  
  # Store the results in the data frame
  cv_results <- rbind(cv_results, data.frame(minbucket = minbucket_val, accuracy = accuracy))
}

# Print the cross-validation results
print(cv_results)

minbucket_val <- 10
best_tree_model <- rpart(Target~.,dt_train,method='class',
                         control=rpart.control(cp=0, minbucket = minbucket_val, minsplit = 2*minbucket_val))

fancyRpartPlot(best_tree_model)

predictions_model <- predict(best_tree_model, newdata = dt_train, type = "class")
cm_train <- confusionMatrix(predictions_model, dt_train$Target)
cm_train
predictions_model.test <- predict(best_tree_model, newdata = dt_test, type = "class")
cm_test <- confusionMatrix(predictions_model.test, dt_test$Target)
cm_test



p1 <- predict(best_tree_model, dt_test, type = 'prob')
p1 <- p1[,2]
r <- multiclass.roc(dt_test$Target, p1, percent = TRUE)
roc <- r[['rocs']]
r1 <- roc[[1]]
plot.roc(r1,
         print.auc=TRUE,
         auc.polygon=TRUE,
         grid=c(0.1, 0.2),
         grid.col=c("green", "red"),
         max.auc.polygon=TRUE,
         auc.polygon.col="lightblue",
         print.thres=TRUE,
         main= 'ROC Curve')

# Random Forest Part


students_data_rf <- data.table(students_data)

head(students_data_rf)
dim(students_data_rf)
# Remove rows with any NA values
students_data_rf <- na.omit(students_data_rf)
validationIndex <- createDataPartition(students_data_rf$Target, p=0.70, list=FALSE)

rf_train <- students_data_rf[validationIndex,] # 70% of data to training
rf_test <- students_data_rf[-validationIndex,] 


# CV RF

# Set the number of trees and nodesize
num_trees <- 500
min_obs_per_leaf <- 5

# Create a grid of mtry values to explore
mtry_values <- c(2, 4, 6, 8, 10, 15, 20)  # Add more values as needed

# Create an empty data frame to store results
rf_results <- data.frame(mtry = numeric(0), error_rate = numeric(0))

# Perform grid search
for (m in mtry_values) {
  # Train the Random Forest model
  rf_model <- randomForest(Target ~ ., data = rf_train, 
                           ntree = num_trees, nodesize = min_obs_per_leaf, mtry = m)
  
  # Make predictions on the training set
  predictions <- predict(rf_model, dt_train)
  
  # Calculate the error rate (you may want to use a more appropriate metric)
  error_rate <- mean(predictions != rf_train$Target)
  
  # Store the results
  rf_results <- rbind(rf_results, data.frame(mtry = m, error_rate = error_rate))
}

# Print the results
print(rf_results)

# Create a line plot
ggplot(rf_results, aes(x = mtry, y = error_rate)) +
  geom_line() +
  geom_point() +
  labs(title = "Error Rate vs. mtry",
       x = "mtry",
       y = "Error Rate") +
  theme_minimal()


best_rf_model <- randomForest(Target ~ ., data = rf_train, 
                              ntree = num_trees, nodesize = min_obs_per_leaf, mtry = 2)

# Make predictions on the train set
rf_predictions.train <- predict(best_rf_model, newdata = rf_train)

# Create the confusion matrix
rf_confusion_matrix.train <- confusionMatrix(rf_predictions.train, rf_train$Target)

# Print the confusion matrix
print(rf_confusion_matrix.train)

# Make predictions on the test set
rf_predictions.test <- predict(best_rf_model, newdata = rf_test)

# Create the confusion matrix
rf_confusion_matrix.test <- confusionMatrix(rf_predictions.test, rf_test$Target)

# Print the confusion matrix
print(rf_confusion_matrix.test)

# Plot variable importance
varImpPlot(best_rf_model)

# Create partial dependence plot
partial_plot <- partial(best_rf_model, pred.var = 'Curricular.units.2nd.sem..approved.', data = dt_test)

# Plot the partial dependence plot
plot(partial_plot)



# Gradient Boosting Machines (GBM) Part

students_data_gbm <- data.table(students_data)
# Making the target value numeric to use in gbm
#students_data_gbm$Target <- as.numeric(students_data_gbm$Target)
#students_data_gbm <- transform(students_data_gbm, Target=Target-1)

head(students_data_gbm)
dim(students_data_gbm)

# Remove rows with any NA values
students_data_gbm <- na.omit(students_data_gbm)
validationIndex <- createDataPartition(students_data_gbm$Target, p=0.70, list=FALSE)

gbm_train <- students_data_gbm[validationIndex,] # 70% of data to training
gbm_test <- students_data_gbm[-validationIndex,]


set.seed(10)


n_folds=10

fitControl=trainControl(method = "cv",
                        number = n_folds,
                        classProbs=TRUE,
                        search = "grid",
                        summaryFunction = multiClassSummary)
## gradient boosting
gbmGrid=expand.grid(interaction.depth = c(3, 5), 
                    n.trees = c(1:5)*100, 
                    shrinkage = c(0.05,0.1),
                    n.minobsinnode = 10)
set.seed(1)                        
gbm_fit=train(Target ~ ., data = gbm_train, 
              method = "gbm", 
              trControl = fitControl, metric='AUC',
              tuneGrid = gbmGrid,
              verbose=F) #verbose is an argument from gbm, prints to screen
gbm_fit


plot(gbm_fit)

selected_gbm=tolerance(gbm_fit$results, metric = "AUC", tol = 2, maximize = TRUE)  
gbm_fit$results[selected_gbm,]


noftrees=100
depth=3
learning_rate=0.05

boosting_model=gbm(Target~., data=gbm_train,distribution = 'multinomial', n.trees = noftrees,
                   interaction.depth = depth, n.minobsinnode = 10, shrinkage =learning_rate, cv.folds = 10)

summary(boosting_model)

gbm.perf(boosting_model, method = "cv")

prediction.train <- predict(boosting_model, newdata = gbm_train, type = "response")


predicted_classes <- apply(prediction.train, 1, which.max)


# Create a confusion matrix
conf_matrix <- table(gbm_train$Target, predicted_classes)

# Print the confusion matrix
print(conf_matrix)


### 3) MALWARE CASE

malware_data_path <- paste0(data_path,"Malware/malware.csv")

malware_data <- read.csv(malware_data_path)

dim(malware_data)

# Data Analysis & Preprocessing

str(malware_data)

summary(malware_data)

colSums(sapply(malware_data, is.na))

skim(malware_data)

malware_data <- na.omit(malware_data)
malware_data$Label <- as.factor(malware_data$Label)
#malware_data <- malware_data[!duplicated(malware_data), ]

ggplot(malware_data, aes(x = Label )) +
  geom_bar(stat = "count") +
  labs(title = "Count of Malware Type", x = "Malware Label", y = "Count")


# knn Part 

scaled_malware_data <- malware_data
scaled_malware_data[,-242] <- scale(scaled_malware_data[,-242])
validationIndex <- createDataPartition(scaled_malware_data$Label, p=0.70, list=FALSE)

knn_train <- scaled_malware_data[validationIndex,] # 70% of data to training
knn_test <- scaled_malware_data[-validationIndex,] 

sum(colSums(sapply(knn_train, is.na)))
threshold <- 1000
missing_counts <- colSums(is.na(scaled_malware_data))
columns_to_drop <- names(missing_counts[missing_counts > threshold])
scaled_malware_data <- scaled_malware_data[, !(names(scaled_malware_data) %in% columns_to_drop)]
missing_counts <- colSums(is.na(knn_train))
columns_to_drop <- names(missing_counts[missing_counts > threshold])
knn_train <- knn_train[, !(names(knn_train) %in% columns_to_drop)]
missing_counts <- colSums(is.na(knn_test))
columns_to_drop <- names(missing_counts[missing_counts > threshold])
knn_test <- knn_test[, !(names(knn_test) %in% columns_to_drop)]


cat("Dimension of the main dataset:",dim(scaled_malware_data))
cat("Dimension of the train dataset:",dim(knn_train))
cat("Dimension of the test dataset:",dim(knn_test))

# knn CV

trainControl <- trainControl(method="repeatedcv", number=10, repeats=5)
metric <- "Accuracy"
set.seed(8)

# kNN Part
fit.knn <- train(Label~., data=knn_train, method="knn",
                 metric=metric ,trControl=trainControl)
knn.optimal.k <- fit.knn$bestTune # keep this Initial k for testing with knn() function in next section

print(fit.knn)

plot(fit.knn)

prediction <- predict(fit.knn, newdata = knn_test)
cm_knn <- confusionMatrix(prediction, knn_test$Label)
print(cm_knn)


initial_k <- sqrt(NROW(scaled_malware_data))
initial_k

knn.floor <- knn(train=knn_train[,-200], test=knn_test[,-200], 
                 cl=knn_train$Label, k=floor(initial_k))

# use confusion matrix to calculate accuracy
cm.floor <- confusionMatrix(knn_test$Label,knn.floor)
cm.floor

# optimal k
knn.best <- knn(train=knn_train[,-200], test=knn_test[,-200], cl=knn_train$Label, k= 5)
cf <- confusionMatrix(knn_test$Label,knn.best)
print(cf)

# Train the KNN model
knn.manhattan <- train.kknn(Label ~ ., data = knn_train, ks = 5, scale = TRUE, distance = 1)
predictions.manhattan <- predict(knn.manhattan, newdata = knn_test)

cm.manhattan <- table(predictions.manhattan, knn_test$Label)
print(cm.manhattan)
accuracy.manhattan <- sum(diag(cm.manhattan)) / sum(cm.manhattan)
print(paste("Accuracy for kNN with Manhattan Distance:", round(accuracy.manhattan,5)))


# Decision Tree Part

malware_data_dt <- data.table(malware_data)

dim(malware_data_dt)
# Remove rows with any NA values
malware_data_dt <- na.omit(malware_data_dt)
validationIndex <- createDataPartition(malware_data_dt$Label, p=0.70, list=FALSE)

dt_train <- malware_data_dt[validationIndex,] # 70% of data to training
dt_test <- malware_data_dt[-validationIndex,] 

# Create a data frame to store cross-validation results
cv_results <- data.frame(minbucket = numeric(), accuracy = numeric())

set.seed(10)

# Define a range of minsplit values to try
minbucket_values <- c(1, 5, 10, 15, 20) 

# Perform cross-validation for each minbucket value
for (minbucket_val in minbucket_values) {
  # Create a decision tree model with the current minbucket value
  tree_model <- rpart(Label~.,dt_train,method='class',
                      control=rpart.control(cp=0, minbucket = minbucket_val, minsplit = 2*minbucket_val))
  
  predictions <- predict(tree_model, newdata = dt_train, type = "class")
  correct_predictions <- sum(predictions == dt_train$Label)
  # Calculate accuracy
  accuracy <- round(correct_predictions / nrow(dt_train),5)
  
  # Store the results in the data frame
  cv_results <- rbind(cv_results, data.frame(minbucket = minbucket_val, accuracy = accuracy))
}

# Print the cross-validation results
print(cv_results)

minbucket_val <- 1
best_tree_model <- rpart(Label~.,dt_train,method='class',
                         control=rpart.control(cp=0, minbucket = minbucket_val, minsplit = 2*minbucket_val))

fancyRpartPlot(best_tree_model)

predictions_model <- predict(best_tree_model, newdata = dt_train, type = "class")
cm_train <- confusionMatrix(predictions_model, dt_train$Label)
cm_train
predictions_model.test <- predict(best_tree_model, newdata = dt_test, type = "class")
cm_test <- confusionMatrix(predictions_model.test, dt_test$Label)
cm_test



p1 <- predict(best_tree_model, dt_test, type = 'prob')
p1 <- p1[,2]
r <- multiclass.roc(dt_test$Label, p1, percent = TRUE)
roc <- r[['rocs']]
r1 <- roc[[1]]
plot.roc(r1,
         print.auc=TRUE,
         auc.polygon=TRUE,
         grid=c(0.1, 0.2),
         grid.col=c("green", "red"),
         max.auc.polygon=TRUE,
         auc.polygon.col="lightblue",
         print.thres=TRUE,
         main= 'ROC Curve')


# Random Forest Part


malware_data_rf <- data.table(malware_data)

dim(malware_data_rf)
# Remove rows with any NA values
malware_data_rf <- na.omit(malware_data_rf)
validationIndex <- createDataPartition(malware_data_rf$Label, p=0.70, list=FALSE)

rf_train <- malware_data_rf[validationIndex,] # 70% of data to training
rf_test <- malware_data_rf[-validationIndex,] 


# CV RF

set.seed(14)

# Set the number of trees and nodesize
num_trees <- 500
min_obs_per_leaf <- 5

# Create a grid of mtry values to explore
mtry_values <- c(2, 4, 6, 8, 10, 15, 20)  # Add more values as needed

# Create an empty data frame to store results
rf_results <- data.frame(mtry = numeric(0), error_rate = numeric(0))

# Perform grid search
for (m in mtry_values) {
  # Train the Random Forest model
  rf_model <- randomForest(Label ~ ., data = rf_train, 
                           ntree = num_trees, nodesize = min_obs_per_leaf, mtry = m)
  
  # Make predictions on the training set
  predictions <- predict(rf_model, dt_train)
  
  # Calculate the error rate (you may want to use a more appropriate metric)
  error_rate <- mean(predictions != rf_train$Label)
  
  # Store the results
  rf_results <- rbind(rf_results, data.frame(mtry = m, error_rate = error_rate))
}

# Print the results
print(rf_results)

# Create a line plot
ggplot(rf_results, aes(x = mtry, y = error_rate)) +
  geom_line() +
  geom_point() +
  labs(title = "Error Rate vs. mtry",
       x = "mtry",
       y = "Error Rate") +
  theme_minimal()


best_rf_model <- randomForest(Label ~ ., data = rf_train, 
                              ntree = num_trees, nodesize = min_obs_per_leaf, mtry = 10)

# Make predictions on the train set
rf_predictions.train <- predict(best_rf_model, newdata = rf_train)

# Create the confusion matrix
rf_confusion_matrix.train <- confusionMatrix(rf_predictions.train, rf_train$Label)

# Print the confusion matrix
print(rf_confusion_matrix.train)

# Make predictions on the test set
rf_predictions.test <- predict(best_rf_model, newdata = rf_test)

# Create the confusion matrix
rf_confusion_matrix.test <- confusionMatrix(rf_predictions.test, rf_test$Label)

# Print the confusion matrix
print(rf_confusion_matrix.test)

# Plot variable importance
varImpPlot(best_rf_model)

# Create partial dependence plot
partial_plot <- partial(best_rf_model, pred.var = 'RECEIVE_BOOT_COMPLETED', data = dt_test)

# Plot the partial dependence plot
plot(partial_plot)


# Gradient Boosting Machines (GBM) Part


malware_data_gbm <- data.table(malware_data)
# Making the target value numeric to use in gbm
malware_data_gbm$Label <- as.numeric(malware_data_gbm$Label)
malware_data_gbm <- transform(malware_data_gbm, Label=Label-1)

dim(malware_data_gbm)

# Remove rows with any NA values
malware_data_gbm <- na.omit(malware_data_gbm)
validationIndex <- createDataPartition(malware_data_gbm$Label, p=0.70, list=FALSE)

gbm_train <- malware_data_gbm[validationIndex,] # 70% of data to training
gbm_test <- malware_data_gbm[-validationIndex,] 

# To use in cross validation I have replace "0" with "no" else "yes"
gbm_train_revised <- gbm_train
gbm_test_revised <- gbm_test 

gbm_train_revised$Label <- ifelse(gbm_train_revised$Label == "0", "malware", "goodware")
gbm_train_revised$Label <- as.factor(gbm_train_revised$Label )

gbm_test_revised$Label <- ifelse(gbm_test_revised$Label == "0", "malware", "goodware")
gbm_test_revised$Label <- as.factor(gbm_test_revised$Label )

cat("Dimension of the main dataset:",dim(malware_data_gbm))
cat("Dimension of the train dataset:",dim(gbm_train))
cat("Dimension of the test dataset:",dim(gbm_test))

set.seed(10)

n_folds=10

fitControl=trainControl(method = "repeatedcv",
                        number = n_folds,
                        classProbs=TRUE, summaryFunction=twoClassSummary)
## gradient boosting
gbmGrid=expand.grid(interaction.depth = c(3, 5), 
                    n.trees = c(1:5)*100, 
                    shrinkage = c(0.05,0.1),
                    n.minobsinnode = 10)
set.seed(1)                        
gbm_fit=train(Label ~ ., data = gbm_train_revised, 
              method = "gbm", 
              trControl = fitControl, metric='ROC',
              tuneGrid = gbmGrid,
              verbose=F) #verbose is an argument from gbm, prints to screen
gbm_fit

plot(gbm_fit)


noftrees=500
depth=5
learning_rate=0.1

boosting_model=gbm(Label~., data=gbm_train,distribution = 'bernoulli', n.trees = noftrees,
                   interaction.depth = depth, n.minobsinnode = 10, shrinkage =learning_rate, cv.folds = 10)

summary(boosting_model)

gbm.perf(boosting_model, method = "cv")

prediction.train <- predict(boosting_model, newdata = gbm_train, type = "response")

confusion_data.train <- data.frame(actual = gbm_train$Label,
                                   predicted = ifelse(prediction.train > 0.5,1,0))  # Assuming a threshold of 0.5 for binary classification

# Create a confusion matrix
conf_matrix.train <- confusionMatrix(as.factor(confusion_data.train$predicted), as.factor(confusion_data.train$actual))

print(conf_matrix.train)


prediction.test <- predict(boosting_model, newdata = gbm_test, type = "response")

confusion_data.test <- data.frame(actual = gbm_test$Label,
                                  predicted = ifelse(prediction.test > 0.5,1,0))  # Assuming a threshold of 0.5 for binary classification

# Create a confusion matrix
conf_matrix.test <- confusionMatrix(as.factor(confusion_data.test$predicted), as.factor(confusion_data.test$actual))

print(conf_matrix.test)


### 4) GLIOMA CASE

glioma_data_path <- paste0(data_path,"Glioma/glioma.csv")

glioma_data <- read.csv(glioma_data_path)

head(glioma_data)

# Data Analysis & Preprocessing
dim(glioma_data)

str(glioma_data)

summary(glioma_data)

colSums(sapply(glioma_data, is.na))

skim(glioma_data)


glioma_data$Grade <- as.factor(glioma_data$Grade)

# One hot encoding
glioma_data <- glioma_data %>%
  mutate(IDH1 = as.factor(IDH1)) %>%
  bind_cols(model.matrix(~ IDH1 - 1, data = .)) %>%
  select(-IDH1)  # Drop the original 'IDH1' column


ggplot(glioma_data, aes(x = Grade )) +
  geom_bar(stat = "count") +
  labs(title = "Count of Glioma Grades", x = "Glioma Grades", y = "Count")

ggcorr(glioma_data,
       method = c("pairwise"),
       nbreaks = 6,
       hjust = 0.8,
       label = TRUE,
       label_size = 3,
       color = "grey20")

# knn Part 
dim(scaled_glioma_data)
scaled_glioma_data <- glioma_data
scaled_glioma_data[,-1] <- scale(scaled_glioma_data[,-1])

validationIndex <- createDataPartition(scaled_glioma_data$Grade, p=0.70, list=FALSE)

knn_train <- scaled_glioma_data[validationIndex,] # 70% of data to training
knn_test <- scaled_glioma_data[-validationIndex,] 


cat("Dimension of the main dataset:",dim(scaled_glioma_data))
cat("Dimension of the train dataset:",dim(knn_train))
cat("Dimension of the test dataset:",dim(knn_test))


# knn CV

trainControl <- trainControl(method="repeatedcv", number=10, repeats=5)
metric <- "Accuracy"
set.seed(8)

# kNN Part
fit.knn <- train(Grade~., data=knn_train, method="knn",
                 metric=metric ,trControl=trainControl)
knn.optimal.k <- fit.knn$bestTune # keep this Initial k for testing with knn() function in next section

print(fit.knn)

plot(fit.knn)

prediction <- predict(fit.knn, newdata = knn_test)
cm_knn <- confusionMatrix(prediction, knn_test$Grade)
print(cm_knn)


initial_k <- sqrt(NROW(scaled_glioma_data))
initial_k

knn.floor <- knn(train=knn_train[,-1], test=knn_test[,-1], 
                 cl=knn_train$Grade, k=floor(initial_k))

# use confusion matrix to calculate accuracy
cm.floor <- confusionMatrix(knn_test$Grade,knn.floor)
cm.floor

# optimal k
knn.best <- knn(train=knn_train[,-1], test=knn_test[,-1], cl=knn_train$Grade, k= 5)
cf <- confusionMatrix(knn_test$Grade,knn.best)
print(cf)

# Train the KNN model
knn.manhattan <- train.kknn(Grade ~ ., data = knn_train, ks = 5, scale = TRUE, distance = 1)
predictions.manhattan <- predict(knn.manhattan, newdata = knn_test)

cm.manhattan <- table(predictions.manhattan, knn_test$Grade)
print(cm.manhattan)
accuracy.manhattan <- sum(diag(cm.manhattan)) / sum(cm.manhattan)
print(paste("Accuracy for kNN with Manhattan Distance:", round(accuracy.manhattan,2)))


# Decision Tree Part

glioma_data_dt <- data.table(glioma_data)

head(glioma_data_dt)
dim(glioma_data_dt)
# Remove rows with any NA values
glioma_data_dt <- na.omit(glioma_data_dt)
validationIndex <- createDataPartition(glioma_data_dt$Grade, p=0.70, list=FALSE)

dt_train <- glioma_data_dt[validationIndex,] # 70% of data to training
dt_test <- glioma_data_dt[-validationIndex,] 

# Create a data frame to store cross-validation results
cv_results <- data.frame(minbucket = numeric(), accuracy = numeric())

set.seed(10)

# Define a range of minsplit values to try
minbucket_values <- c(1, 5, 10, 15, 20) 

# Perform cross-validation for each minbucket value
for (minbucket_val in minbucket_values) {
  # Create a decision tree model with the current minbucket value
  tree_model <- rpart(Grade~.,dt_train,method='class',
                      control=rpart.control(cp=0, minbucket = minbucket_val, minsplit = 2*minbucket_val))
  
  predictions <- predict(tree_model, newdata = dt_train, type = "class")
  correct_predictions <- sum(predictions == dt_train$Grade)
  # Calculate accuracy
  accuracy <- round(correct_predictions / nrow(dt_train),5)
  
  # Store the results in the data frame
  cv_results <- rbind(cv_results, data.frame(minbucket = minbucket_val, accuracy = accuracy))
}

# Print the cross-validation results
print(cv_results)

minbucket_val <- 5
best_tree_model <- rpart(Grade~.,dt_train,method='class',
                         control=rpart.control(cp=0, minbucket = minbucket_val, minsplit = 2*minbucket_val))

fancyRpartPlot(best_tree_model)

predictions_model <- predict(best_tree_model, newdata = dt_train, type = "class")
cm_train <- confusionMatrix(predictions_model, dt_train$Grade)
cm_train
predictions_model.test <- predict(best_tree_model, newdata = dt_test, type = "class")
cm_test <- confusionMatrix(predictions_model.test, dt_test$Grade)
cm_test

p1 <- predict(best_tree_model, dt_test, type = 'prob')
p1 <- p1[,2]
r <- multiclass.roc(dt_test$Grade, p1, percent = TRUE)
roc <- r[['rocs']]
r1 <- roc[[1]]
plot.roc(r1,
         print.auc=TRUE,
         auc.polygon=TRUE,
         grid=c(0.1, 0.2),
         grid.col=c("green", "red"),
         max.auc.polygon=TRUE,
         auc.polygon.col="lightblue",
         print.thres=TRUE,
         main= 'ROC Curve')


# Random Forest Part


glioma_data_rf <- data.table(glioma_data)

head(glioma_data_rf)
dim(glioma_data_rf)
# Remove rows with any NA values
glioma_data_rf <- na.omit(glioma_data_rf)
validationIndex <- createDataPartition(glioma_data_rf$Grade, p=0.70, list=FALSE)

rf_train <- glioma_data_rf[validationIndex,] # 70% of data to training
rf_test <- glioma_data_rf[-validationIndex,] 


# CV RF

# Set the number of trees and nodesize
num_trees <- 500
min_obs_per_leaf <- 5

# Create a grid of mtry values to explore
mtry_values <- c(2, 4, 6, 8, 10, 15, 20)  # Add more values as needed

# Create an empty data frame to store results
rf_results <- data.frame(mtry = numeric(0), error_rate = numeric(0))

# Perform grid search
for (m in mtry_values) {
  # Train the Random Forest model
  rf_model <- randomForest(Grade ~ ., data = rf_train, 
                           ntree = num_trees, nodesize = min_obs_per_leaf, mtry = m)
  
  # Make predictions on the training set
  predictions <- predict(rf_model, dt_train)
  
  # Calculate the error rate (you may want to use a more appropriate metric)
  error_rate <- mean(predictions != rf_train$Grade)
  
  # Store the results
  rf_results <- rbind(rf_results, data.frame(mtry = m, error_rate = error_rate))
}

# Print the results
print(rf_results)

# Create a line plot
ggplot(rf_results, aes(x = mtry, y = error_rate)) +
  geom_line() +
  geom_point() +
  labs(title = "Error Rate vs. mtry",
       x = "mtry",
       y = "Error Rate") +
  theme_minimal()


best_rf_model <- randomForest(Grade ~ ., data = rf_train, 
                              ntree = num_trees, nodesize = min_obs_per_leaf, mtry = 8)

# Make predictions on the train set
rf_predictions.train <- predict(best_rf_model, newdata = rf_train)

# Create the confusion matrix
rf_confusion_matrix.train <- confusionMatrix(rf_predictions.train, rf_train$Grade)

# Print the confusion matrix
print(rf_confusion_matrix.train)

# Make predictions on the test set
rf_predictions.test <- predict(best_rf_model, newdata = rf_test)

# Create the confusion matrix
rf_confusion_matrix.test <- confusionMatrix(rf_predictions.test, rf_test$Grade)

# Print the confusion matrix
print(rf_confusion_matrix.test)

# Plot variable importance
varImpPlot(best_rf_model)

# Create partial dependence plot
partial_plot <- partial(best_rf_model, pred.var = 'Age_at_diagnosis', data = dt_test)

# Plot the partial dependence plot
plot(partial_plot)



# Gradient Boosting Machines (GBM) Part

glioma_data_gbm <- data.table(glioma_data)

head(glioma_data_gbm)
dim(glioma_data_gbm)

# Remove rows with any NA values
glioma_data_gbm <- na.omit(glioma_data_gbm)
validationIndex <- createDataPartition(glioma_data_gbm$Grade, p=0.70, list=FALSE)

gbm_train <- glioma_data_gbm[validationIndex,] # 70% of data to training
gbm_test <- glioma_data_gbm[-validationIndex,]


set.seed(10)

n_folds=10

fitControl=trainControl(method = "cv",
                        number = n_folds,
                        classProbs=TRUE,
                        search = "grid",
                        summaryFunction = multiClassSummary)
## gradient boosting
gbmGrid=expand.grid(interaction.depth = c(3, 5), 
                    n.trees = c(1:5)*100, 
                    shrinkage = c(0.05,0.1),
                    n.minobsinnode = 10)
set.seed(1)                        
gbm_fit=train(Grade ~ ., data = gbm_train, 
              method = "gbm", 
              trControl = fitControl, metric='AUC',
              tuneGrid = gbmGrid,
              verbose=F) #verbose is an argument from gbm, prints to screen
gbm_fit


plot(gbm_fit)


noftrees=100
depth=3
learning_rate=0.1

boosting_model=gbm(Grade~., data=gbm_train,distribution = 'multinomial', n.trees = noftrees,
                   interaction.depth = depth, n.minobsinnode = 10, shrinkage =learning_rate, cv.folds = 10)

summary(boosting_model)

gbm.perf(boosting_model, method = "cv")

prediction.train <- predict(boosting_model, newdata = gbm_train, type = "response")


predicted_classes <- apply(prediction.train, 1, which.max)


# Create a confusion matrix
conf_matrix <- table(gbm_train$Grade, predicted_classes)

# Print the confusion matrix
print(conf_matrix)



### 5) WAVE ENERGY CASE

energy_data_path <- paste0(data_path,"WaveEnergy/energy.csv")

energy_data <- read.csv(energy_data_path)

head(energy_data)

# Data Analysis & Preprocessing
dim(energy_data)

str(energy_data)

summary(energy_data)

colSums(sapply(energy_data, is.na))

energy_data <- energy_data[!duplicated(energy_data), ]

# Assuming 'your_data' is your data frame and 'target_variable' is your continuous target variable
hist(energy_data$Total_Power, main = "Histogram of Total Power", xlab = "Total Power")

# Assuming 'your_data' is your data frame and 'target_variable' is your continuous target variable
plot(density(energy_data$Total_Power), main = "Histogram of Total Power", xlab = "Total Power")

# knn Part 
scaled_energy_data <- energy_data
scaled_energy_data[,-149] <- scale(scaled_energy_data[,-149])

validationIndex <- createDataPartition(scaled_energy_data$Total_Power, p=0.70, list=FALSE)

knn_train <- scaled_energy_data[validationIndex,] # 70% of data to training
knn_test <- scaled_energy_data[-validationIndex,] 


cat("Dimension of the main dataset:",dim(scaled_energy_data))
cat("Dimension of the train dataset:",dim(knn_train))
cat("Dimension of the test dataset:",dim(knn_test))


# knn CV
k_values <- c(1, 2, 3, 5, 7, 10, 15, 20)  

# Create an empty data frame to store results
knn_results <- data.frame(k = numeric(0), MSE = numeric(0))

# Perform grid search
for (k in k_values) {
  knn_model <- knn.reg(knn_train[,-149],y = knn_train$Total_Power,k=k)
  
  mse <- round(mean(sqrt((knn_model$pred - knn_train$Total_Power)^2)),3)
  
  # Store the results
  knn_results <- rbind(knn_results, data.frame(k = k, MSE = mse))
}

# Print the results
print(knn_results)

ggplot(knn_results, aes(x = k, y = MSE)) +
  geom_line() +
  geom_point() +
  labs(title = "MSE vs. k",
       x = "k",
       y = "MSE") +
  theme_minimal()

# Test using test data
knn_best <- knn.reg(knn_test[,-149],y = knn_test$Total_Power,k= 5)

predictions <- knn_best$pred
true_values <- knn_test$Total_Power

# Calculate Mean Absolute Error (MAE)
mae <- mean(abs(predictions - true_values))
print(paste("Mean Absolute Error (MAE):", mae))

# Calculate Root Mean Squared Error (RMSE)
rmse <- sqrt(mean((predictions - true_values)^2))
print(paste("Root Mean Squared Error (RMSE):", rmse))

# Calculate R-squared (R²)
sse <- sum((predictions - true_values)^2)
sst <- sum((true_values - mean(true_values))^2)
rsquared <- 1 - (sse / sst)
print(paste("R-squared (R²):", rsquared))


# Decision Tree Part

energy_data_dt <- data.table(energy_data)

dim(energy_data_dt)
# Remove rows with any NA values
energy_data_dt <- na.omit(energy_data_dt)
validationIndex <- createDataPartition(energy_data_dt$Total_Power, p=0.70, list=FALSE)

dt_train <- energy_data_dt[validationIndex,] # 70% of data to training
dt_test <- energy_data_dt[-validationIndex,] 

# Create a data frame to store cross-validation results
cv_results <- data.frame(minbucket = numeric(), RMSE = numeric())

set.seed(10)

# Define a range of minsplit values to try
minbucket_values <- c(1, 5, 10, 15, 20) 

# Perform cross-validation for each minbucket value
for (minbucket_val in minbucket_values) {
  # Create a decision tree regression model with the current minbucket value
  tree_model <- rpart(Total_Power~.,dt_train,method='anova',
                      control=rpart.control(cp=0, minbucket = minbucket_val, minsplit = 2*minbucket_val))
  
  predictions <- predict(tree_model, newdata = dt_train)
  
  true_values <- dt_train$Total_Power
  
  rmse <- round(sqrt(mean((predictions - true_values)^2)),3)
  
  # Store the results in the data frame
  cv_results <- rbind(cv_results, data.frame(minbucket = minbucket_val, RMSE = rmse))
}

# Print the cross-validation results
print(cv_results)

minbucket_val <- 5
best_tree_model <- rpart(Total_Power~.,dt_train,method='anova',
                         control=rpart.control(cp=0, minbucket = minbucket_val, minsplit = 2*minbucket_val))

# Make predictions on the test set
predictions <- predict(best_tree_model, newdata = dt_test)

# True values from the test set
true_values <- dt_test$Total_Power

# Calculate Mean Absolute Error (MAE)
mae <- mean(abs(predictions - true_values))
print(paste("Mean Absolute Error (MAE):", mae))

# Calculate Root Mean Squared Error (RMSE)
rmse <- round(sqrt(mean((predictions - true_values)^2)),3)
print(paste("Root Mean Squared Error (RMSE):", rmse))

# Calculate R-squared (R²)
sse <- sum((predictions - true_values)^2)
sst <- sum((true_values - mean(true_values))^2)
rsquared <- 1 - (sse / sst)
print(paste("R-squared (R²):", rsquared))


# Create a scatterplot
plot(true_values, predictions, 
     main = "Predictions vs Actual Data",
     xlab = "True Values",
     ylab = "Predicted Values",
     pch = 16,  # Set the point character
     col = "blue"  # Set the point color
)

# Add a diagonal line for reference
abline(0, 1, col = "red", lty = 2)

# Add a legend
legend("topright", legend = "Diagonal Line (Reference)", col = "red", lty = 2, cex = 0.8)



# Random Forest Part


energy_data_rf <- data.table(energy_data)

dim(energy_data_rf)
# Remove rows with any NA values
validationIndex <- createDataPartition(energy_data_rf$Total_Power, p=0.70, list=FALSE)

rf_train <- energy_data_rf[validationIndex,] # 70% of data to training
rf_test <- energy_data_rf[-validationIndex,] 


# CV RF
set.seed(12)
# Set the number of trees and nodesize
num_trees <- 500
min_obs_per_leaf <- 5

# Create a grid of mtry values to explore
mtry_values <- c(2, 4, 6, 8, 10)  # Add more values as needed

# Create an empty data frame to store results
rf_results <- data.frame(mtry = numeric(), RMSE = numeric())

# Perform grid search
for (m in mtry_values) {
  # Train the Random Forest model
  rf_model <- randomForest(Total_Power ~ ., data = rf_train, 
                           ntree = num_trees, nodesize = min_obs_per_leaf, mtry = m)
  
  # Make predictions on the training set
  predictions <- predict(rf_model, newdata = rf_train)
  
  true_values <- rf_train$Total_Power
  
  rmse <- round(sqrt(mean((predictions - true_values)^2)),3)
  
  # Store the results in the data frame
  rf_results <- rbind(rf_results, data.frame(mtry = m, RMSE = rmse))
}

# Print the results
print(rf_results)

# Create a line plot
ggplot(rf_results, aes(x = mtry, y = RMSE)) +
  geom_line() +
  geom_point() +
  labs(title = "RMSE vs. mtry",
       x = "mtry",
       y = "RMSE") +
  theme_minimal()


best_rf_model <- randomForest(Total_Power ~ ., data = rf_train, 
                              ntree = num_trees, nodesize = min_obs_per_leaf, mtry = 10)

# Make predictions on the test set
predictions <- predict(best_rf_model, newdata = rf_test)

# True values from the test set
true_values <- rf_test$Total_Power

# Calculate Mean Absolute Error (MAE)
mae <- mean(abs(predictions - true_values))
print(paste("Mean Absolute Error (MAE):", mae))

# Calculate Root Mean Squared Error (RMSE)
rmse <- round(sqrt(mean((predictions - true_values)^2)),3)
print(paste("Root Mean Squared Error (RMSE):", rmse))

# Calculate R-squared (R²)
sse <- sum((predictions - true_values)^2)
sst <- sum((true_values - mean(true_values))^2)
rsquared <- 1 - (sse / sst)
print(paste("R-squared (R²):", rsquared))

# Plot variable importance
varImpPlot(best_rf_model)

# Create partial dependence plot
partial_plot <- partial(best_rf_model, pred.var = 'qW', data = dt_test)

# Plot the partial dependence plot
plot(partial_plot)


# Gradient Boosting Machines (GBM) Part

energy_data_gbm <- data.table(energy_data)

dim(energy_data_gbm)

# Remove rows with any NA values
validationIndex <- createDataPartition(energy_data_gbm$Total_Power, p=0.70, list=FALSE)

gbm_train <- energy_data_gbm[validationIndex,] # 70% of data to training
gbm_test <- energy_data_gbm[-validationIndex,]


set.seed(10)

n_folds=10

fitControl=trainControl(method = "cv",
                        number = n_folds)
## gradient boosting
gbmGrid=expand.grid(interaction.depth = c(3, 5), 
                    n.trees = c(1:3)*100, 
                    shrinkage = c(0.05),
                    n.minobsinnode = 10)
set.seed(1)                        
gbm_fit=train(Total_Power ~ ., data = gbm_train, 
              method = "gbm", 
              trControl = fitControl, metric='RMSE',
              tuneGrid = gbmGrid,
              verbose=F) #verbose is an argument from gbm, prints to screen
gbm_fit


plot(gbm_fit)


noftrees=100
depth=3
learning_rate=0.1

boosting_model=gbm(Total_Power~., data=gbm_train,distribution = 'gaussian', n.trees = noftrees,
                   interaction.depth = depth, n.minobsinnode = 10, shrinkage =learning_rate, cv.folds = 10)

summary(boosting_model)

gbm.perf(boosting_model, method = "cv")

prediction.test <- predict(boosting_model, newdata = gbm_test, type = "response")

# True values from the test set
true_values <- gbm_test$Total_Power

# Calculate Mean Absolute Error (MAE)
mae <- mean(abs(predictions - true_values))
print(paste("Mean Absolute Error (MAE):", mae))

# Calculate Root Mean Squared Error (RMSE)
rmse <- round(sqrt(mean((predictions - true_values)^2)),3)
print(paste("Root Mean Squared Error (RMSE):", rmse))