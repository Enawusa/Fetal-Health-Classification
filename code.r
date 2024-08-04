
###########################
library(class)
library(MASS)
library(randomForest)
library(ISLR)
library(BART)
library(kernlab)
library(caret)
library(e1071)
library(corrplot)
library(h2o)
library(rpart)
library(car)
library(tree)
library(Metrics)
library(vcdExtra)
library(pROC)
library(ROCR)

############################################## Data 
fetal = read.csv(file = "fetal_health.csv", header = TRUE)
fetal$fetal_health = as.factor(fetal$fetal_health)

features <- setdiff(colnames(fetal),'fetal_health')

plot(fetal$fetal_health)
summary(fetal$fetal_health)

options(repr.plot.width = 14, repr.plot.height = 8) # => bigger plots for the following
cor_features_SP <- cor(fetal[,features], method='spearman')
corrplot(cor_features_SP, tl.cex=0.6,type = "lower",title = "Correlation Plot", method = "number")

###########################################
set.seed(88888)
split = createDataPartition(fetal$fetal_health, p=0.75, list=FALSE)
Train_Fetal = fetal[split,]
test_Fetal = fetal[-split,]
test.y = test_Fetal$fetal_health



##########################################classification trees ------------------- Decision trees
set.seed(88888)
library(tree)

model.tree<-tree(as.factor(fetal_health)~.,data=Train_Fetal)
summary(model.tree)  
plot(model.tree)
text(model.tree)

model.tree


###prediction###########
pred.tree<-predict(model.tree,newdata=test_Fetal,type="class")

mean(pred.tree!=test.y)
table(test.y,pred.tree)
conf_mat = confusionMatrix(pred.tree,test.y)
conf_mat

confidenceInterval = confint(Kappa(conf_mat$table))
confidenceInterval

#################tree pruning###########
cv.p<-cv.tree(model.tree,FUN=prune.misclass)
cv.p

plot(cv.p$size, cv.p$dev, type="b")

#####pruning the tree###########
prune.tree<-prune.misclass(model.tree,best=11)
plot(prune.tree)
text(prune.tree)

prune.tree.pred<-predict(prune.tree,newdata=test_Fetal,type="class")
conf_mat2 = confusionMatrix(prune.tree.pred,test.y)
print(conf_mat2)

###############bagging
set.seed(223)
bagging.model = randomForest(as.factor(fetal_health)~.,data=Train_Fetal,mtry=21,importance=TRUE)
bagging.model


####prediction############
pred.bag<-predict(bagging.model,newdata=test_Fetal)

conf_mat = confusionMatrix(pred.bag,test.y)
print(conf_mat)

importance(bagging.model)           
varImpPlot(bagging.model) 

confidenceInterval = confint(Kappa(conf_mat$table))
confidenceInterval

##################################################### RandomForest
set.seed(223)
rf_model=randomForest(as.factor(fetal_health)~.,data=Train_Fetal,importance=TRUE)
rf_model

plot(rf_model)

####prediction############
pred.rf<-predict(rf_model,test_Fetal)

mean(pred.rf!=test.y)
conf_mat = confusionMatrix(pred.rf,test.y)
print(conf_mat)

importance(rf_model)           
varImpPlot(rf_model)    

################################################ Set up the control parameters for cross-validation
set.seed(223)
ctrl = trainControl(method = "cv", number = 10)  # You can adjust the number of folds

# Train the Random Forest model using cross-validation
rf_model.cv = train(fetal_health~.,data = Train_Fetal,method = "rf",trControl = ctrl) # mtry is 11

# Access the results
print(rf_model.cv)

# Plot the cross-validated performance
plot(rf_model.cv)

####prediction############
pred.rf.cv<-predict(rf_model.cv,test_Fetal)

mean(pred.rf.cv!=test.y)
conf_mat = confusionMatrix(pred.rf.cv,test.y)
print(conf_mat)

#################################### tuning the model
k =tuneRF(Train_Fetal[,-22],Train_Fetal[,22],stepFactor=0.5,plot=TRUE,ntreeTRY=250,trace=TRUE)
print(k)


set.seed(333)
rf_model.tune = randomForest(fetal_health~.,data=Train_Fetal,ntree=300,mtry=8,importance=TRUE,proximity=TRUE)
print(rf_model.tune)
plot(rf_model.tune)

####prediction############
pred.rf.tune = predict(rf_model.tune,test_Fetal)

mean(pred.rf.tune!=test.y)
conf_mat = confusionMatrix(pred.rf.tune,test.y)
print(conf_mat)

confidenceInterval = confint(Kappa(conf_mat$table))
confidenceInterval

hist(treesize(rf_model),col="gray")

#################svm linear kernel ###########
set.seed(223)
model.svm.li = svm(fetal_health~., data = Train_Fetal, kernel = "linear",cost = 10)
summary(model.svm)

####prediction#################################################################
pred.svm.li = predict(model.svm.li,test_Fetal)

mean(pred.svm.li!=test.y)
conf_mat = confusionMatrix(pred.svm.li,test.y)
print(conf_mat)

#################svm radial kernel ###########
set.seed(223)
model.svm.Ra = svm(fetal_health~., data = Train_Fetal, kernel = "radial", cost =10)


####prediction###
pred.svm.Ra = predict(model.svm.Ra,test_Fetal)

mean(pred.svm.Ra!=test.y)
conf_mat = confusionMatrix(pred.svm.Ra,test.y)
print(conf_mat)

confidenceInterval = confint(Kappa(conf_mat$table))
confidenceInterval

#################svm sigmoid kernel ###########
set.seed(223)
model.svm.sg = svm(fetal_health~., data = Train_Fetal, kernel = "sigmoid")
print(model.svm.sg)

####prediction#################################################################
pred.svm.sg = predict(model.svm.sg,test_Fetal)

mean(pred.svm.sg!=test.y)
conf_mat = confusionMatrix(pred.svm.sg,test.y)
print(conf_mat)

#################svm polynomial kernel ###########
set.seed(223)
model.svm.py = svm(fetal_health~., data = Train_Fetal, kernel = "polynomial")
summary(model.svm.py)


####prediction#################################################################
pred.svm.py = predict(model.svm.py,test_Fetal)

mean(pred.svm.py!=test.y)
conf_mat = confusionMatrix(pred.svm.py,test.y)
print(conf_mat)

#################tuning the best model which is the radial kernel
set.seed(223)
tune.out=tune(svm, fetal_health~.,data=Train_Fetal, kernel ="radial", ranges =list(cost=c(0.001 , 0.01, 0.1, 1,5,10,100)))
summary(tune.out)

plot(tune.out)
bestmod =tune.out$best.model

ypred.svm.tune=predict(bestmod,test_Fetal)
table(predict=ypred.svm.tune, truth= test.y)
mean(ypred.svm.tune!=test.y)
conf_mat = confusionMatrix(ypred.svm.tune,test.y)
print(conf_mat)

################################################ADABOOST
set.seed(223)
# Convert the response variable to a factor
Train_Fetal$fetal_health = as.factor(Train_Fetal$fetal_health)

# Fit AdaBoost model
model.adaboost = boosting(fetal_health ~ ., data = Train_Fetal, mfinal = 500,boos = TRUE)
summary(model.adaboost)


# Make predictions on new data
predictions = predict.boosting(model.adaboost, newdata = test_Fetal)

# To get the predicted class labels
predicted_labels <- as.factor(predictions$class)

# Evaluate the model
conf_matrix <- table(predicted_labels, test.y)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

# Print the confusion matrix and accuracy
print(conf_matrix)
cat("Accuracy:", accuracy, "\n")

conf_mat = confusionMatrix(predicted_labels,test.y)
print(conf_mat)

confidenceInterval = confint(Kappa(conf_mat$table))
confidenceInterval

########################################## naivebayes
set.seed(223)
library(naivebayes)

# Fit Naive Bayes model
model.nb = naive_bayes(fetal_health ~ ., data = Train_Fetal)

# Make predictions on new data
pred.naive = predict(model.nb, newdata = test_Fetal)

# To get the predicted class labels
predicted.naive <- as.factor(pred.naive)

# Evaluate the model
# Assuming you have a test set named test_data
conf_matrix <- table(predicted.naive,test.y)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

# Print the confusion matrix and accuracy
print(conf_matrix)
cat("Accuracy:", accuracy, "\n")

conf_mat = confusionMatrix(predicted.naive,test.y)
print(conf_mat)

########################################################### Using cross validation for naivesbayes
set.seed(223)
# Set up the control parameters for cross-validation
ctrl = trainControl(method = "cv", number = 10)  # You can adjust the number of folds

# Train Naive Bayes model
nb_model = train(fetal_health~., data = Train_Fetal, method = "naive_bayes", trControl = ctrl)

summary(nb_model)
# Print the model
print(nb_model)

# Make predictions on new data
predict.cv.naive = predict(nb_model, newdata = test_Fetal)

# Evaluate the model
conf_mat <- confusionMatrix(predict.cv.naive, test.y)
print(conf_mat)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

# Print the confusion matrix and accuracy
print(conf_matrix)
cat("Accuracy:", accuracy, "\n")
conf_mat <- confusionMatrix(predict.cv.naive, test.y)
print(conf_mat)

confidenceInterval = confint(Kappa(conf_mat$table))
confidenceInterval

###########################################knnn
set.seed(223)
library(class)

# Fit KNN model
knn_model = knn(train = Train_Fetal[,-22],test = test_Fetal[, -22],cl = Train_Fetal[,22],
                 k = 3)  # Specify the number of neighbors

# Evaluate the model
conf_matrix <- table(predicted = knn_model, actual = test.y)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

# Print the confusion matrix and accuracy
print(conf_matrix)
cat("Accuracy:", accuracy, "\n")

##############################
conf_mat= confusionMatrix(knn_model,test.y)
print(conf_mat)

ctrl = trainControl(method = "cv", number = 10)  # You can adjust the number of folds

# Train Naive Bayes model
nb_model = train(fetal_health~., data = Train_Fetal, method = "knn", trControl = ctrl,k=3)

summary(nb_model)

# Make predictions on new data
predict.cv.naive = predict(nb_model, newdata = test_Fetal)

# Evaluate the model
conf_mat <- confusionMatrix(predict.cv.naive, test.y)
print(conf_mat)

############################### XGBOOST
library(xgboost)
set.seed(223)

X_train = data.matrix(Train_Fetal[,-22])               
y_train = Train_Fetal[,22]                               

# independent and dependent variables for test
X_test = data.matrix(test_Fetal[,-22])                   
y_test = test_Fetal[,22]



# convert the train and test data into xgboost matrix type.
xgboost_train = xgb.DMatrix(data=X_train, label=y_train)
xgboost_test = xgb.DMatrix(data=X_test, label=y_test)

###################Model
# train a model using our training data
model.xg = xgboost(data = xgboost_train,max.depth=3,nrounds=100)                              # max number of boosting iterations

summary(model)

#use model to make predictions on test data
pred_test = predict(model, xgboost_test)

pred.xg = as.factor((levels(y_test))[round(pred_test)])

conf_mat = confusionMatrix(y_test, pred.xg)
print(conf_mat)

confidenceInterval = confint(Kappa(conf_mat$table))
confidenceInterval

####################################### lda
set.seed(223)
train.lda = Train_Fetal[,-c(2,3,6,7,11,12)]
test.lda = test_Fetal[,-c(2,3,6,7,11,12)]
test.Y = test.lda$fetal_health
lda.fit=lda(fetal_health~., data = train.lda)
lda.fit


lda.pred=predict(lda.fit,test.lda)
lda.pred$class

mean(test.Y!=lda.pred$class)

table(lda.pred$class,test.Y) 
conf_mat = confusionMatrix(test.Y,lda.pred$class)
print(conf_mat)

confidenceInterval = confint(Kappa(conf_mat$table))
confidenceInterval

############################gbm
set.seed(223)
library(gbm)
model.gbm = gbm(fetal_health~., data = Train_Fetal, distribution = "multinomial",n.trees = 300)
plot(model.gbm)

gbm.pred = predict.gbm(model.gbm,test_Fetal, n.trees = 300, type = "response")

predicted_classes.gbm = colnames(gbm.pred)[apply(gbm.pred, 1, which.max)]
conf_mat = confusionMatrix(test.y,factor(predicted_classes.gbm))
print(conf_mat)

#####################tuning the model
# Define a tuning grid
set.seed(223)
tune_grid <- expand.grid(n.trees = c(50, 100, 150),interaction.depth = c(3, 5, 7),shrinkage = c(0.01, 0.1, 0.2),n.minobsinnode = c(5, 10, 15) )

# Set up the control parameters for cross-validation
ctrl <- trainControl(method = "cv", number = 10)

# Tune the GBM model
gbm_tune = train(fetal_health~.,data = Train_Fetal,method = "gbm",trControl = ctrl,tuneGrid = tune_grid)

# Print the best tuning parameters
print(gbm_tune)
summary(gbm_tune)
plot(gbm_tune)


# Access the best model
bestmod = gbm_tune$finalModel

plot(bestmod)
gbm.pred.tune = predict(gbm_tune, newdata = test_Fetal, type = "raw")
gbm.pred.tune 
conf_mat = confusionMatrix(test.y,gbm.pred.tune)
print(conf_mat)

confidenceInterval = confint(Kappa(conf_mat$table))
confidenceInterval

#################################################################
# Assuming you have a data frame with your model names, Kappa values, and confidence intervals
# Replace the values below with your actual data
par(mfrow = c(1, 2))
model_data <- data.frame(
  Model = c("Decision tree", "Bagging", "Random forest", "SVM-Radial", "ADABOOST", 
            "Naïve bayes", "KNN", "XGBOOST", "LDA", "GBM"),
  Kappa = c(0.772, 0.833, 0.843, 0.828, 0.836, 0.518, 0.746, 0.841, 0.573, 0.873),
  Lower_CI = c(0.7066547, 0.7775912, 0.7882941, 0.7712314, 0.7807496, 0.4297575, 0.6778224, 0.7865442, 0.4955130, 0.8237526),
  Upper_CI = c(0.8374695, 0.8890044, 0.8977376, 0.8840006, 0.8906701, 0.6058333, 0.8130310, 0.8947355, 0.6509525, 0.9216412)
)

library(ggplot2)
# Plot with confidence intervals
plot1 = ggplot(model_data, aes(x = Kappa, y = Model, color = Model)) +
  geom_point() +
  geom_errorbarh(aes(xmin = Lower_CI, xmax = Upper_CI), height = 0.2) +
  geom_text(aes(label = round(Kappa, 3)), hjust = -0.2) +
  labs(title = "Model Kappa Comparison with 95% Confidence Intervals",
       x = "Kappa",
       y = "Model") +
  theme_minimal()

# ##################################Data

model_data <- data.frame(
  Model = c("Decision tree", "Bagging", "Random forest", "SVM-Radial", "ADABOOST", 
            "Naïve bayes", "KNN", "XGBOOST", "LDA", "GBM"),
accuracy <- c(0.9208,0.9415,0.9453,0.9396,0.9415,0.8566,0.9113,0.9434,0.8566,0.9547),
  Lower_CI = c(0.8944,0.918,0.9224,0.9158,0.918,0.8238, 0.8838,0.9202,0.8238,0.9334),
  Upper_CI = c(0.9423,0.9599,0.9631,0.9583, 0.9599,0.8853,0.9341,0.9615,0.8853,0.9708)
)
# Plot with confidence intervals
plot2 = ggplot(model_data, aes(x = accuracy, y = Model, color = Model)) +
  geom_point() +
  geom_errorbarh(aes(xmin = Lower_CI, xmax = Upper_CI), height = 0.2) +
  geom_text(aes(label = round(accuracy, 3)), hjust = -0.2) +
  labs(title = "Model Accuracy Comparison with 95% Confidence Intervals",
       x = "Accuracy",
       y = "Model") +
  theme_minimal()

# Arrange the plots side by side
grid.arrange(plot1, plot2, ncol = 2)


par(mfrow = c(5, 5))
################################# Roc and AUC plots
roc.tree = multiclass.roc(as.numeric(test.y),as.numeric(pred.tree) ,plot=TRUE, main = "decision tree")
auc.tree = auc(roc.tree)
auc.tree

roc.tree.prune = multiclass.roc(as.numeric(test.y),as.numeric(prune.tree.pred) ,plot=TRUE, main = "Pruned tree")
auc.tree.prune = auc(roc.tree)
auc.tree.prune

roc.bagged = multiclass.roc(as.numeric(test.y),as.numeric(pred.bag) ,plot=TRUE, main = "Bagged tree")
auc.bag = auc(roc.bagged)
auc.bag

roc.rf = multiclass.roc(as.numeric(test.y),as.numeric(pred.bag) ,plot=TRUE, main = "Random Forest")
auc.rf = auc(roc.rf)
auc.rf

roc.rf.cv = multiclass.roc(as.numeric(test.y),as.numeric(pred.rf.cv) ,plot=TRUE, main = "CV Random Forest")
auc.rf.cv = auc(roc.rf.cv)
auc.rf.cv

roc.rf.tune = multiclass.roc(as.numeric(test.y),as.numeric(pred.rf.tune) ,plot=TRUE, main = "Tune Random Forest")
auc.rf.tune = auc(roc.rf.tune)
auc.rf.tune

|par(mfrow = c(5, 5))
roc.svm.li = multiclass.roc(as.numeric(test.y),as.numeric(pred.svm.li) ,plot=TRUE, main = "linear kernel")
auc.svm.li = auc(roc.svm.li)
roc.svm.li

roc.svm.Ra = multiclass.roc(as.numeric(test.y),as.numeric(pred.svm.Ra) ,plot=TRUE, main = "Radial kernel")
auc.svm.Ra = auc(roc.svm.Ra)
auc.svm.Ra

roc.svm.sg = multiclass.roc(as.numeric(test.y),as.numeric(pred.svm.sg) ,plot=TRUE, main = "Sigmoid kernel")
auc.svm.sg = auc(roc.svm.sg)
auc.svm.sg

roc.svm.py = multiclass.roc(as.numeric(test.y),as.numeric(pred.svm.py) ,plot=TRUE, main = "Polynomial kernel")
auc.svm.py = auc(roc.svm.py)
auc.svm.py

roc.svm.Ra.tune = multiclass.roc(as.numeric(test.y),as.numeric(ypred.svm.tune) ,plot=TRUE, main = "Tuned Radial kernel")
auc.svm.Ra.tune = auc(roc.svm.Ra.tune)
auc.svm.Ra.tune

roc.ada = multiclass.roc(as.numeric(test.y),as.numeric(predicted_labels) ,plot=TRUE, main = "Adaboost")
auc.ada = auc(roc.ada)
auc.ada

par(mfrow = c(5, 5))
roc.naive = multiclass.roc(as.numeric(test.y),as.numeric(predicted.naive) ,plot=TRUE, main = "Naive Bayes")
auc.naive = auc(roc.naive)
auc.naive

roc.naive.cv = multiclass.roc(as.numeric(test.y),as.numeric(predict.cv.naive) ,plot=TRUE, main = "CV Naive Bayes")
auc.naive.cv = auc(roc.naive.cv)
auc.naive.cv

roc.knn = multiclass.roc(as.numeric(test.y),as.numeric(knn_model) ,plot=TRUE, main = "KNN")
auc.knn = auc(roc.knn)
auc.knn

roc.xg = multiclass.roc(as.numeric(test.y),as.numeric(pred.xg) ,plot=TRUE, main = "Xgboost")
auc.xg = auc(roc.xg)
auc.xg

roc.lda = multiclass.roc(as.numeric(test.y),as.numeric(lda.pred$class) ,plot=TRUE, main = "LDA")
auc.lda = auc(roc.lda)
auc.lda

roc.ggm = multiclass.roc(as.numeric(test.y),as.numeric(predicted_classes.gbm) ,plot=TRUE, main = "GBM")
auc.gbm = auc(roc.ggm)
auc.gbm

par(mfrow = c(5, 5))
roc.gbm.tune = multiclass.roc(as.numeric(test.y),as.numeric(gbm.pred.tune) ,plot=TRUE, main = "Tune GBM")
auc.gbm.tune = auc(roc.gbm.tune)
auc.gbm.tune
