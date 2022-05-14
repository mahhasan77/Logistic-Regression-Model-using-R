##Assignment 3: Predictive modelling##
##Student ID: 20047638
##User ID: ZM21AAS

## Task 1:
# Loading dataset Cancer from the library Survival
library(survival)
library(VIM)
data("cancer")
data1 <- cancer
data1

# Modifying the dataset to convert the status column to binary column(0,1)
levels(data1$status)<- c(0,1) # binary outcome
data1<- data.frame(sapply(data1, as.numeric)) #converting from factors to number
data1$status<- data1$status-1
head(data1)
data1

# Summary of the dataset chosen
summary(data1)

library(Hmisc)
describe(data1)

# Imputing the missing data using KNN imputation
data1 <- kNN(data1, variable = c("inst") ,k=15)
data1 <- kNN(data1, variable = c("ph.karno") ,k=15)
data1 <- kNN(data1, variable = c("meal.cal") ,k=15)
data1 <- kNN(data1, variable = c("wt.loss") ,k=15)
data1 <- kNN(data1, variable = c("ph.ecog") ,k=15)
data1 <- kNN(data1, variable = c("pat.karno") ,k=15)
data1 <- data1[,1:10]
data1

# Summary of the dataset chosen after imputations
summary(data1)

library(Hmisc)
describe(data1)

## Task 2:
# Normalizing using Z-Score transformation
# Predictor Columns are set as the rest of the columns in dataset Cancer
library(robustHD)
# Standardizing the columns
data1$inst <- (data1$inst - mean(data1$inst, na.rm = TRUE)) / sd(data1$inst, na.rm = TRUE)
data1$time <- (data1$time - mean(data1$time, na.rm = TRUE)) / sd(data1$time, na.rm = TRUE)
data1$sex <- (data1$sex - mean(data1$sex, na.rm = TRUE)) / sd(data1$sex, na.rm = TRUE)
data1$age <- (data1$time - mean(data1$time, na.rm = TRUE)) / sd(data1$time, na.rm = TRUE)
data1$ph.ecog <- (data1$ph.ecog - mean(data1$ph.ecog, na.rm = TRUE)) / sd(data1$ph.ecog, na.rm = TRUE)
data1$ph.karno <- (data1$ph.karno - mean(data1$ph.karno, na.rm = TRUE)) / sd(data1$ph.karno, na.rm = TRUE)
data1$pat.karno <- (data1$pat.karno - mean(data1$pat.karno, na.rm = TRUE)) / sd(data1$pat.karno, na.rm = TRUE)
data1$meal.cal <- (data1$meal.cal - mean(data1$meal.cal, na.rm = TRUE)) / sd(data1$meal.cal, na.rm = TRUE)
data1$wt.loss <- (data1$wt.loss - mean(data1$wt.loss, na.rm = TRUE)) / sd(data1$wt.loss, na.rm = TRUE)

# Performing the Shapiro-Wilk test for each standardized column
shapiro.test(data1$inst)
shapiro.test(data1$time)
shapiro.test(data1$sex)
shapiro.test(data1$age)
shapiro.test(data1$ph.ecog)
shapiro.test(data1$ph.karno)
shapiro.test(data1$pat.karno)
shapiro.test(data1$meal.cal)
shapiro.test(data1$wt.loss)

## Task 3:
# Splitting the dataset into training and test data
#loading package
library(caTools)
set.seed(6666)
#use Split Ratio for 70%:30% splitting
random_sample = sample.split(data1,SplitRatio = 0.7)
#subset the dataset into Training data(70%)
train_data = subset(data1, random_sample == TRUE)
#subset the dataset into Test data(30%)
test_data = subset(data1, random_sample == FALSE)
train_data
test_data

## Task 4:
# Defining Functions 
# Feature Selection on Training Data
FSCR = function(X, Y, k) # X - matrix with predictors, Y - binary outcome, k top candidates
{
  J<- rep(NA, ncol(X))
  names(J)<- colnames(X)
  for (i in 1:ncol(X))
  {
    X1<- X[which(Y==0),i]
    X2<- X[which(Y==1),i]
    mu1<- mean(X1); mu2<- mean(X2); mu<- mean(X[,i])
    var1<- var(X1); var2<- var(X2)
    n1<- length(X1); n2<- length(X2)
    J[i]<- (n1*(mu1-mu)^2+n2*(mu2-mu)^2)/(n1*var1+n2*var2)
  }
  J<- sort(J, decreasing=TRUE)[1:k]
  return(list(score=J))
}

TSCR = function(X, Y, k) # X - matrix with predictors, Y - binary outcome, k top candidates
{
  J<- rep(NA, ncol(X))
  names(J)<- colnames(X)
  for (i in 1:ncol(X))
  {
    X1<- X[which(Y==0),i]
    X2<- X[which(Y==1),i]
    mu1<- mean(X1); mu2<- mean(X2)
    var1<- var(X1); var2<- var(X2)
    n1<- length(X1); n2<- length(X2)
    J[i]<- (mu1-mu2)/sqrt(var1/n1+var2/n2)
  }
  J<- sort(J, decreasing=TRUE)[1:k]
  return(list(score=J))
}

WLCX = function(X, Y, k) # X - matrix with predictors, Y - binary outcome, k top candidates
{
  J<- rep(NA, ncol(X))
  names(J)<- colnames(X)
  for (i in 1:ncol(X))
  {
    X_rank<- apply(data.matrix(X[,i]), 2, function(c) rank(c))
    X1_rank<- X_rank[which(Y==0)]
    X2_rank<- X_rank[which(Y==1)]
    mu1<- mean(X1_rank); mu2<- mean(X2_rank); mu<- mean(X_rank)
    n1<- length(X1_rank); n2<- length(X2_rank); N<- length(X_rank)
    num<- (n1*(mu1-mu)^2+ n2*(mu2-mu)^2)
    denom<- 0
    for (j in 1:n1)
      denom<- denom+(X1_rank[j]-mu)^2
    for (j in 1:n2)
      denom<- denom+(X2_rank[j]-mu)^2
    J[i]<- (N-1)*num/denom
  }
  J<- sort(J, decreasing=TRUE)[1:k]
  return(list(score=J))
}

# Separating the predictors and binary outcome column (Status)
Data_Y<- train_data$status
Data_X<- train_data[,-c(3)]
K<- 3 # top 3 features

# Performing Fisher Score Test
FSCR(Data_X, Data_Y, K)
# Performing Wilcoxon Score Test
WLCX(Data_X, Data_Y, K)
# Performing T Score Test
TSCR(Data_X, Data_Y, K)

# reducing the training set based on the feature selection methods
reduced_train <- train_data[,c("status","sex", "ph.ecog","ph.karno", "pat.karno")]
# reducing the test set based on the feature selection methods
reduced_test <- test_data[,c("status","sex", "ph.ecog", "ph.karno", "pat.karno")]
reduced_train
reduced_test

## Task 5
# Logistic regression for training dataset with all the predictors included
mod_train<- glm(status~., data=train_data, family="binomial")
summary(mod_train)
# Logistic regression for reduced training data with selected features
mod_redtrain<- glm(status~., data=reduced_train, family="binomial")
summary(mod_redtrain)

## Task 6
# Applying the trained models on our test data
library(Hmisc); library(ggplot2); library(gridExtra)
pr_mod1<- predict(mod_train, test_data, type="response")
pr_mod2<- predict(mod_redtrain, reduced_test, type="response")
pr_mod1
pr_mod2

# ROC-analysis
library(ROCR)
pred1.obj <- prediction(predictions = pr_mod1, labels = test_data$status)
pred2.obj <- prediction(predictions = pr_mod2,  labels = reduced_test$status)
perf1 <- performance(pred1.obj, measure="tpr", x.measure="fpr")
perf2 <- performance(pred2.obj, measure="tpr", x.measure="fpr")
perf1
perf2

# Area Under the curve for test data
auc_test<- somers2(pr_mod1, test_data$status)[1]
auc_redtest<- somers2(pr_mod2, reduced_test$status)[1]
auc_test
auc_redtest

# Plotting the graphs for visual representation
plot(perf1, lty = 1, col = "dark green", lwd = 3, cex.lab = 1.2)
plot(perf2,  lty = 1, col = "blue",  lwd = 3, add = T)
legend(0.45, 0.2, c("Model1 - Test Data", "Model2 - Reduced Test Data"),
       lty = c(1,1), col = c("dark green","blue"), lwd = c(3,3), cex = 0.8, bg = "gray90")
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted", lwd = par("lwd"), equilogs = TRUE)
abline(0, 1, col = "gray30", lwd = 1, lty = 2)



