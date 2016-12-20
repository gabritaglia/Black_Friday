
set.seed(123)
rm(list=ls())
gc(verbose = T)

library(caret)
library(plyr); library(dplyr)
#----------------
#   DATA IMPORT
#----------------
dtrain<-read.csv(file = "train.csv",header = T)

names(dtrain)


#----------------
#   DATA MUNGING
#----------------
data_preparation<-function(df){
  df$User_ID=factor(df$User_ID)
  df$Occupation=factor(df$Occupation)
  df$Age=factor(df$Age)
  df$Age=gsub(pattern = "-",replacement = "_", df$Age)
  df$Stay_In_Current_City_Years=as.character(df$Stay_In_Current_City_Years)
  df$Stay_In_Current_City_Years=gsub(pattern = "\\+",replacement = "", df$Stay_In_Current_City_Years)
  df$Stay_In_Current_City_Years=as.numeric(df$Stay_In_Current_City_Years)
  df$Marital_Status=factor(df$Marital_Status)
  df$multicat<-0
  df$multicat[apply(!is.na(df[,c("Product_Category_1","Product_Category_2","Product_Category_3")]),1,all)]<-1
  df$Product_Category_1=factor(df$Product_Category_1)
  df$Product_Category_2=factor(df$Product_Category_2)
  df$Product_Category_3=factor(df$Product_Category_3)
  return(df)
}

dtrain<-data_preparation(dtrain)

# # dummies<-data.frame(model.matrix(~Gender+Age+Occupation+City_Category+Stay_In_Current_City_Years+Marital_Status+Product_Category_1-1,data=dtrain))
# 
# #Converting every categorical variable to numerical using dummy variables
# dmy1 <- dummyVars(" ~ Gender+Age+Occupation+City_Category+Stay_In_Current_City_Years+Marital_Status+Product_Category_1", data = dtrain,fullRank = T)
# dmy2 <- dummyVars(" ~ Product_Category_2+Product_Category_3", data = dtrain,fullRank = T)
# dtrain_dummies <- data.frame(predict(dmy1, newdata = dtrain),predict(dmy2, newdata = dtrain), Purchase=dtrain$Purchase)
# 
# dtrain<-dtrain_dummies

#----------------
#   REORDER DATA AND SPLIT INTO TRAINING AND VALIDATION
#----------------
ord<-sample(nrow(dtrain),size = nrow(dtrain))

dtrain<-dtrain[ord,]
# 
# perc=0.2
# n<-round(perc*nrow(dtrain))
# dval<-dtrain[1:n,]
# dtrain<-dtrain[(n+1):nrow(dtrain),]

index <- createDataPartition(dtrain$Purchase, p=0.7, list=FALSE)
trainSet <- dtrain[ index,]
valSet <- dtrain[-index,]


#----------------
#   EXPLORATORY DATA ANALYSIS
#----------------
summary(trainSet)
apply(trainSet, 2, length)
apply(trainSet, 2, function(x)sum(is.na(x)))
apply(trainSet, 2, function(x)round(sum(is.na(x))/length(x),2))
str(trainSet)
ggplot(trainSet,aes(Purchase))+geom_histogram(bins=30)
ggplot(trainSet,aes(x=0,y=Purchase))+geom_boxplot()


#set missing as category for Product_Category_2 and Product_Category_3
missing_on_Product_Category<-function(df){
df$Product_Category_2<-as.character(df$Product_Category_2)
df$Product_Category_2[is.na(df$Product_Category_2)]<-"missing"
df$Product_Category_2<-factor(df$Product_Category_2)
df$Product_Category_3<-as.character(df$Product_Category_3)
df$Product_Category_3[is.na(df$Product_Category_3)]<-"missing"
df$Product_Category_3<-factor(df$Product_Category_3)
return(df)
}
trainSet<-missing_on_Product_Category(trainSet)
valSet<-missing_on_Product_Category(valSet)



#----------------
#   SETTING PARAMETERS
#----------------
outcomeName<-'Purchase'
predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]
#Feature selection using rfe in caret
# control <- rfeControl(functions = rfFuncs,
#                       method = "repeatedcv",
#                       repeats = 3,
#                       verbose = FALSE)
# 
# Loan_Pred_Profile <- rfe(trainSet[,predictors], trainSet[,outcomeName],
#                          rfeControl = control)


varx<-c("Gender", "Age", "Occupation", "City_Category", "Stay_In_Current_City_Years", "Marital_Status", "Product_Category_1", "Product_Category_2", "Product_Category_3", "multicat")
# varx<-predictors
X = trainSet[,varx]
y = trainSet[,"Purchase"]

Xv = valSet[,varx]
yv = valSet[,"Purchase"]



#Make a custom trainControl
myControl <- trainControl(
  method = "cv", number = 5,
  verboseIter = TRUE
)
#----------------
#   RF
#----------------
#tune parameter for RF


myGrid1<-data.frame(mtry=c(4,5,6,7,8))
n1<-sample(nrow(X), 20000)
mod_rf_tune<-train(x =X[n1,],
                   y=y[n1],
                   method = "ranger",
                   metric = "RMSE", 
                   # tuneLength =ncol(X),
                   tuneGrid = myGrid1,
                   trControl = myControl)
plot(mod_rf_tune)
summary(mod_rf_tune)

myGrid2<-mod_rf_tune$results  %>% top_n(n=-1, wt=RMSE) %>% select(mtry)
print(myGrid2)
n2<-sample(nrow(X), 40000)
mod_rf<-train(x =X[n2,],
              y=y[n2],
              method = "ranger",
              metric = "RMSE",
              tuneGrid =  myGrid2,
              trControl = myControl)

# mod_rf<-mod_rf_tune
print(mod_rf)
plot(mod_rf)
mod_rf$results$RMSE

#----------------
#   GLMNET
#----------------

myGridglmnet <- expand.grid(alpha = 0:1,
                            lambda = seq(0.0001, 0.1, length = 10)
)
mod_glmnet <- train(x = X,
                    y = y,
                    method = "glmnet",
                    tuneGrid = myGridglmnet,
                    trControl = myControl,
                    preProcess = c("medianImpute", "center", "scale"))
# Plot results
plot(mod_glmnet)


varImp(object=mod_glmnet)
plot(varImp(object=mod_glmnet),main="Variable Importance")

subvar<-varImp(object=mod_glmnet)$importance
subvar<-cbind(var=rownames(subvar), subvar) %>% arrange(desc(Overall))
subvar<-subvar$var[1:20]


mod_glmnet <- train(x = X[,subvar],
                    y = y,
                    method = "glmnet",
                    tuneGrid = myGridglmnet,
                    trControl = myControl,
                    preProcess = c("medianImpute", "center", "scale"))
# Plot results


print(mod_glmnet)
coef(mod_glmnet$finalModel)

#----------------
#test model on VALIDATION SET
#----------------
y_val<-valSet$Purchase
yhat<-predict(mod_rf, newdata = valSet)

#Predictions
yhat<-predict(object=mod_glmnet, newdata = Xv)

postResample(pred = yhat,obs = y_val)

#----------------
#import test set and submit solution
#----------------
dtest<-read.csv(file = "test.csv",header = T)
dtest<-data_preparation(dtest)

y_hat<-predict(mod_rf, newdata = dtest)
dtest<-cbind(dtest, Purchase=y_hat)

submission<-dtest[,c("User_ID","Product_ID","Purchase")]

sum(is.na(submission$Purchase))
ggplot(dtest, aes(Purchase))+geom_histogram()

write.csv(submission, "submission.csv", row.names = F)



