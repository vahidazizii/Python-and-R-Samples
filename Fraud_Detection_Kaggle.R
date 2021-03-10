
#Load packages and read data***********************************

options(warn = -1, digits= 4)
library(Boruta)
library(ranger)
library(magrittr)
library(ggplot2)
library(data.table)
library(xgboost)
library(caret)
library(plyr)
library(dplyr)
library(knitr)
library(RWeka)
library(klaR)
library(e1071)
library(DMwR)
library(pROC)


set.seed(876)
load("train.rda")
dataset1<-train.data %>% filter(is_attributed==1)
dataset0<-train.data %>% filter(is_attributed==0) %>% sample_n(size=3000000)

data<-rbind(dataset0,dataset1)
data<- data[sample(nrow(data)),]


data$is_attributed <- factor(ifelse(data$is_attributed== 0, "zero", "one"))

data2<-data

#feature construction*************************************


#1. Deleting   "attributed_time" feature from data data

data<-data[,-7]  


#2. creating new feature "hour" and deleting "click_time" feature ***********

data$hour<-substr(data$click_time,start = 12,stop = 13)
data<-data[,-6]



#3. creating new feature "likely download time" ******************************
most_frequent_hours_download  <- c("01","03","02","04","13")
least_frequent_hours_download <- c("15","16","17","18","19","20","21","22","23")

data$hour_download= ifelse(data$hour %in% most_frequent_hours_download, 1,
                           ifelse(data$hour %in% least_frequent_hours_download, 3, 2))

#4. creating group_by features    ********************************************


#creating groups of two combined feature for data data
data<-data %>%
  group_by(ip, app) %>%
  mutate(ip_app = n())

data<-data %>%
  group_by(ip, device) %>%
  mutate(ip_device = n())

data<-data %>%
  group_by(ip, os) %>%
  mutate(ip_os = n())

data<-data %>%
  group_by(ip, channel) %>%
  mutate(ip_channel = n())

data<-data %>%
  group_by(ip, hour) %>%
  mutate(ip_hour = n())

data<-data %>%
  group_by(app, device) %>%
  mutate(app_device = n())

data<-data %>%
  group_by(app, os) %>%
  mutate(app_os = n())

data<-data %>%
  group_by(app, channel) %>%
  mutate(app_channel = n())

data<-data %>%
  group_by(app, hour) %>%
  mutate(app_hour = n())


data<-data %>%
  group_by(device, os) %>%
  mutate(device_os = n())

data<-data %>%
  group_by(device, channel) %>%
  mutate(device_channel = n())

data<-data %>%
  group_by(device, hour) %>%
  mutate(device_hour = n())

data<-data %>%
  group_by(os, channel) %>%
  mutate(os_channel = n())

data<-data %>%
  group_by(os, hour) %>%
  mutate(os_hour = n())


data<-data %>%
  group_by(channel, hour) %>%
  mutate(channel_hour = n())




#Splitting data to four subsets in order to featur selection analysis, tunning, training and validation
dfeature<-rbind(data[1:100000,])
dtune<-rbind(data[(100001):(200000),])
dtrain<-rbind(data[(200001):(1700000),])
dvalid<-rbind(data[(1700001):(3456846),])


#Using Boruta package for feature selection***************************************************
boruta_output <- Boruta(dfeature$is_attributed ~ ., data=na.omit(dfeature),maxRuns = 20, doTrace=2)
boruta_signif <- names(boruta_output$finalDecision[boruta_output$finalDecision %in% c("Confirmed", "Tentative")])
selected_attributes<-c(boruta_signif,"is_attributed")
saveRDS(boruta_output, file = "boruta_output.rds")

plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")

dtrain = dtrain[, colnames(dtrain) %in% selected_attributes]
dvalid = dvalid[, colnames(dvalid) %in% selected_attributes]
dvalid$is_attributed<- factor(ifelse(dvalid$is_attributed== "zero",0, 1)) 




saveRDS(dvalid, file = "dvalid.rds")
saveRDS(dtrain, file = "dtrain.rds")
saveRDS(dfeature, file = "dfeature.rds")
saveRDS(dtune, file = "dtune.rds")

#Tunning****************************************************

trControl = trainControl(classProbs = TRUE)

tuneGridXGB <- expand.grid(
  nrounds=c(1000),
  max_depth = c(4, 6),
  eta = c(0.05, 0.1),
  gamma = c(0.01),
  colsample_bytree = c(0.75),
  subsample = c(0.50),
  min_child_weight = c(0))

xgb_tune <- train(is_attributed~.,data=dtune,
                  method = 'xgbTree',
                  metric = 'auc',
                  trControl = trControl,
                  tuneGrid = tuneGridXGB)

#Smapling***************************************************

#Upsampling
downsample_train <- downSample(x = dtrain[, -6], y = dtrain$is_attributed)
downsample_train2<-downsample_train
#Downsampling
upsample_train <- upSample(x = dtrain[, -6], y = dtrain$is_attributed)
upsample_train2<-upsample_train


saveRDS(data, file = "data.rds")
saveRDS(downsample_train, file = "downsample_train.rds")
saveRDS(upsample_train, file = "upsample_train.rds")

#Random forest *********************************************
#Random forest down sample
rf_down_model <- ranger(Class~., data=downsample_train)
pred <- predict(rf_down_model,as.data.frame(dvalid))
pred$predictions<- factor(ifelse(pred$predictions== "zero",0, 1)) 
rf_down_model_confusionMatrix<-confusionMatrix(pred$predictions, dvalid$is_attributed)

saveRDS(rf_down_model, file = "rf_down_model.rds")
saveRDS(rf_down_model_confusionMatrix, file = "rf_down_model_confusionMatrix.rds")
saveRDS(pred$predictions, file = "rf_down_pred.rds")

#Random forest up sample
rf_up_model <- ranger(Class~., data=upsample_train)
pred <- predict(rf_up_model,as.data.frame(dvalid))
pred$predictions<- factor(ifelse(pred$predictions== "zero",0, 1)) 
rf_up_model_confusionMatrix<-confusionMatrix(pred$predictions, dvalid$is_attributed)

saveRDS(rf_up_model, file = "rf_up_model.rds")
saveRDS(rf_up_model_confusionMatrix, file = "rf_up_model_confusionMatrix.rds")
saveRDS(pred$predictions, file = "rf_up_pred.rds")

#Random forest all train data
rf_alldata_model <- ranger(is_attributed~., data=dtrain)
pred <- predict(rf_alldata_model,as.data.frame(dvalid))
pred$predictions<- factor(ifelse(pred$predictions== "zero",0, 1)) 
rf_alldata_model_confusionMatrix<-confusionMatrix(pred$predictions, dvalid$is_attributed)

saveRDS(rf_alldata_model, file = "rf_alldata_model.rds")
saveRDS(rf_alldata_model_confusionMatrix, file = "rf_alldata_model_confusionMatrix.rds")
saveRDS(pred$predictions, file = "rf_alldata_pred.rds")


#Xgboost**************************************************
downsample_train<-downsample_train2
upample_train<-upsample_train2
dvalid$is_attributed<-as.numeric(dvalid$is_attributed)-1

#xgboost down sample
downsample_train$Class<- factor(ifelse(downsample_train$Class== "zero",0,1))
downsample_train$Class<-as.numeric(downsample_train$Class)-1

X<-downsample_train[,!colnames(downsample_train) %in% c("Class")]
X[, colnames(X)] <- lapply(X[, colnames(X)], as.numeric)
downsample_train2 <- xgb.DMatrix(as.matrix(X), label = downsample_train$Class)

XX<-dvalid[,!colnames(dvalid) %in% c("is_attributed")]
XX[, colnames(XX)] <- lapply(XX[, colnames(XX)], as.numeric)
dvalid2 <- xgb.DMatrix(as.matrix(XX), label = dvalid$is_attributed)

params <- list( objective   = "binary:logistic", 
                grow_policy = "lossguide",
                tree_method = "auto",
                eval_metric = "error", 
                eta = 0.05, 
                max_depth = 6,
                gamma = 0.01,
                colsample_bytree = 0.75,
                subsample = 0.5,
                min_child_weight = 0
)


xgb_down_model <- xgb.train(data = downsample_train2, params = params, 
                       silent = 1, watchlist = list(valid = dvalid2), nthread = 16, 
                       nrounds=1000, print_every_n = 25, early_stopping_rounds = 100)

preds <- predict(xgb_down_model, newdata = dvalid2, ntreelimit = xgb_down_model$best_ntreelimit)
preds<-round(preds, 0)
xgb_down_model_confusionMatrix<-confusionMatrix(preds, dvalid$is_attributed)

saveRDS(xgb_down_model, file = "xgb_down_model.rds")
saveRDS(xgb_down_model_confusionMatrix, file = "xgb_down_model_confusionMatrix.rds")
saveRDS(preds, file = "xgb_down_pred.rds")



#xgboost up sample

upsample_train$Class<- factor(ifelse(upsample_train$Class== "zero",0,1))
upsample_train$Class<-as.numeric(upsample_train$Class)-1

X<-upsample_train[,!colnames(upsample_train) %in% c("Class")]
X[, colnames(X)] <- lapply(X[, colnames(X)], as.numeric)
upsample_train2 <- xgb.DMatrix(as.matrix(X), label = upsample_train$Class)


XX<-dvalid[,!colnames(dvalid) %in% c("is_attributed")]
XX[, colnames(XX)] <- lapply(XX[, colnames(XX)], as.numeric)
dvalid2 <- xgb.DMatrix(as.matrix(XX), label = dvalid$is_attributed)

params <- list( objective   = "binary:logistic", 
                grow_policy = "lossguide",
                tree_method = "auto",
                eval_metric = "error", 
                eta = 0.05, 
                max_depth = 6,
                gamma = 0.01,
                colsample_bytree = 0.75,
                subsample = 0.5,
                min_child_weight = 0
)

xgb_up_model <- xgb.train(data = upsample_train2, params = params, 
                       silent = 1, watchlist = list(valid = dvalid2), nthread = 16, 
                       nrounds=1000, print_every_n = 25, early_stopping_rounds = 100)

preds <- predict(xgb_up_model, newdata = dvalid2, ntreelimit = xgb_up_model$best_ntreelimit)
preds<-round(preds, 0)
xgb_up_model_confusionMatrix<-confusionMatrix(preds, dvalid$is_attributed)

saveRDS(xgb_up_model, file = "xgb_up_model.rds")
saveRDS(xgb_up_model_confusionMatrix, file = "xgb_up_model_confusionMatrix.rds")
saveRDS(preds, file = "xgb_up_pred.rds")

#xgboost all train data
dtrain0<-dtrain
dtrain0$is_attributed<- factor(ifelse(dtrain0$is_attributed== "zero",0,1))
dtrain0$is_attributed<-as.numeric(dtrain0$is_attributed)-1
dtrain0<-as.data.frame(dtrain0)
X<-dtrain0[,!colnames(dtrain0) %in% c("is_attributed")]
X[, colnames(X)] <- lapply(X[, colnames(X)], as.numeric)
dtrain2 <- xgb.DMatrix(as.matrix(X), label = dtrain0$is_attributed)


XX<-dvalid0[,!colnames(dvalid0) %in% c("is_attributed")]
XX[, colnames(XX)] <- lapply(XX[, colnames(XX)], as.numeric)
dvalid2 <- xgb.DMatrix(as.matrix(XX), label = dvalid0$is_attributed)

params <- list( objective   = "binary:logistic", 
                grow_policy = "lossguide",
                tree_method = "auto",
                eval_metric = "error", 
                eta = 0.05, 
                max_depth = 6,
                gamma = 0.01,
                colsample_bytree = 0.75,
                subsample = 0.5,
                min_child_weight = 0
)


xgb_alldata_model <- xgb.train(data = dtrain2, params = params, 
                          silent = 1, watchlist = list(valid = dvalid2), nthread = 16, 
                          nrounds=1000, print_every_n = 25, early_stopping_rounds = 100)



preds <- predict(xgb_alldata_model, newdata = dvalid2, ntreelimit = xgb_alldata_model$best_ntreelimit)
preds<-round(preds, 0)
xgb_alldata_model_confusionMatrix<-confusionMatrix(preds, dvalid$is_attributed)


saveRDS(xgb_alldata_model, file = "xgb_alldata_model.rds")
saveRDS(xgb_alldata_model_confusionMatrix, file = "xgb_alldata_model_confusionMatrix.rds")
saveRDS(preds, file = "xgb_alldata_pred.rds")


# Performance evaluation******************************
#Feature importance
kable(xgb.importance(model = xgb_alldata_model))

##ROC plots

readRDS("dvalid.rds")

rf_down_pred<-readRDS("rf_down_pred.rds")
rf_up_pred<-readRDS("rf_up_pred.rds")
rf_alldata_pred<-readRDS("rf_alldata_pred.rds")
xgb_down_pred<-readRDS("xgb_down_pred.rds")
xgb_up_pred<-readRDS("xgb_up_pred.rds")
xgb_alldata_pred<-readRDS("xgb_alldata_pred.rds")


roc_rf_down_pred<-roc(dvalid$is_attributed,as.numeric(rf_down_pred))
roc_rf_up_pred<-roc(dvalid$is_attributed,as.numeric(rf_up_pred))
roc_rf_alldata_pred<-roc(dvalid$is_attributed,as.numeric(rf_alldata_pred))
roc_xgb_down_pred<-roc(dvalid$is_attributed,as.numeric(xgb_down_pred))
roc_xgb_up_pred<-roc(dvalid$is_attributed,as.numeric(xgb_up_pred))
roc_xgb_alldata_pred<-roc(dvalid$is_attributed,as.numeric(xgb_alldata_pred))



plot(roc_rf_down_pred)
plot(roc_rf_up_pred)
plot(roc_rf_alldata_pred)
plot(roc_xgb_down_pred)
plot(roc_xgb_up_pred)
plot(roc_xgb_alldata_pred)


#Random forest predict process test data**************************************
test <- read.csv("test.csv")
sub <- data.table(click_id = test$click_id, is_attributed = NA) 
test$click_id <- NULL
test[, 1:6] <- lapply(test[, 1:6], as.numeric)
test<-as.data.frame(test)
preds <- predict(rf_up_model, data = test)
sub$is_attributed = round(preds$prediction, 5)
fwrite(sub, "Test_rf_Results2.csv")




#Xgboost predict process test data**************************************
test <- read.csv("test.csv")
sub <- data.table(click_id = test$click_id, is_attributed = NA) 
test$click_id <- NULL
invisible(gc())
test[, 1:6] <- lapply(test[, 1:6], as.numeric)
test  <- xgb.DMatrix(as.matrix(test))

preds <- predict(xgb_up_model, newdata = test, ntreelimit = xgb_up_model$best_ntreelimit)
preds <- as.data.frame(preds)
sub$is_attributed = round(preds, 0)
fwrite(sub, "Test_Xgb_Results.csv")