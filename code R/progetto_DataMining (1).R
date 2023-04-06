library(readxl)
data_bankruptcy <- read_excel("~/Desktop/data_bankruptcy.xlsx")
View(data_bankruptcy)  

dim(data_bankruptcy)
str(data_bankruptcy)
head(data_bankruptcy)

table(data_bankruptcy$`Bankrupt?`)
table(data_bankruptcy$`Bankrupt?`)/nrow(data_bankruptcy)
# 3% di bankrupcy, il dataset non sembra bilanciato

library(funModeling)
library(dplyr)

# step1: build models (decide models, preprocessng, tuning, model selection)  #####

# control variables, if there are strange things####
status=df_status(data_bankruptcy, print_results = F)
head(status%>% arrange(type))
head(status%>% arrange(unique))
head(status%>% arrange(-p_na))

library(caret)

table(data_bankruptcy$`Net_Income_Flag`)
# variabile con zero variance, da eliminare
dataset <- data_bankruptcy
# test near zero variance
nzv = nearZeroVar(dataset, saveMetrics = TRUE)
nzv
dataset$Net_Income_Flag <- NULL
dataset$Liability_Assets_Flag <- NULL

# conversione variabili da character a numeric
char_columns <- sapply(dataset, is.character)            
dataset[ , char_columns] <- as.data.frame(  
  apply(dataset[ , char_columns], 2, as.numeric))
sapply(dataset, class)

# Caret want target as factor and coded with letters: recode the target as no yes
library(car)
dataset$Bankrupt <-recode(dataset$`Bankrupt?`, recodes="0='no'; else='yes'")

# classificative models want target as factor
dataset$Bankrupt=as.factor(dataset$Bankrupt)

# Create a copy of target=Qual as numeric var 0/1 (some model/package  want y as c0/c1, other, outside caret want y as 0,1)####
Bankrupt<-recode(dataset$`Bankrupt?`, recodes="0=0; else=1")

# remove old target
dataset$`Bankrupt?` <- NULL

# eliminazione covariate collineari
R=cor(dataset[,-94])
correlatedPredictors = findCorrelation(R, cutoff = 0.95, names = TRUE)
correlatedPredictors

dataset_nocorr <- dataset[,-c(1,2,4,8,10,16,17,19,22,23,27,37,64,66,88,90)]

# otteniamo il dataset di score
set.seed(1234)
split <- createDataPartition(y=dataset_nocorr$Bankrupt, p = 0.90, list = FALSE)
data <- dataset_nocorr[split,]
data_score <- dataset_nocorr[-split,]

table(data$Bankrupt)/nrow(data)
table(data_score$Bankrupt)/nrow(data_score)

# divisione in training e validation
set.seed(1234)
split1 <- createDataPartition(y=data$Bankrupt, p = 0.66, list = FALSE)
data_training <- data[split1,]
data_validation <- data[-split1,]

table(data_training$Bankrupt)/nrow(data_training)
table(data_validation$Bankrupt)/nrow(data_validation)

#oversampling
train_over<-upSample(data_training[,-78], data_training$Bankrupt, list = FALSE, yname = "Bankrupt")
table(train_over$Bankrupt)/nrow(train_over)
# train_over dataset no preprocessing no model selection 

# creazione dataset preprocessato per correlazione, near zero variance e scaling  

# sistema nomi variabili
colnames(train_over) <- make.names(colnames(train_over))
colnames(data_validation) <- make.names(colnames(data_validation))
colnames(data_score) <- make.names(colnames(data_score))

set.seed(1234)
cvCtrl <- trainControl(method = "cv", number=10, search="grid", classProbs = TRUE,
                       summaryFunction = twoClassSummary)
rpartTuneCvA <- train(Bankrupt ~ . , data = train_over, method = "rpart",
                      tuneLength = 10, metric = "Spec",
                      trControl = cvCtrl)

rpartTuneCvA
plot(rpartTuneCvA)
confusionMatrix(rpartTuneCvA)
getTrainPerf(rpartTuneCvA)
# var imp of the tree
varImp(rpartTuneCvA)
plot(varImp(object=rpartTuneCvA),main="train tuned - Variable Importance")

# select only important variables
VI=as.data.frame(rpartTuneCvA$finalModel$variable.importance)
viname=row.names(VI)
viname

# new train with selected covariates+ target
train_ms=train_over[,viname]
train_ms=cbind(train_over$Bankrupt, train_ms)
names(train_ms)[1] <- "Bankrupt"
head(train_ms)
# train_ms dataset no preprocessing sì model slection

validation_ms=data_validation[,viname]
validation_ms=cbind(data_validation$Bankrupt, validation_ms)
names(validation_ms)[1] <- "Bankrupt"
head(validation_ms)
# validation_ms dataset di validation con model selection


# modello logistico
set.seed(1234)
Control=trainControl(method= "cv", number=10, summaryFunction = twoClassSummary, classProbs = TRUE)
logistico = train(Bankrupt ~ ., data=train_ms ,  method = "glm", trControl = Control, tuneLength=10, metric = "Spec",
                  preProcess=c("corr", "scale"))

summary(logistico)
logistico
confusionMatrix(logistico)


# fit a lasso-glm model: alpha=1  (alpha=0 IS RIDGE PENALIZATION) (preprocessed)
set.seed(1234)
grid = expand.grid(.alpha=1,.lambda=seq(0, 1, by = 0.01))
Control1=trainControl(method= "cv",number=10, summaryFunction = twoClassSummary, classProbs=TRUE)
lasso=train(Bankrupt ~ ., data=train_over , method = "glmnet", family ="binomial",
            trControl = Control1, tuneLength=10, tuneGrid=grid,  metric="Spec",
            preProcess=c("corr", "scale"))
lasso
plot(lasso)

# coefficients of lasso model....see how it does model selection
coef(lasso$finalModel, s=lasso$bestTune$lambda)
# cross-validated confusion matrix
confusionMatrix(lasso)

# PLS regression (preprocessed) #####
# caret tune the number of linear components

set.seed(1234)
Control2=trainControl(method= "cv",number=10, summaryFunction = twoClassSummary, classProbs=TRUE)
pls=train(Bankrupt ~ ., data=train_ms , method = "pls", 
          trControl = Control2, tuneLength=10, metric="Spec",
          preProcess=c("corr", "scale"))
pls
plot(pls)
# cross-valudated confusion matrix
confusionMatrix(pls)

# knn
set.seed(1234)
control3 =trainControl(method="cv", number = 10, classProbs = TRUE,
                   summaryFunction=twoClassSummary)
grid1 = expand.grid(k=seq(5,20,3))
knn=train(Bankrupt~., data=train_ms, method = "knn",
          trControl = control3, tuneLength=10,
          tuneGrid=grid1, metric = "Spec", preProcess=c("scale", "center"))
knn
plot(knn)
confusionMatrix(knn)

# random forest

set.seed(1234)
control4 <- trainControl(method = "cv", number=10, search="grid", classProbs = TRUE,
                       summaryFunction = twoClassSummary)
random_forest <- train(Bankrupt~., data=train_over, method = "rf",
                tuneLength = 10, trControl = control4, metric = "Spec")
random_forest
plot(random_forest)
confusionMatrix(random_forest)

vimp=varImp(random_forest)
plot(varImp(object=random_forest),main="train tuned - Variable Importance")



library(gbm)
set.seed(1234)
control8=trainControl(method= "cv",number=10, search="grid", summaryFunction = twoClassSummary, classProbs=TRUE)
gbm_grid <- expand.grid(interaction.depth = c(1,2,3,4,5,6,7,8,9),
                        n.trees = 50,
                        shrinkage = c(0.075,0.1,0.5,0.7),
                        n.minobsinnode = 20)
gradient_boost <- train(Bankrupt ~ ., data=train_over, method = "gbm", tuneGrid=gbm_grid,
                        metric="Spec", trControl=control8, verbose=FALSE)

gradient_boost
plot(gradient_boost)
ggplot(gradient_boost)
confusionMatrix(gradient_boost)

set.seed(1234)
Control5=trainControl(method= "cv",number=10, summaryFunction = twoClassSummary, classProbs=TRUE)
lda=train(Bankrupt ~ ., data=train_ms , method = "lda", 
          trControl = Control2, tuneLength=10, metric="Spec",
          preProcess=c("corr", "scale", "center"))
lda
# cross-valudated confusion matrix
confusionMatrix(lda)

set.seed(1234)
control9 = trainControl(method="cv", number=10, search = "grid", summaryFunction = twoClassSummary, classProbs=TRUE)
tunegrid <- expand.grid(size=seq(1,7, by=1), decay = c(0.001, 0.01, 0.05 , .1, .3))
nnet <- train(train_ms[-1], train_ms$Bankrupt,
                   method = "nnet",
                   preProcess = c("corr", "range"), 
                   tuneLength = 10, metric="Spec", trControl=control9, tuneGrid=tunegrid,
                   maxit = 300)
nnet
ggplot(nnet)
confusionMatrix(nnet)

#STEP 2

results <- resamples(list(logistic=logistico, lasso=lasso, pls=pls, tree=rpartTuneCvA, lda=lda, knn=knn, random_forest=random_forest, gradient_boost=gradient_boost, neural_network=nnet))
bwplot(results)

# estimate probs P(M)
data_validation$logistic = predict(logistico       , data_validation, "prob")[,1]
data_validation$lasso = predict(lasso         , data_validation, "prob")[,1]
data_validation$pls = predict(pls    , data_validation, "prob")[,1]
data_validation$tree = predict(rpartTuneCvA      , data_validation, "prob")[,1]
data_validation$lda = predict(lda, data_validation, "prob")[,1]
data_validation$knn = predict(knn, data_validation, "prob")[,1]
data_validation$random_forest = predict(random_forest, data_validation, "prob")[,1]
data_validation$gradient_boost = predict(gradient_boost, data_validation, "prob")[,1]
data_validation$neural_network = predict(nnet, data_validation, "prob")[,1]

library(pROC)
# roc values
roc_log=roc(Bankrupt ~ logistic, data = data_validation); roc_log
roc_lasso=roc(Bankrupt ~ lasso, data = data_validation); roc_lasso
roc_pls=roc(Bankrupt ~ pls, data = data_validation); roc_pls
roc_tree=roc(Bankrupt ~ tree, data = data_validation); roc_tree
roc_lda=roc(Bankrupt ~ lda, data = data_validation); roc_lda
roc_knn=roc(Bankrupt ~ knn, data = data_validation); roc_knn
roc_rf=roc(Bankrupt ~ random_forest, data = data_validation); roc_rf
roc_gb=roc(Bankrupt ~ gradient_boost, data = data_validation); roc_gb
roc_nnet=roc(Bankrupt ~ neural_network, data = data_validation); roc_nnet

roc_rf

plot(roc_log, col="red")
plot(roc_lasso,add=T,col="orange")
plot(roc_pls,add=T,col="yellow")
plot(roc_tree,add=T,col="green")
plot(roc_lda,add=T,col="blue")
plot(roc_knn,add=T,col="violet")
plot(roc_rf,add=T,col="pink")
plot(roc_gb,add=T,col="grey")
plot(roc_nnet,add=T,col="brown")
legend(x="bottomright",
       legend = c("logistic","lasso","pls","tree","lda","knn","rf","gb","nnet"),
       lty = 1, lwd = 2, cex=0.45,
       col=c("red","orange","yellow","green","blue","violet","pink","grey","brown"))


#curva lift rf
rf_old_pr1 = predict(random_forest, data_validation, "prob")[,2]
rf_pred_yes<-as.numeric(rf_old_pr1) 
rf_pred_no = 1 - rf_pred_yes 
rho1 = 0.5; rho0 = 0.5; true0 = 0.968; true1 = 0.032 
rf_den = rf_pred_yes*(true1/rho1)+rf_pred_no*(true0/rho0) 
data_validation$rf_pred1_true = rf_pred_yes*(true1/rho1)/rf_den
data_validation$rf_pred0_true = rf_pred_no*(true0/rho0)/rf_den

library(funModeling)
gain_lift(data = data_validation, score = 'rf_pred1_true', target = 'Bankrupt')

#curva lift lasso
lasso_old_pr1 = predict(lasso, data_validation, "prob")[,2]
lasso_pred_yes<-as.numeric(lasso_old_pr1) 
lasso_pred_no = 1 - lasso_pred_yes 
rho1 = 0.5; rho0 = 0.5; true0 = 0.968; true1 = 0.032 
lasso_den = lasso_pred_yes*(true1/rho1)+lasso_pred_no*(true0/rho0) 
data_validation$lasso_pred1_true = lasso_pred_yes*(true1/rho1)/lasso_den
data_validation$lasso_pred0_true = lasso_pred_no*(true0/rho0)/lasso_den

library(funModeling)
gain_lift(data = data_validation, score = 'lasso_pred1_true', target = 'Bankrupt')


#curva lift pls
pls_old_pr1 = predict(pls, data_validation, "prob")[,2]
pls_pred_yes<-as.numeric(pls_old_pr1) 
pls_pred_no = 1 - pls_pred_yes 
rho1 = 0.5; rho0 = 0.5; true0 = 0.968; true1 = 0.032 
pls_den = pls_pred_yes*(true1/rho1)+pls_pred_no*(true0/rho0) 
data_validation$pls_pred1_true = pls_pred_yes*(true1/rho1)/pls_den
data_validation$pls_pred0_true = pls_pred_no*(true0/rho0)/pls_den

library(funModeling)
gain_lift(data = data_validation, score = 'pls_pred1_true', target = 'Bankrupt')

#curva lift gb
gb_old_pr1 = predict(gradient_boost, data_validation, "prob")[,2]
gb_pred_yes<-as.numeric(gb_old_pr1) 
gb_pred_no = 1 - gb_pred_yes 
rho1 = 0.5; rho0 = 0.5; true0 = 0.968; true1 = 0.032 
gb_den = gb_pred_yes*(true1/rho1)+gb_pred_no*(true0/rho0) 
data_validation$gb_pred1_true = gb_pred_yes*(true1/rho1)/gb_den
data_validation$gb_pred0_true = gb_pred_no*(true0/rho0)/gb_den

library(funModeling)
gain_lift(data = data_validation, score = 'gb_pred1_true', target = 'Bankrupt')

# random forest modello migliore sia per ROC sia per LIFT

#STEP 3

# con random forest
# take probabilities fitted on the validation set 
data_validation$rf_pred0_true

# save validation results: observed target an pred probs
df=data.frame(cbind(data_validation$Bankrupt , data_validation$rf_pred0_true, data_validation$rf_pred1_true))
head(df)
colnames(df)=c("Bankrupt","ProbNO","ProbYES")
head(df)


df=df[,1:2]
df$Bankrupt=ifelse(df$Bankrupt ==1, "no","yes")
df$Bankrupt <- as.factor(df$Bankrupt)
head(df)

str(df)
# one factor and one numeric

# create a cycle: for each ProbM of validation set create confusion matrices (TP, FN....) and measure of interest (TPR, FPR, )##########

library(dplyr)


thresholds <- seq(from = 0, to = 1, by = 0.005)
prop_table <- data.frame(threshold = thresholds, prop_true_no = NA,  prop_true_yes = NA, true_no = NA,  true_yes = NA ,fn_no=NA)

for (threshold in thresholds) {
  pred <- ifelse(df$ProbNO > threshold, "no", "yes")  # be careful here!!!
  pred_t <- ifelse(pred == df$Bankrupt, TRUE, FALSE)
  
  group <- data.frame(df, "pred" = pred_t) %>%
    group_by(Bankrupt, pred) %>%
    dplyr::summarise(n = n())
  
  group_no <- filter(group, Bankrupt == "no")
  
  true_no=sum(filter(group_no, pred == TRUE)$n)
  prop_no <- sum(filter(group_no, pred == TRUE)$n) / sum(group_no$n)
  
  prop_table[prop_table$threshold == threshold, "prop_true_no"] <- prop_no
  prop_table[prop_table$threshold == threshold, "true_no"] <- true_no
  
  fn_no=sum(filter(group_no, pred == FALSE)$n)

  prop_table[prop_table$threshold == threshold, "fn_no"] <- fn_no
  
  
  group_yes <- filter(group, Bankrupt == "yes")
  
  true_yes=sum(filter(group_yes, pred == TRUE)$n)
  prop_yes <- sum(filter(group_yes, pred == TRUE)$n) / sum(group_yes$n)
  
  prop_table[prop_table$threshold == threshold, "prop_true_yes"] <- prop_yes
  prop_table[prop_table$threshold == threshold, "true_yes"] <- true_yes
  
}

head(prop_table, n=10)

# prop_true_no = sensitivity
# prop_true_yes = specificicy

# calculate other missing measures

# n of observations of the validation set    
prop_table$n=nrow(data_validation)

# false positive (fp_M) by difference of   n and            tn,                 tp,         fn, 
prop_table$fp_no=nrow(data_validation)-prop_table$true_yes-prop_table$true_no-prop_table$fn_no

# find accuracy
prop_table$acc=(prop_table$true_yes+prop_table$true_no)/nrow(data_validation)

# find precision
prop_table$prec_no=prop_table$true_no/(prop_table$true_no+prop_table$fp_no)

# find F1 =2*(prec*sens)/(prec+sens)

prop_table$F1=2*(prop_table$prop_true_no*prop_table$prec_no)/(prop_table$prop_true_no+prop_table$prec_no)

# verify not having NA metrics at start or end of data 
tail(prop_table)
# we have typically some NA in the precision and F1 at the boundary..put,impute 1,0 respectively 

library(Hmisc)
#impute NA as 0, this occurs typically for precision
prop_table$prec_no=impute(prop_table$prec_no, 1)
prop_table$F1=impute(prop_table$F1, 0)
tail(prop_table, n=10)

colnames(prop_table)

# drop counts, PLOT only metrics

prop_table2 = prop_table[,-c(4:8)] 
tail(prop_table2, n=11)

# plot measures vs soglia##########
# before we must impile data vertically: one block for each measure
library(dplyr)
library(tidyr)

gathered=prop_table2 %>%
  gather(x, y, prop_true_no:F1)

head(gathered)

# plot measures 
library(ggplot2)
gathered %>%
  ggplot(aes(x = threshold, y = y, color = x)) +
  geom_point() +
  geom_line() +
  scale_color_brewer(palette = "Set1") +
  labs(y = "measures",
       color = "no: event\nyes: nonevent")


# con pls

df1=data.frame(cbind(data_validation$Bankrupt , data_validation$pls_pred0_true, data_validation$pls_pred1_true))
head(df1)
colnames(df1)=c("Bankrupt","ProbNO","ProbYES")
head(df1)


df1=df1[,1:2]
df1$Bankrupt=ifelse(df1$Bankrupt ==1, "no","yes")
df1$Bankrupt <- as.factor(df1$Bankrupt)
head(df1)

str(df1)
# one factor and one numeric

# create a cycle: for each ProbM of validation set create confusion matrices (TP, FN....) and measure of interest (TPR, FPR, )##########

library(dplyr)


thresholds <- seq(from = 0, to = 1, by = 0.005)
prop_table <- data.frame(threshold = thresholds, prop_true_no = NA,  prop_true_yes = NA, true_no = NA,  true_yes = NA ,fn_no=NA)

for (threshold in thresholds) {
  pred <- ifelse(df1$ProbNO > threshold, "no", "yes")  # be careful here!!!
  pred_t <- ifelse(pred == df1$Bankrupt, TRUE, FALSE)
  
  group <- data.frame(df1, "pred" = pred_t) %>%
    group_by(Bankrupt, pred) %>%
    dplyr::summarise(n = n())
  
  group_no <- filter(group, Bankrupt == "no")
  
  true_no=sum(filter(group_no, pred == TRUE)$n)
  prop_no <- sum(filter(group_no, pred == TRUE)$n) / sum(group_no$n)
  
  prop_table[prop_table$threshold == threshold, "prop_true_no"] <- prop_no
  prop_table[prop_table$threshold == threshold, "true_no"] <- true_no
  
  fn_no=sum(filter(group_no, pred == FALSE)$n)

  prop_table[prop_table$threshold == threshold, "fn_no"] <- fn_no
  
  
  group_yes <- filter(group, Bankrupt == "yes")
  
  true_yes=sum(filter(group_yes, pred == TRUE)$n)
  prop_yes <- sum(filter(group_yes, pred == TRUE)$n) / sum(group_yes$n)
  
  prop_table[prop_table$threshold == threshold, "prop_true_yes"] <- prop_yes
  prop_table[prop_table$threshold == threshold, "true_yes"] <- true_yes
  
}

head(prop_table, n=10)

# prop_true_no = sensitivity
# prop_true_yes = specificicy

# calculate other missing measures

# n of observations of the validation set    
prop_table$n=nrow(data_validation)

# false positive (fp_M) by difference of   n and            tn,                 tp,         fn, 
prop_table$fp_no=nrow(data_validation)-prop_table$true_yes-prop_table$true_no-prop_table$fn_no

# find accuracy
prop_table$acc=(prop_table$true_yes+prop_table$true_no)/nrow(data_validation)

# find precision
prop_table$prec_no=prop_table$true_no/(prop_table$true_no+prop_table$fp_no)

# find F1 =2*(prec*sens)/(prec+sens)

prop_table$F1=2*(prop_table$prop_true_no*prop_table$prec_no)/(prop_table$prop_true_no+prop_table$prec_no)

# verify not having NA metrics at start or end of data 
tail(prop_table)
# we have typically some NA in the precision and F1 at the boundary..put,impute 1,0 respectively 

library(Hmisc)
#impute NA as 0, this occurs typically for precision
prop_table$prec_no=impute(prop_table$prec_no, 1)
prop_table$F1=impute(prop_table$F1, 0)
tail(prop_table, n=10)

colnames(prop_table)

# drop counts, PLOT only metrics
prop_table2 = prop_table[,-c(4:8)] 
tail(prop_table2, n=11)

# plot measures vs soglia##########
# before we must impile data vertically: one block for each measure
library(dplyr)
library(tidyr)

gathered=prop_table2 %>%
  gather(x, y, prop_true_no:F1)

head(gathered)

# plot measures 
library(ggplot2)
gathered %>%
  ggplot(aes(x = threshold, y = y, color = x)) +
  geom_point() +
  geom_line() +
  scale_color_brewer(palette = "Set1") +
  labs(y = "measures",
       color = "no: event\nyes: nonevent")


# con gradient boosting

df2=data.frame(cbind(data_validation$Bankrupt , data_validation$gb_pred0_true, data_validation$gb_pred1_true))
head(df2)
colnames(df2)=c("Bankrupt","ProbNO","ProbYES")
head(df2)


df2=df2[,1:2]
df2$Bankrupt=ifelse(df2$Bankrupt ==1, "no","yes")
df2$Bankrupt <- as.factor(df2$Bankrupt)
head(df2)

str(df2)
# one factor and one numeric

# create a cycle: for each ProbM of validation set create confusion matrices (TP, FN....) and measure of interest (TPR, FPR, )##########

library(dplyr)

thresholds <- seq(from = 0, to = 1, by = 0.005)
prop_table <- data.frame(threshold = thresholds, prop_true_no = NA,  prop_true_yes = NA, true_no = NA,  true_yes = NA ,fn_no=NA)

for (threshold in thresholds) {
  pred <- ifelse(df2$ProbNO > threshold, "no", "yes")  # be careful here!!!
  pred_t <- ifelse(pred == df2$Bankrupt, TRUE, FALSE)
  
  group <- data.frame(df2, "pred" = pred_t) %>%
    group_by(Bankrupt, pred) %>%
    dplyr::summarise(n = n())
  
  group_no <- filter(group, Bankrupt == "no")
  
  true_no=sum(filter(group_no, pred == TRUE)$n)
  prop_no <- sum(filter(group_no, pred == TRUE)$n) / sum(group_no$n)
  
  prop_table[prop_table$threshold == threshold, "prop_true_no"] <- prop_no
  prop_table[prop_table$threshold == threshold, "true_no"] <- true_no
  
  fn_no=sum(filter(group_no, pred == FALSE)$n)
  
  prop_table[prop_table$threshold == threshold, "fn_no"] <- fn_no
  
  
  group_yes <- filter(group, Bankrupt == "yes")
  
  true_yes=sum(filter(group_yes, pred == TRUE)$n)
  prop_yes <- sum(filter(group_yes, pred == TRUE)$n) / sum(group_yes$n)
  
  prop_table[prop_table$threshold == threshold, "prop_true_yes"] <- prop_yes
  prop_table[prop_table$threshold == threshold, "true_yes"] <- true_yes
  
}

head(prop_table, n=10)

# prop_true_no = sensitivity
# prop_true_yes = specificicy

# calculate other missing measures

# n of observations of the validation set    
prop_table$n=nrow(data_validation)

# false positive   by difference of   n and            tn,                 tp,         fn, 
prop_table$fp_no=nrow(data_validation)-prop_table$true_yes-prop_table$true_no-prop_table$fn_no

# find accuracy
prop_table$acc=(prop_table$true_yes+prop_table$true_no)/nrow(data_validation)

# find precision
prop_table$prec_no=prop_table$true_no/(prop_table$true_no+prop_table$fp_no)

# find F1 =2*(prec*sens)/(prec+sens)

prop_table$F1=2*(prop_table$prop_true_no*prop_table$prec_no)/(prop_table$prop_true_no+prop_table$prec_no)

# verify not having NA metrics at start or end of data 
tail(prop_table)
# we have typically some NA in the precision and F1 at the boundary..put,impute 1,0 respectively 

library(Hmisc)
#impute NA as 0, this occurs typically for precision
prop_table$prec_no=impute(prop_table$prec_no, 1)
prop_table$F1=impute(prop_table$F1, 0)
tail(prop_table, n=10)

colnames(prop_table)

# drop counts, PLOT only metrics
prop_table2 = prop_table[,-c(4:8)] 
tail(prop_table2, n=11)

# plot measures vs soglia##########
# before we must impile data vertically: one block for each measure
library(dplyr)
library(tidyr)

gathered=prop_table2 %>%
  gather(x, y, prop_true_no:F1)

head(gathered)

# plot measures 
library(ggplot2)
gathered %>%
  ggplot(aes(x = threshold, y = y, color = x)) +
  geom_point() +
  geom_line() +
  scale_color_brewer(palette = "Set1") +
  labs(y = "measures",
       color = "no: event\nyes: nonevent")


# con lasso

df3=data.frame(cbind(data_validation$Bankrupt , data_validation$lasso_pred0_true, data_validation$lasso_pred1_true))
head(df3)
colnames(df3)=c("Bankrupt","ProbNO","ProbYES")
head(df3)


df3=df3[,1:2]
df3$Bankrupt=ifelse(df3$Bankrupt ==1, "no","yes")
df3$Bankrupt <- as.factor(df3$Bankrupt)
head(df3)


str(df3)
# one factor and one numeric

# create a cycle: for each ProbM of validation set create confusion matrices (TP, FN....) and measure of interest (TPR, FPR, )##########

library(dplyr)

thresholds <- seq(from = 0, to = 1, by = 0.005)
prop_table <- data.frame(threshold = thresholds, prop_true_no = NA,  prop_true_yes = NA, true_no = NA,  true_yes = NA ,fn_no=NA)

for (threshold in thresholds) {
  pred <- ifelse(df3$ProbNO > threshold, "no", "yes")  # be careful here!!!
  pred_t <- ifelse(pred == df3$Bankrupt, TRUE, FALSE)
  
  group <- data.frame(df3, "pred" = pred_t) %>%
    group_by(Bankrupt, pred) %>%
    dplyr::summarise(n = n())
  
  group_no <- filter(group, Bankrupt == "no")
  
  true_no=sum(filter(group_no, pred == TRUE)$n)
  prop_no <- sum(filter(group_no, pred == TRUE)$n) / sum(group_no$n)
  
  prop_table[prop_table$threshold == threshold, "prop_true_no"] <- prop_no
  prop_table[prop_table$threshold == threshold, "true_no"] <- true_no
  
  fn_no=sum(filter(group_no, pred == FALSE)$n)
  
  prop_table[prop_table$threshold == threshold, "fn_no"] <- fn_no
  
  
  group_yes <- filter(group, Bankrupt == "yes")
  
  true_yes=sum(filter(group_yes, pred == TRUE)$n)
  prop_yes <- sum(filter(group_yes, pred == TRUE)$n) / sum(group_yes$n)
  
  prop_table[prop_table$threshold == threshold, "prop_true_yes"] <- prop_yes
  prop_table[prop_table$threshold == threshold, "true_yes"] <- true_yes
  
}

head(prop_table, n=10)

# prop_true_no = sensitivity
# prop_true_yes = specificicy

# calculate other missing measures

# n of observations of the validation set    
prop_table$n=nrow(data_validation)

# false positive   by difference of   n and            tn,                 tp,         fn, 
prop_table$fp_no=nrow(data_validation)-prop_table$true_yes-prop_table$true_no-prop_table$fn_no

# find accuracy
prop_table$acc=(prop_table$true_yes+prop_table$true_no)/nrow(data_validation)

# find precision
prop_table$prec_no=prop_table$true_no/(prop_table$true_no+prop_table$fp_no)

# find F1 =2*(prec*sens)/(prec+sens)

prop_table$F1=2*(prop_table$prop_true_no*prop_table$prec_no)/(prop_table$prop_true_no+prop_table$prec_no)

# verify not having NA metrics at start or end of data 
tail(prop_table)
# we have typically some NA in the precision and F1 at the boundary..put,impute 1,0 respectively 

library(Hmisc)
#impute NA as 0, this occurs typically for precision
prop_table$prec_no=impute(prop_table$prec_no, 1)
prop_table$F1=impute(prop_table$F1, 0)
tail(prop_table, n=10)

colnames(prop_table)

# drop counts, PLOT only metrics
prop_table2 = prop_table[,-c(4:8)] 
tail(prop_table2, n=15)

# plot measures vs soglia##########
# before we must impile data vertically: one block for each measure
library(dplyr)
library(tidyr)

gathered=prop_table2 %>%
  gather(x, y, prop_true_no:F1)

head(gathered)

# plot measures 
library(ggplot2)
gathered %>%
  ggplot(aes(x = threshold, y = y, color = x)) +
  geom_point() +
  geom_line() +
  scale_color_brewer(palette = "Set1") +
  labs(y = "measures",
       color = "no: event\nyes: nonevent")


# con random forest
df$decision=ifelse(df$ProbNO>0.995,"no","yes")
#valid.df$decision=ifelse(df$ProbM>0.65,"M","R")
head(df)

table(df$Bankrupt,df$decision)
# use function of Caret
confusionMatrix(as.factor(df$decision),as.factor(df$Bankrupt), positive = "no")

# con pls
df1$decision=ifelse(df1$ProbNO>0.97,"no","yes")
#valid.df$decision=ifelse(df$ProbM>0.65,"M","R")
head(df1)

table(df1$Bankrupt,df1$decision)
# use function of Caret
confusionMatrix(as.factor(df1$decision),as.factor(df1$Bankrupt), positive = "no")

# con gradient boosting
df2$decision=ifelse(df2$ProbNO>0.995,"no","yes")
#valid.df$decision=ifelse(df$ProbM>0.65,"M","R")
head(df2)

table(df2$Bankrupt,df2$decision)
# use function of Caret
confusionMatrix(as.factor(df2$decision),as.factor(df2$Bankrupt), positive = "no")

# con lasso
df3$decision=ifelse(df3$ProbNO>0.93,"no","yes")
#valid.df$decision=ifelse(df$ProbM>0.65,"M","R")
head(df3)

table(df3$Bankrupt,df3$decision)
# use function of Caret
confusionMatrix(as.factor(df3$decision),as.factor(df3$Bankrupt), positive = "no")

df3$decision1=ifelse(df3$ProbNO>0.95,"no","yes")
#valid.df$decision=ifelse(df$ProbM>0.65,"M","R")
head(df3)

table(df3$Bankrupt,df3$decision1)
# use function of Caret
confusionMatrix(as.factor(df3$decision1),as.factor(df3$Bankrupt), positive = "no")

df3$decision2=ifelse(df3$ProbNO>0.98,"no","yes")
#valid.df$decision=ifelse(df$ProbM>0.65,"M","R")
head(df3)

table(df3$Bankrupt,df3$decision2)
# use function of Caret
confusionMatrix(as.factor(df3$decision2),as.factor(df3$Bankrupt), positive = "no")

df3$decision3=ifelse(df3$ProbNO>0.995,"no","yes")
#valid.df$decision=ifelse(df$ProbM>0.65,"M","R")
head(df3)

table(df3$Bankrupt,df3$decision3)
# use function of Caret
confusionMatrix(as.factor(df3$decision3),as.factor(df3$Bankrupt), positive = "no")


# step 4
prob_old = predict(random_forest, data_score, "prob")[,2]
rf_pred_yes_score<-as.numeric(prob_old) 
rf_pred_no_score = 1 - rf_pred_yes_score 
rho1 = 0.5; rho0 = 0.5; true0 = 0.968; true1 = 0.032 
rf_den_score = rf_pred_yes_score*(true1/rho1)+rf_pred_no_score*(true0/rho0) 
data_score$rf_pred1_true = rf_pred_yes_score*(true1/rho1)/rf_den_score
data_score$rf_pred0_true = rf_pred_no_score*(true0/rho0)/rf_den_score

data_score$pred_y=ifelse(data_score$rf_pred0_true>0.995, "no","yes")
head(data_score)

confusionMatrix(as.factor(data_score$pred_y),as.factor(data_score$Bankrupt), positive = "no")










