rm (list=ls ())#clear env
set.seed(123)#set seed so data is repoducible
#Installations and librarieS
#install.packages('tidyverse')
#install.packages('vctrs')
#install.packages('sentimentr')
#install.packages("wordcloud")
library(caret)
library(dplyr)
library(ggplot2)
library(glue)
library(scales)
library(pROC)
library(stringr)
library(sentimentr)
library(tidyr)
library(tidytext)
library(tidyverse)
library(viridis)
library(wordcloud)
library(reshape2)
library(syuzhet)
library(ROCR)
library(ggplot2)
library(caret)
library(glmnet)
require(methods)
require("tm")

#==================READING IN AND PREPPING DATA============================================
#function to remove clothing-specific stop words not identified by sentimentr package
clothingstopwords <- c("top","tops","fall","falls","worn","ruffle","tank","buckle","bust")
remove_stop <- function(str) {
  str<-removeWords(str,clothingstopwords)
}
#read in data
consumerdata <- read.csv('finaldata.csv')
#consumerdata<-consumerdata[1:100,]#just for testing purposes atm

#remove clothing-specific stop words not identified by sentimentr package
consumerdata<-apply(consumerdata,1:2,FUN=remove_stop )
consumerdata<- as.data.frame(consumerdata)
#consumerdata[,c('Review_Text','Title')]

#remove rows with empty titles/reviewscontent
consumerdata <- with(consumerdata, consumerdata[!(Review_Text == "" | is.na(Review_Text)), ])
consumerdata <- with(consumerdata, consumerdata[!(Title == "" | is.na(Title)), ])

#re code some variables
consumerdata <- within(consumerdata, {
  Age = as.numeric(Age)#1 = female 2 = male
  Rating <- factor(Rating, levels=c(1,2,3,4,5))
  Recommended_IND <- factor(Recommended_IND)
  Positive_Feedback_Count = as.numeric(Positive_Feedback_Count)
  Department_Name <- factor(Department_Name)
  Class_Name <- factor(Class_Name)
})

#get sentiment_scores
consumerdata$afin_score_review=get_sentiment(consumerdata$Review_Text)
consumerdata$afin_score_title=get_sentiment(consumerdata$Title)
consumerdata$afin_score_review <-as.numeric(consumerdata$afin_score_review)
consumerdata$afin_score_title <-as.numeric(consumerdata$afin_score_title)


#word count column - disregarded in final model
#consumerdata$wordcount=sapply(consumerdata$Review_Text , wordcount)
#consumerdata$wordcount <-as.numeric(consumerdata$wordcount)
#removing unnecessary columns 
#cols_remove <- c("Title","X","Review_Text","Department_Name","Class_Name","Rating")
#consumerdata=consumerdata[, !(colnames(consumerdata) %in% cols_remove)]


#--------------------AFIN GLM---------------------------------------------
#split into 80% training data and 20% test data
smp_size <- floor(0.80 * nrow(consumerdata))
train_ind <- sample(seq_len(nrow(consumerdata)), size = smp_size)
train_afin <- consumerdata[train_ind, ]
test_afin <- consumerdata[-train_ind, ]


#looking at correlations
cor(train_afin$Age,train_afin$afin_score_review,method = c("pearson"))#close to no correlation

#fitting a GLM model
formula <- Recommended_IND~afin_score_title+afin_score_review+Positive_Feedback_Count
fit <- glm( formula  , family=binomial, data = train_afin)
summary(fit)

# ROC and Performance function
mypredict = predict(fit,test_afin, type="response")
ROCRpred = prediction(mypredict,test_afin$Recommended_IND)
ROCRperf = performance(ROCRpred, "tpr", "fpr")
# Plot ROC curve
plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7),main = "ROC - receiver operating characteristic curve,")
#each one actually has a very high occurence of false positives....

#zooming in to look at the thresholds
plot(ROCRperf,
     xlim=c(0.1,1),
     ylim=c(0.4,1),colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,3),main = "ROC - receiver operating characteristic curve,")


table_mat <- table(test_afin$Recommended_IND, mypredict > 0.52)
table_mat
sum(diag(table_mat))/sum(table_mat)
#0.52 > 0.8528379

#SENSITIVIY V SPRECIFITY GRAPH 1
#source - https://stackoverflow.com/questions/23240182/deciding-threshold-for-glm-logistic-regression-model-in-r
predictions <- prediction(mypredict,test_afin$Recommended_IND)

plot(unlist(performance(predictions, "sens")@x.values), unlist(performance(predictions, "sens")@y.values), 
     type="l", lwd=2, ylab="Sensitivity", xlab="Cutoff")
par(new=TRUE)
plot(unlist(performance(predictions, "spec")@x.values), unlist(performance(predictions, "spec")@y.values), 
     type="l", lwd=2, col='red', ylab="", xlab="")
axis(4, at=seq(0,1,0.2),labels=z)
mtext("Specificity",side=4, padj=-2, col='red')



#SENSITIVIY V SPRECIFITY GRAPH 2
sens <- data.frame(x=unlist(performance(predictions, "sens")@x.values), 
                   y=unlist(performance(predictions, "sens")@y.values))
spec <- data.frame(x=unlist(performance(predictions, "spec")@x.values), 
                   y=unlist(performance(predictions, "spec")@y.values))

sens %>% ggplot(aes(x,y)) + 
  geom_line() + 
  geom_line(data=spec, aes(x,y,col="red")) +
  scale_y_continuous(sec.axis = sec_axis(~., name = "Specificity")) +
  labs(x='Cutoff', y="Sensitivity") +
  theme(axis.title.y.right = element_text(colour = "red"), legend.position="none") 

sens = cbind(unlist(performance(predictions, "sens")@x.values), unlist(performance(predictions, "sens")@y.values))
spec = cbind(unlist(performance(predictions, "spec")@x.values), unlist(performance(predictions, "spec")@y.values))
THRESHOLD = sens[which.min(apply(sens, 1, function(x) min(colSums(abs(t(spec) - x))))), 1]



#source = https://github.com/ethen8181/machine-learning/blob/master/unbalanced/unbalanced_code/unbalanced_functions.R
AccuracyCutoffInfo <- function( train, test, predict, actual )
{
  # change the cutoff value's range as you please 
  cutoff <- seq( .4, .8, by = .05 )
  
  accuracy <- lapply( cutoff, function(c)
  {
    # use the confusionMatrix from the caret package
    cm_train <- confusionMatrix( as.numeric( train[[predict]] > c ), train[[actual]] )
    cm_test  <- confusionMatrix( as.numeric( test[[predict]]  > c ), test[[actual]]  )
    
    dt <- data.table( cutoff = c,
                      train  = cm_train$overall[["Accuracy"]],
                      test   = cm_test$overall[["Accuracy"]] )
    return(dt)
  }) %>% rbindlist()
  
  # visualize the accuracy of the train and test set for different cutoff value 
  # accuracy in percentage.
  accuracy_long <- gather( accuracy, "data", "accuracy", -1 )
  
  plot <- ggplot( accuracy_long, aes( cutoff, accuracy, group = data, color = data ) ) + 
    geom_line( size = 1 ) + geom_point( size = 3 ) +
    scale_y_continuous( label = percent ) +
    ggtitle( "Train/Test Accuracy for Different Cutoff" )
  
  return( list( data = accuracy, plot = plot ) )
}





#Accuracy graph 1
#accuracy for test
cutoffs <- seq(0.1,0.9,0.1)
accuracy_test <- NULL
for (i in seq(along = cutoffs)){
  table_mat <- table(test_afin$Recommended_IND, mypredict > cutoffs[i])
  accuracy_test <- c(accuracy_test,sum(diag(table_mat))/sum(table_mat))
  }
#accuracy for train
mypredict_train = predict(fit,train_afin, type="response")
accuracy_train <- NULL
for (i in seq(along = cutoffs)){
  table_mat <- table(train_afin$Recommended_IND, mypredict_train > cutoffs[i])
  accuracy_train <- c(accuracy_train,sum(diag(table_mat))/sum(table_mat))
}

#rough plot them both 
plot(cutoffs, accuracy_test, pch =19,type='b',col= "steelblue",
     main ="Logistic Regression", xlab="Cutoff Level", ylab = "Accuracy %")
lines(cutoffs,accuracy_train,col="green",pch =19,type='b')
#accuracy level of 0.5 is chosen
abline(v=0.52)

threshold = 0.52

#train_afin

train_afin$prediction <- predict( fit, newdata = train_afin, type = "response" )
test_afin$prediction  <- predict( fit, newdata = test_afin , type = "response" )

train_afin <- train_afin[ , c("Recommended_IND","prediction")]
test_afin <- test_afin[ , c("Recommended_IND","prediction")]


#Accuracy graph 2
accuracy_info <- AccuracyCutoffInfo( train = train_afin, test = test_afin, 
                                     predict = "prediction", actual = "Recommended_IND" )
# define the theme for the next plot
ggthemr("light")
accuracy_info$plot








