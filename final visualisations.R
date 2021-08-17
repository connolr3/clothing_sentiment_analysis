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
theme_set(theme_minimal())
resultcolors <- c("cadetblue3","coral3" ,"darkseagreen")
setHook("plot.new", function() par(col = "black"))
theme_set(theme_minimal())


#==================READING IN AND FURTHER PREPPING DATA============================================
#function to remove clothing-specific stop words not identified by sentimentr package
#i.e. buckle seen as negative in sentiment package however in the context of clothing data it isn't necessarily postive or negative
clothingstopwords <- c("top","tops","fall","falls","worn","ruffle","tank","buckle","bust")
remove_stop <- function(str) {
  str<-removeWords(str,clothingstopwords)
}

#read in data
consumerdata <- read.csv('finaldata.csv')

#remove clothing-specific stop words not identified by sentimentr package
consumerdata<-apply(consumerdata,1:2,FUN=remove_stop)

#convert back to df
consumerdata<- as.data.frame(consumerdata)

#remove rows with empty review titles/content
consumerdata <- with(consumerdata, consumerdata[!(Review_Text == "" | is.na(Review_Text)), ])
consumerdata <- with(consumerdata, consumerdata[!(Title == "" | is.na(Title)), ])

#re code variables where necessary
consumerdata <- within(consumerdata, {
  Age = as.numeric(Age)#1 = female 2 = male
  Rating <- factor(Rating, levels=c(1,2,3,4,5))
  Recommended_IND <- factor(Recommended_IND)
  Positive_Feedback_Count = as.numeric(Positive_Feedback_Count)
  Department_Name <- factor(Department_Name)
  Class_Name <- factor(Class_Name)
})

#get sentiment_scores and add as column
#get_sentiment retrieves a sentiment score from the afin sentiment library
consumerdata$afin_score_review=get_sentiment(consumerdata$Review_Text)
consumerdata$afin_score_title=get_sentiment(consumerdata$Title)
consumerdata$afin_score_review <-as.numeric(consumerdata$afin_score_review)
consumerdata$afin_score_title <-as.numeric(consumerdata$afin_score_title)

#get word count and add as column
consumerdata$wordcount=sapply(consumerdata$Review_Text , wordcount)
consumerdata$wordcount <-as.numeric(consumerdata$wordcount)

#removing review title and content columns as these are reflected in new afin columns 
#cols_remove <- c("Title","X","Review_Text")
#consumerdata=consumerdata[, !(colnames(consumerdata) %in% cols_remove)]


#=================PREP FOR REGRESSION - LASSO===========================
#set up data in format glmnet can use (must have dummy vars for categorical predictors)
#removed intercept term when making xfactors
xfactors <- model.matrix(Recommended_IND ~ Department_Name+Class_Name, data=consumerdata)[, -1]
xnonfactors<- subset(consumerdata, select = c("Age","wordcount","Positive_Feedback_Count","afin_score_review", "afin_score_title"))
prepared.dat <- cbind(xnonfactors, consumerdata$Recommended_IND)
names(prepared.dat)[length(names(prepared.dat))] <- c("Recommended_IND")

## SPLITTING DATA INTO training, validation, and test (60%, 20%, 20% split)
smp_size <- floor(0.60 * nrow(prepared.dat))
train_ind <- sample(seq_len(nrow(prepared.dat)), size = smp_size)
train <- prepared.dat[train_ind, ]
remainingdata <- prepared.dat[-train_ind, ]
smp_size <- floor(0.50 * nrow(remainingdata))#50% of the remaining 40% is 20% of the whole data
test_ind <- sample(seq_len(nrow(remainingdata)), size = smp_size)
test <- remainingdata[test_ind, ]
validation <- remainingdata[-test_ind, ]

#create predictor matrix and outcome vector for training, validation, and test data
train.X <- data.matrix(within(train, rm(Recommended_IND)))
val.X <- data.matrix(within(validation, rm(Recommended_IND)))
test.X <- data.matrix(within(test, rm(Recommended_IND)))
train.y <- train$Recommended_IND
val.y <- validation$Recommended_IND
test.y <- test$Recommended_IND


#==================lasso model..... used in variable selction====================================
#ridge/elastic net also analysed but removed as lasso most accurate measure

#cross-validate to tune lambda for ridge and lasso
cvlasso <- cv.glmnet(train.X, train.y, family="binomial", alpha=1, nlambda=20, type.measure="deviance")

#fit models with final lambda
lassomod <- glmnet(train.X, train.y, family="binomial", alpha = 1, lambda = cvlasso$lambda.1se)

#fit lasso 
fit.lasso <- predict(lassomod, val.X, type="response")

#get threshold
thresh.l <- coords(roc(val.y, as.vector(fit.lasso)), "best", best.method="youden", transpose=TRUE, ret="threshold")

#predict classifications in test data)
final.l <- predict(lassomod, test.X, type="response")

#use caret to see various measures of performance
class.lasso <- as.factor(ifelse(final.l <= thresh.l, "0", "1"))
cfl = confusionMatrix(class.lasso, test.y, positive = "1")

accuracy_l<-cfl$overall['Accuracy']
coef(lassomod)
#afin_score_review and afin_score_title are selected as variables to use
#however, the GLM gives a better accurcay with these variables

#Lasso regression gives us the releveant coefficients.... now using GLM model as more accurate
#seeing as we ended up with only 2 predictor variables.....




#--------------------AFIN scores with GLM model---------------------------------------------
#split into 80% training data and 20% test data
smp_size <- floor(0.80 * nrow(consumerdata))
train_ind <- sample(seq_len(nrow(consumerdata)), size = smp_size)
train_afin <- consumerdata[train_ind, ]
test_afin <- consumerdata[-train_ind, ]

#fitting a GLM model
formula <- Recommended_IND~afin_score_title+afin_score_review
fit <- glm( formula  , family=binomial, data = train_afin)
summary(fit)

#choosing threshold.........
mypredict = predict(fit,test_afin, type="response")
ROCRpred = prediction(mypredict,test_afin$Recommended_IND)
ROCRperf = performance(ROCRpred, "tpr", "fpr")
# Plot ROC curve
plot(ROCRperf,
     xlim=c(0.1,1),
     ylim=c(0.4,1),colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,3),main = "ROC - Receiver Operating Characteristic Curve")
#from ROC curve threhold of between 0.5 and 0.7 look to be good candiates
#the roc curve also rveeals many cutoff points give a high FP rate.
#use another method to further refine threshold.....


#plot threshold versus accurcay given for both training and testing data should give a better indication of thershold to choose
#Accuracy graph 
#compute accuracy for test
cutoffs <- seq(0.1,0.9,0.1)
accuracy_test <- NULL
for (i in seq(along = cutoffs)){
  table_mat <- table(test_afin$Recommended_IND, mypredict > cutoffs[i])
  accuracy_test <- c(accuracy_test,sum(diag(table_mat))/sum(table_mat))
}
#compute accuracy for train
mypredict_train = predict(fit,train_afin, type="response")
accuracy_train <- NULL
for (i in seq(along = cutoffs)){
  table_mat <- table(train_afin$Recommended_IND, mypredict_train > cutoffs[i])
  accuracy_train <- c(accuracy_train,sum(diag(table_mat))/sum(table_mat))
}
#resultcolors <- c("cadetblue3","coral3" ,"darkseagreen")
#rough plot them both 
plot(cutoffs, accuracy_test,pch =19,type='l',col= "coral3",
     main ="Accuracy Achieved", xlab="Threshold", ylab = "Accuracy %")
lines(cutoffs,accuracy_train,col="darkseagreen",pch =19,type='l')
#by inspection - accuracy level of 0.52 is chosen to maximise accuracy
abline(v=0.515,col = "black", lwd = 2)
threshold = 0.52
text(x = 0.4,                   # Add text for mean
     y = 0.7,
     paste("Threshold: ",threshold),
     col = "black",
     cex = 1)

legend(0.08,0.65, legend=c("Test", "Train"),
       col=c("coral3", "darkseagreen"), lty=1:1, cex=0.8)








table_mat <- table(test_afin$Recommended_IND, mypredict > 0.52)
table_mat
sum(diag(table_mat))/sum(table_mat)
#0.52 gives a max accuracy of 0.8528379
specificty = 233/(233+361)#0.3922559
sensitivity =  2703/(2703+121)#0.957153

#and so final model is a glm model using the afin score of the review title and the afin score of the review content
#the final threshold is 0.52... which gives an accuracy of 0.8528379
#Coefficients:
#(Intercept)       -0.48090    
# afin_score_title   2.06610   
# afin_score_review  0.53753    

fitcoefficients = coef(fit)
#title has more bearing than content...



library(ROCR)
library(grid)
library(broom)
library(caret)
library(tidyr)
library(dplyr)
library(scales)
library(ggplot2)
library(ggthemr)
library(ggthemes)
library(gridExtra)
library(data.table)

install.packages("ggthemr")
library(ggthemr)
cm_info <- ConfusionMatrixInfo( data = test_afin, predict = "mypredict", 
                                actual = "left", cutoff = .6 )
ggthemr("flat")
cm_info$plot
















library(jtools)
library(ggstance)
library(broom.mixed)


#effect plot 
effect_plot(fit, pred = Recommended_IND, interval = TRUE, plot.points = TRUE)
plot_summs(fit)




#===================VISUALISATIONS USING BING======================
#transform data into a tidy text format... source - https://www.tidytextmining.com/tidytext.html
testdata=as_tibble(consumerdata)#convert to tibble
#breaks the text into individual tokens = "tokenisation"
#tokenisation: punctuation is ignored and text converted to lower case
token_data=unnest_tokens(testdata,word,Review_Text)

#join df to bing sentiment.... i.e. a column explaining whether words were postive or negative... source - https://www.youtube.com/watch?v=BqNTcewq0k0
CareTibbleBing  <- token_data %>%
  inner_join(get_sentiments("bing"))
#bing will filter out "stop" words i.e. Words without a sentiment value 

CareTibbleBing <-CareTibbleBing %>%
  with(CareTibbleBing, !(word =="issue"))

#Wordcloud
CareTibbleBing %>%
  count(word, sentiment, sort=TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("#F8766D","darkseagreen"),
                   max.words = 100)



#ggplots
resultcolors <- c("cadetblue3","coral3" ,"darkseagreen")



#hist - review content
par(oma=c(0,0,2,0))
hist=hist(consumerdata$afin_score_review,breaks = 70,col = "slategray1",main="Sentiment Score Distribution",xlab="Score")
averageafincontent=mean(consumerdata$afin_score_review)
#color negative reviews as red
ccat = cut(hist$breaks, c(-Inf, 0,  Inf))
plot(hist, col=c("#F8766D","darkseagreen")[ccat],main="Review Content",xlab="Sentiment Score (afin)")
#add a line at average point
abline(v = averageafincontent, col = "black", lwd = 2)
text(x = 4.8,                   # Add text for mean
     y = 900,
     paste("Mean = ",round(averageafincontent,2)),
     col = "black",
     cex = 1.2)

sdcontent = sd(consumerdata$afin_score_review)


consumerdata0 <- consumerdata[ which(consumerdata$Recommended_IND=='0'),]
mean_review_not_recc <- mean(consumerdata0$afin_score_review)
consumerdata1 <- consumerdata[ which(consumerdata$Recommended_IND=='1'),]
mean_review__recc <- mean(consumerdata1$afin_score_review)


consumerdata0 <- consumerdata[ which(consumerdata$Recommended_IND=='0'),]
mean_title_not_recc <- mean(consumerdata0$afin_score_title)
consumerdata1 <- consumerdata[ which(consumerdata$Recommended_IND=='1'),]
mean_title_recc <- mean(consumerdata1$afin_score_title)



#hist - review title
par(oma=c(0,0,2,0))
hist=hist(consumerdata$afin_score_title,breaks = 70,col = "slategray1",main="Sentiment Score Distribution",xlab="Score")
averageafintitle=mean(consumerdata$afin_score_title)

#color negative reviews as red
ccat = cut(hist$breaks, c(-Inf, 0,  Inf))
plot(hist, col=c("#F8766D","darkseagreen")[ccat],main="Review Title",xlab="Sentiment Score (afin)")

#add a line at average point
abline(v = averageafintitle, col = "black", lwd = 2)
text(x = averageafintitle-1,                   # Add text for mean
     y = 4000,
     paste("Mean = ",round(averageafintitle,digits=2)),
     col = "black",
     cex = 1.2)












# colors <- c(rep("red",0), rep("blue",3), rep("orange",3))
# ggplot(data=consumerdata, aes(afin_score_review)) + 
#   geom_histogram(bins = 50,fill="slategray2")+ geom_vline(xintercept = averageafin, linetype="dashed", 
#                                         color = "black", size=1)+
#   ggtitle("Sentiment Distribution") + 
#   theme(plot.title = element_text(size = 15, face = "bold"))+xlab("Afin Score")+
#  geom_vline(xintercept = 0,
#                color = "slategray1", size=1)






#positive feedback plot
ggplot(consumerdata, aes(fill=Recommended_IND, y=Positive_Feedback_Count, x=Positive_Feedback_Count)) + 
  geom_bar(position="stack", stat="identity"  )+
  scale_fill_manual("Recommended", values = resultcolors)+
  xlab("")+ylab("")+
  theme(legend.position = "none",axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank())+
  ggtitle("Positive Feedback Count") + 
  theme(plot.title = element_text(size = 15, face = "bold"))+
  scale_x_continuous(breaks = c(0,10,20,30,40,50,60,70), labels = c("0","10","20","30","40","50","60","70"))+
  theme(axis.text=element_text(size=12),axis.title=element_text(size=20,face="bold",colour = "black"))


















qplot(consumerdata$afin_score_review,
      geom="histogram",bins=40,
      main="Sentiment Distribution",xlab="Score" ,fill = I("slategray3"),col=I("slategray4"))


abline(v = 0, col = "black", lwd = 2)



#Create age bracket column
#https://rstudio-pubs-static.s3.amazonaws.com/222993_e1059369754f419a9360e7b0d431f3e1.html
labs <- c(paste(seq(1, 95, by = 10), seq(1 + 10 - 1, 100 - 1, by = 10),
                sep = "-"), paste(100, "+", sep = ""))
consumerdata$AgeGroup <- cut(consumerdata$Age, breaks = c(seq(0, 100, by = 10), Inf), labels = labs, right = FALSE)
#head(consumerdata[c("Age", "AgeGroup")], 15)






#ggplots
resultcolors <- c("cadetblue3","coral3" ,"darkseagreen")


ggplot(consumerdata, aes(fill=Recommended_IND, y=Department_Name, x=Department_Name)) + 
  geom_bar(position="stack", stat="identity"  )+scale_fill_manual("Recommended", values = resultcolors)+
  xlab("")+ylab("")+theme(legend.position = "none",axis.title.x=element_blank(),
                          axis.text.x=element_blank(),
                          axis.ticks.x=element_blank())+ coord_flip()+theme(text = element_text(size=18))+ggtitle("Department") + 
  theme(plot.title = element_text(size = 20, face = "bold"))
#rating plot
 ratingplot<-ggplot(consumerdata, aes(fill=Recommended_IND, y=Rating, x=Rating)) + 
  geom_bar(position="stack", stat="identity"  )+
  scale_fill_manual("Recommended", values = resultcolors)+
  xlab("")+
  ylab("")+theme(legend.position = "none")+ggtitle("Rating") + 
   theme(plot.title = element_text(size = 20, face = "bold"))+theme(text = element_text(size=18))

ggplot(consumerdata, aes(fill=Recommended_IND, y=Class_Name, x=Class_Name)) + 
  geom_bar(position="stack", stat="identity"  )+scale_fill_manual("Recommended", values = resultcolors,labels=c("No","Yes"))+xlab("Class")+ylab("")

#age group
ggplot(consumerdata, aes(fill=Recommended_IND, y=AgeGroup, x=AgeGroup)) + 
  geom_bar(position="stack", stat="identity"  )+scale_fill_manual("Recommended", values = resultcolors)+
  xlab("")+ylab("")+  theme(legend.position = "none")+ggtitle("Age") + 
  theme(plot.title = element_text(size = 20, face = "bold"))+theme(text = element_text(size=20))+
theme(legend.position = "none",axis.title.y=element_blank(),
      axis.text.y=element_blank(),
      axis.ticks.y=element_blank())+scale_x_discrete(labels = c("","...U30","31-40","41-50","51-60","61-70","71-80","O80...",""))

#positive feedback plot
ggplot(consumerdata, aes(fill=Recommended_IND, y=Positive_Feedback_Count, x=Positive_Feedback_Count)) + 
  geom_bar(position="stack", stat="identity"  )+
  scale_fill_manual("Recommended", values = resultcolors)+
  xlab("")+ylab("")+
  theme(legend.position = "none",axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank())+
  ggtitle("Positive Feedback Count") + 
  theme(plot.title = element_text(size = 15, face = "bold"))+
  scale_x_continuous(breaks = c(0,10,20,30,40,50,60,70), labels = c("0","10","20","30","40","50","60","70"))+
  theme(axis.text=element_text(size=12),axis.title=element_text(size=20,face="bold",colour = "black"))

#age plot
ageplot<-ggplot(consumerdata, aes(fill=Recommended_IND, y=Age, x=Age)) + 
  geom_bar(position="stack", stat="identity"  )+
  scale_fill_manual("Recommended", values = resultcolors)+
  xlab("")+ylab("")+theme(legend.position = "none",axis.title.y=element_blank(),
                             axis.text.y=element_blank(),axis.ticks.y=element_blank())+ggtitle("Age") + 
  theme(plot.title = element_text(size = 20, face = "bold"))+
  scale_x_continuous(breaks = c(20,30,40,50,60,70), labels = c("20","30","40","50","60","70"))+theme(axis.text=element_text(size=12),
                                                                       axis.title=element_text(size=20,face="bold",colour = "black"))
  
  



meanage <-mean(consumerdata$Age)


library(ggpubr)


figure<-ggarrange(ageplot, dpmtplot, ratingplot + rremove("x.text"), 
          labels = c("Age", "Department", "Rating"),
          ncol = 2, nrow = 2)

annotate_figure(figure,top = text_grob("Consumer Data Distribution",, face = "bold", size = 18))

#http://www.sthda.com/english/articles/24-ggpubr-publication-ready-plots/81-ggplot2-easy-way-to-mix-multiple-graphs-on-the-same-page/#:~:text=To%20arrange%20multiple%20ggplot2%20graphs,multiple%20ggplots%20on%20one%20page


#arrange titles annotate_figure source = https://rpkgs.datanovia.com/ggpubr/reference/annotate_figure.html

