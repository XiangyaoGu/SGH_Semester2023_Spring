# dataset 'Mushroom'
# UCI LINK: https://archive.ics.uci.edu/ml/datasets/mushroom

install.packages('caret') # Feature Selection
install.packages('klaR') # NavieBayes() 
install.packages('pROC') # Draw ROC curve
library(caret)
library(klaR)
library(pROC)

###################
#  Prepare data   #
###################

mushroom <- read.csv('agaricus_lepiota.csv')

columnnames <- c('type','cap.shape','cap.surface','cap.color','bruises','odor','gill.attachment',
                 'gill.spacing','gill.size','gill.color','stalk.shape','stalk.root',
                 'stalk.surface.above.ring','stalk.surface.below.ring','stalk.color.above.ring',
                 'stalk.color.below.ring','veil.type','veil.color','ring.number','ring.type',
                 'spore.print.color','population','habitat')
names(mushroom) <- columnnames
View(mushroom)


# Split dataset into train and test set
# Select first 14 variables for feature selection
# Since the total 22 variables can be quite time consuming
set.seed(12)
index <- sample(1:nrow(mushroom), size = 0.75*nrow(mushroom))
train <- mushroom[index,0:15]
test <- mushroom[-index,0:15]

# Check whether the sampling is consistent with the population
prop.table(table(mushroom$type))
# > prop.table(table(mushroom$type))  
#            e         p 
#        0.5180352 0.4819648          

prop.table(table(train$type))
# > prop.table(table(train$type))  
#            e         p 
#        0.5205187 0.4794813          

prop.table(table(test$type))
# > prop.table(table(test$type))  
#            e         p 
#        0.5105859 0.4894141          

# The proportion is roughly the same
# It can be considered that the sampling results could reflect the overall situation
# and further modeling and testing can be carried out.


#########################
#   Feature selection   #
#########################

# We use the rfe() from caret pkg
# A simple backwards selection, a.k.a. recursive feature elimination (RFE)
??rfe
??rfeControl

# Construct the control parameters of the rfe()
# 5-fold cross-validation sampling method, draw 5 groups of sample,with random forest
rfeControls_rf <- rfeControl(
  functions = rfFuncs, #/lmFuncs
  method = 'cv',
  number = 5)

# Very important step
# Change the target into 2-level 
X_train_ = train[,-1]
y_train_ = as.factor(unlist(train[,1]))

# feature selection using rfe()	
# cost around 2 mins
fs_nb <- rfe(x = X_train_,
             y = y_train_,
             #sizes=seq(2,22,2),
             size=seq(2,14,2),
             rfeControl = rfeControls_rf)


fs_nb
plot(fs_nb, type = c('g','o'))
fs_nb$optVariables


##################################
#   Modeling with NaiveBayes()   #
##################################

??NaiveBayes

# Reconstruct the trainset with selected variables
vars <- c('type',fs_nb$optVariables)
train[,1] <- y_train_
fit <- NaiveBayes(type ~ ., data = train[,vars])
View(fit)
fit$apriori
# get the conditional probability table for the "odor" variable
odor_table <- fit$tables[['odor']]
# convert to barplot
barplot(as.matrix(odor_table), beside = TRUE, col = c("grey", "darkgreen"), ylim = c(0, 1.2),
        names.arg = c("a", "c", "f", "l", "m", "n", "p", "s", "y"),
        xlab = "odor", ylab = "Probability", main = "Conditional Probability of odor")
legend("topright", legend = c("edible", "poisonous"), fill = c("grey", "darkgreen"))
                                                               

# Prediction
pred <- predict(fit, newdata = test[,vars][,-1])

################
#  Evaluation  #
################

# Confusion Matrix
freq <- table(pred$class, test[,1])
freq

# Evaluation measure 1: accuracy
accuracy <- sum(diag(freq))/sum(freq)
accuracy


# Evaluation measure 2: ROC curve
names(pred)
# [1] "class"     "posterior"

# select class 'p' as positive
# FPR/TPR
modelroc <- roc(test[,1] == "p", 
                pred$posterior[,2])

# Draw ROC and calculate AUC
plot(modelroc, print.auc = TRUE, auc.polygon = TRUE, 
     grid = c(0.1,0.2), grid.col = c('green','red'),
     max.auc.polygon = TRUE, auc.polygon.col = 'steelblue')
# AUC:0.995

