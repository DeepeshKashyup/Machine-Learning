# Simple Linear Regression

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)

split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting the simple regression to the training set 

regressor = lm(formula = Salary ~ YearsExperience,data = training_set)

# Evaluating the model
summary(regressor)

# Predicting the test set result

y_pred = predict(regressor,newdata = test_set)

# Visualizing the training set result
##install.packages('ggplot2')
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience,y=training_set$Salary),
             color ='red') + 
  geom_line(aes(x=training_set$YearsExperience,y=predict(regressor,newdata = training_set)),
            color = 'blue') + 
  ggtitle("Salary vs Experience(Training Set)") + 
  xlab("Years of Experience") + 
  ylab("Salary")



# Visualizing the test set result

library(ggplot2)
ggplot() + 
  geom_point(aes(x = test_set$YearsExperience,y=test_set$Salary),
             color ='red') + 
  geom_line(aes(x=training_set$YearsExperience,y=predict(regressor,newdata = training_set)),
            color = 'blue') + 
  ggtitle("Salary vs Experience(Test Set)") + 
  xlab("Years of Experience") + 
  ylab("Salary")