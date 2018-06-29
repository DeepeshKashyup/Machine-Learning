# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)


# Fitting Linear Regression Model to the Dataset
Lin_reg = lm(formula = Salary ~ .,data= dataset)
summary(Lin_reg)

dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4

# Fitting Polynomial Regression Model to the Dataset
Poly_reg = lm(formula = Salary ~ ., data = dataset)
summary(Poly_reg)

library(ggplot2)
# Visualize linear regression results
ggplot() + geom_point(aes(x = dataset$Level,y=dataset$Salary),color = 'Red') +
  geom_line(aes(x=dataset$Level,y= predict(Lin_reg,newdata = dataset)),color='Blue') +
  xlab('Levels') + ylab('Salary') + ggtitle('Truth or Bluff(Linear Regression)')

# Visualize polynomial regression results

ggplot() + geom_point(aes(x = dataset$Level,y=dataset$Salary),color = 'Red') +
  geom_line(aes(x=dataset$Level,y= predict(Poly_reg,newdata = dataset)),color='Blue') +
  xlab('Levels') + ylab('Salary') + ggtitle('Truth or Bluff(Polynomial Regression)')

# Predicting new result with Linear Regression

y_pred1 = predict(Lin_reg,newdata = data.frame(Level=6.5))

# Predicting new result with Polynomial Regression

y_pred2 = predict(Poly_reg,newdata = data.frame(Level=6.5,
                                               Level2=6.5^2,
                                               Level3=6.5^3,
                                               Level4=6.5^4))