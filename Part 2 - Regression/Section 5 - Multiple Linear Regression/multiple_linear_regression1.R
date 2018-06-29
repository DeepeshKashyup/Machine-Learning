# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding Categorical Variables

dataset$State = factor(dataset$State,
                         levels = c('New York', 'California', 'Florida'),
                         labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting multiple Linear regression to the training set 
regressor = lm(formula = Profit ~ .
               ,data= training_set)


#regressor = lm(formula = Profit ~ R.D.Spend
#               ,data= training_set)
summary(regressor)

# predicting the test set result

y_pred <- predict(regressor,newdata = test_set)

# Building the optimal model using backward Elimination 
# using complete dataset to have better picture of significance levels of independent variables
regressor <- lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                data = dataset)
summary(regressor)

# Step 2  remove state since it has highest P-value

regressor <- lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
                data = dataset)
summary(regressor)

# Step 3 remove  Administration since it has highest P-value

regressor <- lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
                data = dataset)
summary(regressor)

# Step 4 remove  Marketing Spend since it has highest P-value
# However, we can keep marketing spend as p-value is very close to 0.05
# and it is slightly significant

regressor <- lm(formula = Profit ~ R.D.Spend,
                data = dataset)
summary(regressor)