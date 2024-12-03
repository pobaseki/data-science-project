#!/usr/bin/env python
# coding: utf-8

# ## Project: Problem Statement - Personal Loan Campaign Modelling
# ## Thera Bank Personal Loan Campaign
#  
# #### Data Description:
# 
# The dataset contains data on 5000 customers. The data include customer demographic information (age, income, etc.), the customer's relationship with the bank (mortgage, securities account, etc.), and the customer response to the last personal loan campaign (Personal Loan). Among these 5000 customers, only 480 (= 9.6%) accepted the personal loan that was offered to them in the earlier campaign.
# 
#  
# 
# #### Domain:Banking
# 
#  
# 
# Context:
# This case is about a bank (Thera Bank) whose management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors). A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. This has encouraged the retail marketing department to devise campaigns with better target marketing to increase the success ratio with a minimal budget.
# 
# ## Attribute Information:
# - ID: Customer ID
# - Age: Customer's age in completed years
# - Experience: #years of professional experience
# - Income: Annual income of the customer. ($000)
# 
# - ZIP Code: Home Address ZIP
# 
# - Family: Family size of the customer
# 
# - CCAvg: Avg. spending on credit cards per month ($000)
# 
# - Education: Education Level. 1: Undergrad; 2: Graduate; 3: Advanced/Professional
# - Mortgage: Value of house mortgage if any. ($000)
# - Personal Loan: Did this customer accept the personal loan offered in the last campaign?
# - Securities Account: Does the customer have a securities account with the bank?
# - CD Account: Does the customer have a certificate of deposit (CD) account with the bank?
# - Online: Does the customer use internet banking facilities?
# - Credit card: Does the customer use a credit card issued by the bank?
#  
# 
# ### Learning Outcomes:
# 
# - Exploratory Data Analysis
# - Preparing the data to train a model
# - Training and making predictions using a classification model
# - Model evaluation
#  
# 
# ### Objective:
# The classification goal is to predict the likelihood of a liability customer buying personal loans.
# 
#  
# 
# ## Steps and tasks:
# 1. Import the datasets and libraries, check datatype, statistical summary, shape, null values or incorrect imputation. (5 marks)
# 2. EDA: Study the data distribution in each attribute and target variable, share your findings (20 marks)
# - Number of unique in each column?
# - Number of people with zero mortgage?
# - Number of people with zero credit card spending per month?
# - Value counts of all categorical columns.
# - Univariate and Bivariate
# - Get data model ready
# 3. Split the data into training and test set in the ratio of 70:30 respectively (5 marks)
# 4. Use the Logistic Regression model to predict whether the customer will take a personal loan or not. Print all the metrics related to evaluating the model performance (accuracy, recall, precision, f1score, and roc_auc_score). Draw a heatmap to display confusion matrix (15 marks)
# 5.Find out coefficients of all the attributes and show the output in a data frame with column names? For test data show all the rows where the predicted class is not equal to the observed class. (10 marks)
# 6. Give conclusion related to the Business understanding of your model? (5 marks)
# 

# In[1]:


#Step 1 - Import the datasets and libraries, check datatype, statistical summary, shape, null values or incorrect imputation.


# #### Import packages

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


file_path = r'C:\Users\ask4d\Documents\Artificial Intelligence _ Machine Learning\Bank Loan Campaign.csv'
bankloan= pd.read_csv(file_path)


# In[5]:


bankloan.head()


# #### Explore Data

# In[6]:


#CHECK DATA TYPE
bankloan.info()


# In[7]:


# Check missing values via heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(bankloan.isna())
plt.show()


# In[8]:


bankloan.shape


# In[13]:


bankloan.isnull()


# In[11]:


# Create summary statistics for numeric fields
bankloan.describe().T


# In[19]:


bankloan[bankloan.duplicated()].count()


# No duplication of variables! 

# #### Exploratory Data Analysis

# In[ ]:


#EDA: Study the data distribution in each attribute and target variable, share your findings


# In[14]:


bankloan.mean()


# In[15]:


bankloan.median()


# In[16]:


bankloan.mode()


# #### Skewness

# In[18]:


bankloan.skew()


# In[20]:


# code here
Q1 = bankloan.quantile(0.25)
Q3 = bankloan.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[21]:


bankloan.max()


# In[22]:


bankloan.min()


# In[23]:


bankloan.var()


# In[24]:


bankloan.std()


# In[25]:


bankloan.cov().T


# In[26]:


bankloan.corr()


# ### Pairplot to check correlations

# In[27]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[28]:


sns.pairplot(bankloan)
plt.show()


# In[ ]:





# ### Check the distribution of each attribute

# In[32]:


import warnings
warnings.filterwarnings('ignore')


# In[33]:


sns.distplot(bankloan['ID'])
plt.show()


# sns.distplot(bankloan['Age'])
# plt.show()

# In[37]:


sns.distplot(bankloan['Experience'])
plt.show()


# In[38]:


sns.distplot(bankloan['Income'])
plt.show()


# In[39]:


sns.distplot(bankloan['ZIP Code'])
plt.show()


# In[40]:


sns.distplot(bankloan['Family'])
plt.show()


# In[41]:


sns.distplot(bankloan['CCAvg'])
plt.show()


# In[42]:


sns.distplot(bankloan['Education'])
plt.show()


# In[43]:


sns.distplot(bankloan['Mortgage'])
plt.show()


# In[44]:


sns.distplot(bankloan['Personal Loan'])
plt.show()


# In[45]:


sns.distplot(bankloan['Securities Account'])
plt.show()


# In[46]:


sns.distplot(bankloan['CD Account'])
plt.show()


# In[47]:


sns.distplot(bankloan['Online'])
plt.show()


# In[48]:


sns.distplot(bankloan['CreditCard'])
plt.show()


# Based on the skweness of the data While experience is not a bell-shaped curve it is very spread out,  Income is right skewed, and based on the data we can see that the average income seems to be about 50-70 thousand dollars. The credit card average is very right skewed with the average credit card being about 2,000 dollars. The mortgage is left-skewed with the average mortgage loan between 0-70,000 dollars. 

# ### Number of unique in each column

# In[50]:


bankloan.nunique()


# ### Number of people with zero mortgage

# In[55]:


bankloan[bankloan['Mortgage'] == 0].shape[0]


# ### Number of people with zero credit card spending per month

# In[56]:


bankloan[bankloan['CCAvg'] == 0].shape[0]


# ### Value counts of all categorical columns

# In[59]:


bankloan['Family'].value_counts()


# In[60]:


bankloan['Education'].value_counts()


# In[61]:


bankloan['Personal Loan'].value_counts()


# In[62]:


bankloan['Securities Account'].value_counts()


# In[63]:


bankloan['CD Account'].value_counts()


# In[64]:


bankloan['Online'].value_counts()


# In[71]:


bankloan['CreditCard'].value_counts()


# ### Univariate Analysis

# In[74]:


plt.hist(bankloan['Age'], bins=50)


# In[87]:


plt.hist(bankloan['Experience'], bins=50)


# In[75]:


plt.hist(bankloan['Income'], bins=50)


# In[77]:


plt.hist(bankloan['CCAvg'], bins=50)


# In[86]:


plt.hist(bankloan['Mortgage'], bins=10)


# ### Bivariate Analysis

# In[88]:


Agecount = pd.crosstab(index=bankloan["Age"], 
                     columns="count")                 
Agecount


# In[92]:


bankloan.groupby(by=['Education'])['Income'].sum().reset_index().sort_values(['Education']).tail(10).plot(x='Education',
                                                                                                           y='Income',
                                                                                                           kind='bar',
                                                                                                           figsize=(15,5))
plt.show()


# In[91]:


bankloan.groupby(by=['Age'])['Income'].sum().reset_index().sort_values(['Age']).tail(10).plot(x='Age',
                                                                                                           y='Income',
                                                                                                           kind='bar',
                                                                                                           figsize=(15,5))
plt.show()


# In[97]:


plt.figure(figsize=(10,5))  # setting the figure size
ax = sns.barplot(x='Education', y='Personal Loan', data=bankloan, palette='muted') 


# In[103]:


figure = plt.figure(figsize=(15,5))

ax = sns.scatterplot(x=bankloan['Age'],y='Income', data=bankloan, size = "Income") # scatter plot


# In[105]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[107]:


plt.figure(figsize=(10,5))
sns.heatmap(bankloan.corr(), annot=True, linewidths=.5, fmt= '.1f', center = 1 )  
plt.show()


# ###  Data Preparation

# In[109]:


bankloan.info()


# ### Looking at the data there is no column with an object data type. 

# In[111]:


X = bankloan.drop('Personal Loan', axis=1)
y = bankloan[['Personal Loan']]

print(X.head())
print(y.head())


# In[112]:


print(X.shape)
print(y.shape)


# In[113]:


X = X.values
y = y.values


# In[114]:


#split the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[115]:


from sklearn.linear_model import LinearRegression
linearregression = LinearRegression()                                    
linearregression.fit(X_train, y_train)                                  

print("Intercept of the linear equation:", linearregression.intercept_) 
print("\nCOefficients of the equation are:", linearregression.coef_)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pred = linearregression.predict(X_test)    


# In[116]:


# Mean Absolute Error
mean_absolute_error(y_test, pred)


# In[117]:


# RMSE
mean_squared_error(y_test, pred)**0.5


# In[118]:


# R2 Squared:
r2_score(y_test, pred)


# In[119]:


# Training Score
linearregression.score(X_train, y_train)


# In[120]:


# Testing score
linearregression.score(X_test, y_test)


# In[121]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': pred.flatten()})
df


# In[134]:


regression_model = LinearRegression()
regression_model.fit(X_train, y_train)


# In[142]:


# Train the logistic regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Make predictions
y_test_pred = logistic_model.predict(X_test)

# Predicted probabilities for ROC AUC
y_test_pred_proba = logistic_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_pred_proba)

# Print evaluation metrics
print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1}')
print(f'ROC AUC Score: {roc_auc}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Plot Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[122]:


df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# # Business Recommendations:
# 
# For Thera Bank whose main goal is to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors some common recommendations would be 
# 
# 1. Variables like Income, Education and having a CD accounts have a strong positive correlation with Personal loan. Therefore the bank should be intentional about campaigning with people that have a strong income, education and with CD's account targeting them with personalized targeted ads. 

# 
# 
