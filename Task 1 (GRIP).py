#!/usr/bin/env python
# coding: utf-8

# # The Sparks foundation (Grip23)

# # By ~ Anish Solanky

# ## Prediction using supervised learning
# ## Linear Regression Model

# ## Installing required Libraries

# In[11]:


get_ipython().system('pip install pandas')


# In[12]:


get_ipython().system('pip install numpy')


# In[13]:


get_ipython().system('pip install matplotlib')


# In[14]:


get_ipython().system('pip install')


# ## Importing necessary Libraries

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ## Read the dataset through url

# In[10]:


data = pd.read_csv("http://bit.ly/w-data")


# In[11]:


data


# In[12]:


data.head(10)


# In[14]:


# Splitting the data into features (X) and labels (y)

from sklearn.model_selection import train_test_split
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


# In[15]:


# Splitting the data into training(80%)and test sets(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ## Now using linear regression model

# In[17]:


from sklearn.linear_model import LinearRegression

# Create an instance of the LinearRegression model
regressor = LinearRegression()

# Fit the model to the training data
regressor.fit(X_train, y_train)

print("Training done.")


# In[18]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# ## After training our algorithm, we can utilize it to make predictions based on the learned patterns.

# In[19]:


print(X_test)  # Displaying the testing data - Hours studied

y_pred = regressor.predict(X_test)  # Making predictions of the scores


# In[29]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  # Creating a DataFrame to compare actual and predicted scores

df  # Displaying the DataFrame with actual and predicted scores


# ## Chechking for mean absolute error of the predictions

# In[28]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,r2_score,mean_absolute_error
print(r2_score(y_test,y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred)) 


# # Task  
# 
# ## Question: If a student studies for 9.25 hours per day, what would be the predicted score?

# In[31]:


hours = 9.25
hours_2d = np.array(hours).reshape(1, -1)  # Reshape the input data to a 2D array

own_pred = regressor.predict(hours_2d)  # Make the prediction using the reshaped input
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# # TASK COMPLETED !

# In[ ]:




