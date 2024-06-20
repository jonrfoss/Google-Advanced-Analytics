#!/usr/bin/env python
# coding: utf-8

# # **TikTok Project**
# **Course 6 - The Nuts and bolts of machine learning**

# Recall that you are a data professional at TikTok. Your supervisor was impressed with the work you have done and has requested that you build a machine learning model that can be used to determine whether a video contains a claim or whether it offers an opinion. With a successful prediction model, TikTok can reduce the backlog of user reports and prioritize them more efficiently.
# 
# A notebook was structured and prepared to help you in this project. A notebook was structured and prepared to help you in this project. Please complete the following questions.

# # **Course 6 End-of-course project: Classifying videos using machine learning**
# 
# In this activity, you will practice using machine learning techniques to predict on a binary outcome variable.
# <br/>
# 
# **The purpose** of this model is to increase response time and system efficiency by automating the initial stages of the claims process.
# 
# **The goal** of this model is to predict whether a TikTok video presents a "claim" or presents an "opinion".
# <br/>
# 
# *This activity has three parts:*
# 
# **Part 1:** Ethical considerations
# * Consider the ethical implications of the request
# 
# * Should the objective of the model be adjusted?
# 
# **Part 2:** Feature engineering
# 
# * Perform feature selection, extraction, and transformation to prepare the data for modeling
# 
# **Part 3:** Modeling
# 
# * Build the models, evaluate them, and advise on next steps
# 
# Follow the instructions and answer the questions below to complete the activity. Then, you will complete an Executive Summary using the questions listed on the PACE Strategy Document.
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.
# 
# 

# # **Classify videos using machine learning**

# <img src="images/Pace.png" width="100" height="100" align=left>
# 
# # **PACE stages**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## **PACE: Plan**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.
# 
# In this stage, consider the following questions:
# 
# 
# 1.   **What are you being asked to do? What metric should I use to evaluate success of my business/organizational objective?**
# 
# 2.   **What are the ethical implications of the model? What are the consequences of your model making errors?**
#   *   What is the likely effect of the model when it predicts a false negative (i.e., when the model says a video does not contain a claim and it actually does)?
# 
#   *   What is the likely effect of the model when it predicts a false positive (i.e., when the model says a video does contain a claim and it actually does not)?
# 
# 3.   **How would you proceed?**
# 

# ==> ENTER YOUR RESPONSES HERE

# ### **Task 1. Imports and data loading**
# 
# Start by importing packages needed to build machine learning models to achieve the goal of this project.

# In[2]:


# Import packages for data manipulation
### YOUR CODE HERE ###
import pandas as pd
import numpy as np

# Import packages for data visualization
### YOUR CODE HERE ###
import matplotlib.pyplot as plt
import seaborn as sns

# Import packages for data preprocessing
### YOUR CODE HERE ###
from sklearn.feature_extraction.text import CountVectorizer

# Import packages for data modeling
### YOUR CODE HERE ###
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score, \
recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance


# Now load the data from the provided csv file into a dataframe.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[4]:


# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")


# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## **PACE: Analyze**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

# ### **Task 2: Examine data, summary info, and descriptive stats**

# Inspect the first five rows of the dataframe.

# In[5]:


# Display first few rows
### YOUR CODE HERE ###
data.head(10)


# Get the number of rows and columns in the dataset.

# In[6]:


# Get number of rows and columns
### YOUR CODE HERE ###
data.shape


# Get the data types of the columns.

# In[7]:


# Get data types of columns
### YOUR CODE HERE ###
data.dtypes


# Get basic information about the dataset.

# In[8]:


# Get basic information
### YOUR CODE HERE ###
data.info()


# Generate basic descriptive statistics about the dataset.

# In[9]:


# Generate basic descriptive stats
### YOUR CODE HERE ###
data.describe()


# Check for and handle missing values.

# In[10]:


# Check for missing values
### YOUR CODE HERE ###
data.isna().sum()


# In[11]:


# Drop rows with missing values
### YOUR CODE HERE ###
data = data.dropna(axis=0)


# In[12]:


# Display first few rows after handling missing values
### YOUR CODE HERE ###
data.head(10)


# Check for and handle duplicates.

# In[13]:


# Check for duplicates
### YOUR CODE HERE ###
data.duplicated().sum()


# Check for and handle outliers.

# In[14]:


fig, axes = plt.subplots(2,3, figsize=(25,6))
sns.boxplot(data=data, x="video_view_count", orient="h", ax=axes[0, 0])

sns.boxplot(data=data, x="video_like_count", orient="h", ax=axes[0, 1])

sns.boxplot(data=data, x="video_share_count", orient="h", ax=axes[0, 2])

sns.boxplot(data=data, x="video_download_count", orient="h", ax=axes[1, 0])

sns.boxplot(data=data, x="video_comment_count", orient="h", ax=axes[1, 1])

axes[0, 1].set_title("Boxplot to check for outlier");


# In[17]:


### YOUR CODE HERE ###
print('Tree-based models are robust to outliers, so there is no need to impute or drop any values based on where they fall in their distribution.')


# Check class balance.

# In[18]:


# Check class balance
### YOUR CODE HERE ###
data['claim_status'].value_counts(normalize=True)


# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## **PACE: Construct**
# Consider the questions in your PACE Strategy Document to reflect on the Construct stage.

# ### **Task 3: Feature engineering**

# Extract the length of each `video_transcription_text` and add this as a column to the dataframe, so that it can be used as a potential feature in the model.

# In[19]:


# Extract the length of each `video_transcription_text` and add this as a column to the dataframe
### YOUR CODE HERE ###
data["text_length"] = data["video_transcription_text"].apply(func=lambda text: len(text))
data.head()


# Calculate the average text_length for claims and opinions.

# In[20]:


# Calculate the average text_length for claims and opinions
### YOUR CODE HERE ###
data[["claim_status", "video_transcription_text"]].groupby(by="claim_status")[["video_transcription_text"]].agg(func=lambda array: np.mean([len(text) for text in array]))


# Visualize the distribution of `text_length` for claims and opinions.

# In[22]:


# Visualize the distribution of `text_length` for claims and opinions
# Create two histograms in one plot
### YOUR CODE HERE ###
sns.histplot(data=data, stat="count", multiple="layer", x="text_length",
             kde=False, palette="pastel", hue="claim_status",
             element="bars", legend=True)
plt.xlabel("video_transcription_text length (number of characters)")
plt.ylabel("Count")
plt.title("Distribution of video_transcription_text length for claims and opinions")
plt.show()


# **Feature selection and transformation**

# Encode target and catgorical variables.

# In[17]:


# Create a copy of the X data
### YOUR CODE HERE ###
X=data.copy()

# Drop unnecessary columns
### YOUR CODE HERE ###
X = X.drop(['#', 'video_id', 'video_transcription_text'], axis=1)

# Encode target variable
### YOUR CODE HERE ###
X['claim_status'] = X['claim_status'].replace({'opinion': 0, 'claim': 1})

# Dummy encode remaining categorical values
### YOUR CODE HERE ###
X = pd.get_dummies(X,
                   columns=['verified_status', 'author_ban_status'],
                   drop_first=True)
X.head()


# ### **Task 4: Split the data**

# Assign target variable.

# In[18]:


# Isolate target variable
### YOUR CODE HERE ###
 qy = X['claim_status']


# Isolate the features.

# In[19]:


# Isolate features
### YOUR CODE HERE ###
X = X.drop(['claim_status'], axis=1)

# Display first few rows of features dataframe
### YOUR CODE HERE ###
X.head(10)


# #### **Task 5: Create train/validate/test sets**

# Split data into training and testing sets, 80/20.

# In[20]:


# Split the data into training and testing sets
### YOUR CODE HERE ###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# Split the training set into training and validation sets, 75/25, to result in a final ratio of 60/20/20 for train/validate/test sets.

# In[21]:


# Split the training data into training and validation sets
### YOUR CODE HERE ###
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 0)


# Confirm that the dimensions of the training, validation, and testing sets are in alignment.

# In[46]:


# Get shape of each training, validation, and testing set
### YOUR CODE HERE ###
X_tr.shape, X_val.shape, X_test.shape, y_tr.shape, y_val.shape, y_test.shape


# ### **Task 6. Build models**
# 

# ### **Build a random forest model**

# Fit a random forest model to the training set. Use cross-validation to tune the hyperparameters and select the model that performs best on recall.

# In[23]:


split_index = [0 if x in X_val.index else -1 for x in X_train.index]
custom_split = PredefinedSplit(split_index)


# In[24]:


# Instantiate the random forest classifier
### YOUR CODE HERE ###
rf = RandomForestClassifier(random_state=0)

# Create a dictionary of hyperparameters to tune
### YOUR CODE HERE ###
cv_params = {"max_depth" : [2,3,4,5, None],
                   "min_samples_leaf" : [1,2,3],
                   "min_samples_split" : [2,3,4],
                   "max_features" : [2,3,4],
                   "n_estimators" : [75, 100, 125, 150]
                  }

# Define a dictionary of scoring metrics to capture
### YOUR CODE HERE ###
scoring = {'accuracy', 'precision', 'recall', 'f1'}

# Instantiate the GridSearchCV object
### YOUR CODE HERE ###
rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv = custom_split, refit='recall')


# In[25]:


get_ipython().run_cell_magic('time', '', 'rf_cv.fit(X_train, y_train)\n')


# In[26]:


# Examine best recall score
### YOUR CODE HERE ###
rf_cv.best_score_


# In[50]:


# Get all the results from the CV and put them in a df
### YOUR CODE HERE ###

# Isolate the row of the df with the max(mean precision score)
### YOUR CODE HERE ###
def make_results(model_name, model_object):
  '''
  Accepts as arguments a model name (your choice - string) and
  a fit GridSearchCV model object.

  Returns a pandas df with the F1, recall, precision, and accuracy scores
  for the model with the best mean F1 score across all validation folds.  
  '''

  # Get all the results from the CV and put them in a df
  cv_results = pd.DataFrame(model_object.cv_results_)

  # Isolate the row of the df with the max(mean f1 score)
  best_estimator_results = cv_results.iloc[cv_results['mean_test_precision'].idxmax(), :]

  # Extract accuracy, precision, recall, and f1 score from that row
  f1 = best_estimator_results.mean_test_f1
  recall = best_estimator_results.mean_test_recall
  precision = best_estimator_results.mean_test_precision
  accuracy = best_estimator_results.mean_test_accuracy

  # Create table of results
  table = pd.DataFrame({'Model': [model_name],
                        'F1': [f1],
                        'Recall': [recall],
                        'Precision': [precision],
                        'Accuracy': [accuracy]
                       }
                      )

  return table

rf_cv_results = make_results('Random Forest CV', rf_cv)
rf_cv_results


# In[28]:


# Examine best parameters
### YOUR CODE HERE ###
rf_cv.best_estimator_


# In[34]:


rf_cv.best_params_


# **Question:** How well is your model performing? Consider average recall score and precision score.

# ### **Build an XGBoost model**

# In[29]:


# Instantiate the XGBoost classifier
### YOUR CODE HERE ###
xgb = XGBClassifier(objective='binary:logistic', random_state=0)

# Create a dictionary of hyperparameters to tune
### YOUR CODE HERE ###
cv_params = {'max_depth': [4, 6],
              'min_child_weight': [3, 5],
              'learning_rate': [0.1, 0.2, 0.3],
              'n_estimators': [5,10,15],
              'subsample': [0.7],
              'colsample_bytree': [0.7]
              }

# Define a dictionary of scoring metrics to capture
### YOUR CODE HERE ###
scoring = {'accuracy', 'precision', 'recall', 'f1'}

# Instantiate the GridSearchCV object
### YOUR CODE HERE ###
xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=5, refit='recall')


# In[30]:


get_ipython().run_cell_magic('time', '', 'xgb_cv.fit(X_train, y_train)\n')


# In[58]:


# Get all the results from the CV and put them in a df
### YOUR CODE HERE ###

# Isolate the row of the df with the max(mean precision score)
### YOUR CODE HERE ###
def make_results(model_name, model_object):
  '''
  Accepts as arguments a model name (your choice - string) and
  a fit GridSearchCV model object.

  Returns a pandas df with the F1, recall, precision, and accuracy scores
  for the model with the best mean F1 score across all validation folds.  
  '''

  # Get all the results from the CV and put them in a df
  cv_results = pd.DataFrame(model_object.cv_results_)

  # Isolate the row of the df with the max(mean f1 score)
  best_estimator_results = cv_results.iloc[cv_results['mean_test_precision'].idxmax(), :]

  # Extract accuracy, precision, recall, and f1 score from that row
  f1 = best_estimator_results.mean_test_f1
  recall = best_estimator_results.mean_test_recall
  precision = best_estimator_results.mean_test_precision
  accuracy = best_estimator_results.mean_test_accuracy

  # Create table of results
  table = pd.DataFrame({'Model': [model_name],
                        'F1': [f1],
                        'Recall': [recall],
                        'Precision': [precision],
                        'Accuracy': [accuracy]
                       }
                      )

  return table

xgb_cv_results = make_results('XGBoost CV', xgb_cv)
xgb_cv_results


# **Question:** How well does your model perform? Consider recall score and precision score.

# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## **PACE: Execute**
# Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

# ### **Task 7. Evaluate model**
# 
# Evaluate models against validation criteria.

# #### **Random forest**

# In[35]:


# Use the random forest "best estimator" model to get predictions on the encoded testing set
### YOUR CODE HERE ###
rf_y_pred = rf_cv.best_estimator_.predict(X_test)


# Display the predictions on the encoded testing set.

# In[36]:


# Display the predictions on the encoded testing set
### YOUR CODE HERE ###
rf_y_pred


# Display the true labels of the testing set.

# In[37]:


# Display the true labels of the testing set
### YOUR CODE HERE ###
y_test


# Create a confusion matrix to visualize the results of the classification model.

# In[38]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Create a confusion matrix to visualize the results of the classification model

# Compute values for confusion matrix
### YOUR CODE HERE ###
cm = confusion_matrix(y_test, rf_y_pred)

# Create display of confusion matrix
### YOUR CODE HERE ###
disp = ConfusionMatrixDisplay(cm, display_labels = rf_cv.classes_)

# Plot confusion matrix
### YOUR CODE HERE ###


# Display plot
### YOUR CODE HERE ###
disp.plot()


# Create a classification report that includes precision, recall, f1-score, and accuracy metrics to evaluate the performance of the model.

# In[39]:


# Create a classification report
# Create classification report for random forest model
### YOUR CODE HERE ###
print(classification_report(y_test, rf_y_pred))


# **Question:** What does your classification report show? What does the confusion matrix indicate?

# #### **XGBoost**

# In[41]:


#Evaluate XGBoost model
### YOUR CODE HERE ###
xgb_cv.best_estimator_


# In[43]:


xgb_cv.best_params_


# In[44]:


# Compute values for confusion matrix
### YOUR CODE HERE ###
xgb_y_pred = xgb_cv.best_estimator_.predict(X_test)

# Create display of confusion matrix
### YOUR CODE HERE ###
cm = confusion_matrix(y_test, xgb_y_pred)

# Plot confusion matrix
### YOUR CODE HERE ###
disp = ConfusionMatrixDisplay(cm, display_labels = xgb_cv.classes_)

# Display plot
### YOUR CODE HERE ###
disp.plot()


# In[54]:


# Create a classification report
### YOUR CODE HERE ###
print(classification_report(y_test, xgb_y_pred))


# **Question:** Describe your XGBoost model results. How does your XGBoost model compare to your random forest model?

# ### **Use champion model to predict on test data**

# In[55]:


### YOUR CODE HERE ###


# In[56]:


# Compute values for confusion matrix
### YOUR CODE HERE ###
rf_y_pred = rf_cv.best_estimator_.predict(X_test)

cm = confusion_matrix(y_test, rf_y_pred)

# Create display of confusion matrix
### YOUR CODE HERE ###
disp = ConfusionMatrixDisplay(cm, display_labels = rf_cv.classes_)
# Plot confusion matrix
### YOUR CODE HERE ###

# Display plot
### YOUR CODE HERE ###
disp.plot()


# #### **Feature importances of champion model**
# 

# In[57]:


### YOUR CODE HERE ###
from xgboost import plot_importance
plot_importance(xgb_cv.best_estimator_)


# **Question:** Describe your most predictive features. Were your results surprising?

# ### **Task 8. Conclusion**
# 
# In this step use the results of the models above to formulate a conclusion. Consider the following questions:
# 
# 1. **Would you recommend using this model? Why or why not?**
# 
# 2. **What was your model doing? Can you explain how it was making predictions?**
# 
# 3. **Are there new features that you can engineer that might improve model performance?**
# 
# 4. **What features would you want to have that would likely improve the performance of your model?**
# 
# Remember, sometimes your data simply will not be predictive of your chosen target. This is common. Machine learning is a powerful tool, but it is not magic. If your data does not contain predictive signal, even the most complex algorithm will not be able to deliver consistent and accurate predictions. Do not be afraid to draw this conclusion.
# 

# 1. **Would you recommend using this model? Why or why not?** Yes, one can recommend this model because it performed well on both the validation and test holdout data. Furthermore, both precision and F1 scores were consistently high. The model very successfully classified claims and opinions.
# <br>
# 
# 2. **What was your model doing? Can you explain how it was making predictions?** The model's most predictive features were all related to the user engagement levels associated with each video. It was classifying videos based on how many views, likes, shares, and downloads they received.
# <br>
# 
# 3. **Are there new features that you can engineer that might improve model performance?** Because the model currently performs nearly perfectly, there is no need to engineer any new features.
# <br>
# 
# 4. **What features would you want to have that would likely improve the performance of your model?** The current version of the model does not need any new features. However, it would be helpful to have the number of times the video was reported. It would also be useful to have the total number of user reports for all videos posted by each author.

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
