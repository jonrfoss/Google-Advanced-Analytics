# Google-Advanced-Analytics

# Introduction

## Capstone Background
The HR department at Salifort Motors is dedicated to enhancing employee satisfaction and retention within the company. They are particularly focused on understanding the factors that are likely to cause an employee to leave. By accurately predicting employee attrition, the company can identify and address the underlying issues contributing to turnover. Improving employee retention will not only reduce the costs associated with recruiting, interviewing, and training new employees but also foster a more stable and motivated workforce. Ultimately, this proactive approach will significantly benefit Salifort Motors.

## Explore
* Different factors affecting turnover
* Best model for predicting whether an employee will leave

## Executive Summary
The research attempts to identify the factors for an employee leaving. To make this determination, the following methodologies where used:
* **Explore** the data
* **Wrangle** data to create left / stayed outcome variable
* **Explore** data with data visualization techniques
* **Analyze** the data
* **Build Models** to predict leaving outcomes using logistic regression, decision tree, random forest and xg boost models

## Exploratory Data Analysis
* People with a higher number of projects, above 6 projects, seem to leave at a higher frequency.
* People who are on the low and high end for average monthly hours worked with low satisfaction levels seem to leave more.
* People tend to leave earlier in their tenure.
* Low performers and high performers, based on the last evaluation, tend to leave more.
* People who are not promoted in the last 5 years who also work a higher average number of hours each month tend to leave.
* Sales, technical and support were the three departments that had the most departures.

# Conclusion
* **Model Performance:** All models performed similarly with the xg boost model slightly outperforming the rest.
* **Top Features:** The top four features identified for predicting employees leaving are: last evaluation, number of projects, tenure, overworked
