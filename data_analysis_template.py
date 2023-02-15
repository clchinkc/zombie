
"""
In an analytical competition, you can perform a wide range of data analysis tasks, depending on the nature of the competition and the data provided. Here are some common types of data analysis you might perform:

Problem Statement: Clearly state the problem you are trying to solve and the question you are trying to answer.

Data Collection: Describe how you collected the data, including the sources and methods used.

Data Cleaning: Detail the steps you took to clean and preprocess the data, including handling missing values, outliers, and other anomalies.

Exploratory Data Analysis (EDA): This involves understanding the basic characteristics of the data, such as its distribution, missing values, outliers, etc.

Data Visualization: This involves creating visual representations of the data to gain insights and help communicate your findings.

Hypothesis Testing: This involves using statistical methods to test if a certain relationship exists between variables in the data.

Predictive modeling - this involves building a model that predicts an outcome based on the data. For example, you could build a model that predicts customer churn, sales, or stock prices.

Time Series Analysis: This involves analyzing data collected over time to understand trends, seasonality, and patterns in the data.

Cluster analysis - if the data includes a large number of variables, you could use cluster analysis to group the data into similar groups.

Sentiment analysis - if the data includes text data, you could perform sentiment analysis to classify the sentiment of the text.

Customer segmentation - you could segment the data based on customer behavior, preferences, or demographics to better understand customer groups and target marketing efforts.

Machine learning - you could use machine learning algorithms to analyze the data and make predictions or identify patterns.

Model Interpretation: This involves understanding the factors that drive the model's predictions and explaining the results to others.
"""


import matplotlib.pyplot as plt
import numpy as np

# Import libraries
import pandas as pd
import seaborn as sns

# Load the data
df = pd.read_csv('data.csv')

# Create the first DataFrame
df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'],
                    'key1': ['A', 'B', 'C', 'D'],
                    'value': [1, 2, 3, 4]})
# Create the second DataFrame
df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'],
                    'key2': ['B', 'D', 'E', 'F'],
                    'value': [5, 6, 7, 8]})
# Merge the two DataFrames on the 'key' column
merged_df = df1.merge(df2, on='key') # returns key column
merged_df = df1.merge(df2, on=['key1','key2']) # returns key1 and key2 columns
"""
how: Specifies the type of join to perform. Can be 'inner', 'outer', 'left', or 'right'.
on: Specifies the column to join on. If not specified, the intersection of the columns is used.
"""

# Exploratory Data Analysis (EDA)
df.info()
df.describe()

# Data cleaning
# Drop rows with missing values, inplace means the changes are saved to the original DataFrame
df.dropna(inplace=True, axis=0) # axis=0 means rows, axis=1 means columns
# Drop specific rows and columns by index or label
df.drop([0,1], axis=0, inplace=True)
df.drop(['Count', 'Rank'], axis=1)
# fill missing values with 0
df.fillna(0)
"""
value: Specifies the value to use to fill in the missing values. Can be a scalar value or a dict of values for different columns.
method: Specifies the method to use for filling in missing values. Can be 'ffill' (forward-fill) or 'bfill' (backward-fill) or 'interpolate'(interpolate values) or 'pad' or 'backfill'
axis: Specifies the axis along which to fill in missing values. It can be 0 (rows) or 1 (columns).
inplace: Whether to fill in the missing values in place (modifying the original DataFrame) or to return a new DataFrame with the missing values filled in.
limit: Specifies the maximum number of consecutive missing values to fill.
downcast: Specifies a dictionary of values to use to downcast the data types of columns.
"""
# fill missing values using interpolation with a limit and downcasting
df.interpolate(limit=1, downcast='float')
# Drop duplicate rows
df.drop_duplicates(inplace=True)
# Drop columns
df.drop(['column1', 'column2'], axis=1, inplace=True)
# Rename columns
df.rename(columns={'old_name1': 'new_name1', 'old_name2': 'new_name2'}, inplace=True)
# You can also rename the index similarly
df.rename(index={0:'first',1:'second',2:'third'}, inplace=True)
# Change data types
df['column'] = df['column'].astype('int')
# Reset index, drop means the old index is dropped
df.reset_index(drop=True, inplace=True)
# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])
# Convert datetime column to date
df['date'] = df['date'].dt.date
# Select specific rows using .iloc
print(df.iloc[3:6]) # row index, column index
# Select specific rows using .loc
print(df.loc[df['column_name'] == value]) # row label, column label
# Add a new column to the DataFrame
df.assign(new_column=new_value) # new_column_name, new_column_value as a calculation from current DataFrame
# Return filtered DataFrame based on boolean conditions
df_queried = df.query('column_name > value')
# Sort values by a column
df_sorted = df.sort_values(by=['Count', 'Rank'], ascending=False, inplace=True)
# Take a random sample from the DataFrame
df.sample(frac=0.1)
# Sample 2 rows without replacement with specific column to be chosen
df_sample = df.sample(n=2, replace=False, random_state=1, axis=1)

# Data Visualization
# Histogram
df['column'].hist(bins=20)
# Scatter plot
df.plot.scatter(x='column1', y='column2')
df.plot(x='column_name1', y='column_name2', kind='scatter')
"""
kind: specifies the type of plot you want to create. Available options are line, bar, barh, hist, box, kde, density, area, pie, scatter, and hexbin.
title: is used to set the title of the plot.
xlabel: is used to set the label of the x-axis.
ylabel: is used to set the label of the y-axis.
legend: is used to specify whether to show the legend or not.
grid: is used to specify whether to show the grid in the plot or not.
xlim: is used to set the limits of the x-axis.
ylim: is used to set the limits of the y-axis.
xticks: is used to set the ticks of the x-axis.
yticks: is used to set the ticks of the y-axis.
"""
# Bar plot
df['column'].value_counts().plot.bar()
# Box plot
df.boxplot(column='column')
# Line plot
df.plot.line(x='column1', y='column2')
# Heatmap
sns.heatmap(df.corr(), annot=True)
# Pairplot
sns.pairplot(df, x_vars=['feature1', 'feature2', 'feature3'], y_vars='target', height=7, aspect=0.7)
# Countplot
sns.countplot(x='target', data=df)
# Violin plot
sns.violinplot(x='target', y='feature1', data=df)
# FacetGrid
g = sns.FacetGrid(df, col='target')
g.map(plt.hist, 'feature1', bins=20)
# JointGrid
g = sns.JointGrid(x='feature1', y='feature2', data=df)
g.plot(sns.regplot, sns.distplot)
# Boxen plot
sns.boxenplot(x='target', y='feature1', data=df)
# Swarm plot
sns.swarmplot(x='target', y='feature1', data=df)
# Catplot
sns.catplot(x='target', y='feature1', data=df, kind='boxen')
# Box plot
sns.boxplot(x='target', y='feature1', data=df)
# Bar plot
sns.barplot(x='target', y='feature1', data=df)
# Scatter plot
sns.scatterplot(x='feature1', y='feature2', data=df)
# Line plot
sns.lineplot(x='feature1', y='feature2', data=df)
# Histogram
sns.distplot(df['feature1'], bins=20, kde=False)
# KDE plot
sns.kdeplot(df['feature1'], shade=True)
# Create a pivot table
pivot_table = pd.pivot_table(df, index=['column_name1','column_name2'], values= 'Count' , aggfunc=['sum','count'])
pivot_table.head(20)
# Group rows by a column and aggregate using a function, e.g. calculate mean, sum, count, etc, for each group
grouped = df.groupby('column_name').agg({'column_name': 'mean'})

# Hypothesis Testing
# One-sample t-test
from scipy import stats

stats.ttest_ind(df[df['target'] == 0]['feature1'], df[df['target'] == 1]['feature1'])

# Predictive modeling
# Linear Regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
X = df[['feature1', 'feature2', 'feature3']]
y = df['target']
model.fit(X, y)

# Time Series Analysis
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df['column_name'], model='additive', freq=1)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.subplot(411)
plt.plot(df['column_name'], label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

# Cluster analysis
# K-means clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
df['segment'] = kmeans.predict(X)

# Sentiment analysis
# TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer

sentiment = SentimentIntensityAnalyzer()
df['sentiment'] = df['text'].apply(lambda x: sentiment.polarity_scores(x)['compound'])

# Machine learning
# ...

# Insight generation
# ...

# Export the processed DataFrame to a CSV file
df.to_csv('path/to/new_data.csv')
"""
index: specifies whether to include the index of the DataFrame in the CSV file or not.
header: specifies whether to include the column names of the DataFrame in the CSV file or not.
columns: is used to specify the columns to be included in the CSV file.
sep: is used to specify the separator to be used in the CSV file.
na_rep: is used to specify the string to be used for missing values in the CSV file.
"""
