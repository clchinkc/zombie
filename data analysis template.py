
# EDA (Exploratory Data Analysis) - Data Analysis Template

# US Police Shootings Dataset

# https://colab.research.google.com/drive/10pcm9FA6uuboKCPNjlk4HHKX0_MnpfTZ?usp=sharing
# write me python code for basic data exploration and visualization of the dataset
# date column in my dataset is in object datatype, convert it to datetime format column
# what are the count of deaths by age, show in visualization
# can we use seaborn library with a regression line to plot count of deaths by age as a visualization


# Import libraries
from unicodedata import numeric

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Load the dataset
loc_file = 'https://raw.githubusercontent.com/MuhammadAli437/ChatGPT/main/shootings.csv'
data = pd.read_csv(loc_file)

"""
data:
id, name, date, manner_of_death, armed, age, gender, race, city, state, signs_of_mental_illness, threat_level, flee, body_camera, arms_category
3, Tim Elliot, 2015-01-02, shot, gun, 53.0, M, Asian, Shelton, WA, True, attack, Not fleeing, False, Guns
"""

"""
# Get summary statistics of the data
print("Summary statistics of the data:")
print(data.describe())
# Get the data types of the columns
print("Data types of the columns:")
print(data.info())
# Get the first 5 rows of the data
print("First 5 rows of the data:")
print(data.head())
# Get the null rows in each column
print("Null rows in each column:")
print(data.isnull().sum())
# Get the number of na rows in each column
print("Number of na rows in each column:")
print(data.isna().sum())
# Get the number of duplicated rows
print("Number of duplicated rows:")
print(data.duplicated().sum())
# Get the number of unique values in each column
print("Number of unique values in each column:")
print(data.nunique())
# Check for outliers
print("Outliers:")
print(data.select_dtypes(include='number').apply(lambda x: np.abs(stats.zscore(x)) > 5).sum())
# Check for correlations
print("Correlations:")
print(data.corr(numeric_only=True))
# Check for all class imbalance
print("Class imbalance:")
data_grouped = [data.groupby(classes).size() for classes in data.columns]
# plot the class imbalance in subplot
plt.figure(figsize=(20, 20))
for i in range(len(data_grouped)):
    plt.subplot(math.ceil(len(data_grouped)/4), 4, i+1)
    plt.bar(data_grouped[i].index, data_grouped[i].values)
    plt.title(str(data.columns[i]))
plt.tight_layout()
plt.show()
# TODO: Unclean plot, fix it
"""


# Data cleaning

# Drop rows with na values
print("Drop rows with missing values:")
data = data.dropna(axis=0)
print("Number of rows after dropping rows with missing values:", data.shape[0])
# Drop columns that are not useful for the analysis
print("Drop columns that are not useful for the analysis:")
data = data.drop(['name'], axis=1)
print("Number of columns after dropping columns that are not useful for the analysis:", data.shape[1])
# Drop duplicate rows
print("Drop duplicate rows:")
data = data.drop_duplicates()
print("Number of rows after dropping duplicate rows:", data.shape[0])
# Fill missing value in numerical column with the mean value
print("Fill missing value in numerical column with the mean value:")
numeric_cols = data.select_dtypes(include='number').columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
print("Number of rows after filling missing value in numerical column with the mean value:", data.shape[0])
# Fill missing value in categorical column with the mode value
print("Fill missing value in categorical column with the mode value:")
categorical_cols = data.select_dtypes(include='object').columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
print("Number of rows after filling missing value in categorical column with the mode value:", data.shape[0])
# Drop rows with outliers using z-score
#data[numeric_cols] = data[numeric_cols and (np.abs(stats.zscore(data)) < 5).all(axis=1)]
print("Drop rows with outliers using z-score:")
data.loc[(np.abs(stats.zscore(data[numeric_cols])) < 5).all(axis=1), numeric_cols] = data[numeric_cols]
print("Number of rows after dropping rows with outliers using z-score:", data.shape[0])

# Data wrangling

# Turn the date column into a datetime object
data["date"] = pd.to_datetime(data["date"])
# Extract the year from the date column
data['year'] = data['date'].dt.year
# Extract the month from the date column
data['month'] = data['date'].dt.month
# Extract the day from the date column
data['day'] = data['date'].dt.day
# Change the data type of the age column to integer
data['age'] = data['age'].astype(int)
# Rename columns
data = data.rename(columns={'manner_of_death': 'death_manner', 'signs_of_mental_illness': 'mental_illness', 'threat_level': 'threat', 'body_camera': 'bodycam', 'arms_category': 'weapon_category'})
# Change the data type of the mental_illness column to boolean
data['mental_illness'] = data['mental_illness'].astype(bool)
# Change the data type of the bodycam column to boolean
data['bodycam'] = data['bodycam'].astype(bool)
# Create a new column called age_group that categorizes the age of the individuals shot
data = data.assign(age_group=data['age'].apply(lambda x: 'child' if x < 18 else 'adult' if x >= 18 and x < 65 else 'elderly'))
data['age_group'] = pd.cut(data['age'], bins=[0, 17, 64, 100], labels=['child', 'adult', 'elderly'])
# Use query to filter the data based on whether the individual has a mental illness or not
mental_illness = True
no_mental_illness = False
data_mental_illness = data.query('mental_illness == @mental_illness').copy()
data_no_mental_illness = data.query('mental_illness == @no_mental_illness').copy()
# Use query to filter the data based on whether the individual was armed or not
data_armed = data.query('armed == True').copy()
data_not_armed = data.query('armed == False').copy()
# Group the data by year and calculate the total number of deaths for each year
deaths_by_year = data.groupby(['year', 'month']).sum(numeric_only=True)
# Sort values by year
deaths_by_year = deaths_by_year.sort_values(by='year', ascending=True)

"""
To optimize the data analysis, we can consider the following suggestions:

Instead of using data.describe() and data.info(), we can use pandas_profiling library to generate a detailed report on the data, which includes summary statistics, data types, correlation matrix, missing values, and data distributions.

Instead of dropping rows with missing values, we can impute missing values using more advanced methods such as K-nearest neighbors (KNN) imputation or multiple imputation. Dropping rows can result in loss of information and bias in the analysis.

Instead of dropping columns that are not useful for the analysis, we can use feature selection methods such as correlation analysis, principal component analysis (PCA), or recursive feature elimination (RFE) to identify the most important features for the analysis.

Instead of filling missing values with the mean value, we can use more appropriate methods such as median or mode imputation, or use machine learning models to predict missing values.

Instead of dropping rows with outliers, we can use more robust statistical methods such as Tukey's fences or Winsorization to deal with outliers. Alternatively, we can use machine learning models that are less sensitive to outliers.

Instead of using data.groupby() to group the data by year and calculate the total number of deaths for each year, we can use data.resample() function to resample the data by month, quarter, or year and calculate the aggregate statistics such as mean, sum, or count. This can provide more flexibility in analyzing the temporal patterns in the data.

Instead of using query() function to filter the data based on certain conditions, we can use Boolean indexing or SQL-like syntax to filter the data. This can provide more flexibility in complex filtering conditions.

Instead of using simple statistical methods such as mean, median, or correlation to analyze the data, we can use more advanced machine learning models such as decision trees, random forests, or neural networks to analyze the data. This can provide more accurate and reliable predictions and insights.
"""

"""
univariate analysis
bivariate analysis
variable transformation
dimensionality reduction
"""
"""
classification model
regression model
clustering model
time series model
"""



"""

# Scatterplot of the number of deaths by year
sns.scatterplot(x='year', y='id', data=deaths_by_year).set(title="Scatterplot of Number of Deaths by Year")
plt.show()


# Visualize the distribution of ages
sns.histplot(data=data, x='age', kde=True, bins=20).set(title="Histogram of Age of Individuals Shot")
plt.show()
# no need

# Visualize the number of deaths by race
sns.countplot(data=data, x='race').set(title="Count Chart of Race of Individuals Shot")
plt.show()
# no need

# Bar plot of the frequency of each race
sns.countplot(data=data, x='race').set(title="Count Chart of Race of Individuals Shot")
plt.show()
# use data['race'].value_counts() and barplot(x=race_counts.index, y=race_counts.values) if want the data to be sorted by the frequency

# Visualize the number of deaths by gender through years
data_gender = data.groupby(['year', 'gender'])['id'].count().unstack()
sns.barplot(x='year', y='total', hue='gender', data=data_gender.reset_index().melt(id_vars=['year'], var_name='gender', value_name='total'), palette='deep').set(title="Count plot of Gender of Individuals Shot")
plt.show()
# can be done on other columns through years as well

# Bar plot of the average age by race
sns.barplot(x='race', y='age', data=data)
plt.show()

# Stacked bar plot of the frequency of each race and gender
counts = data.groupby(['race', 'gender']).size().unstack()
counts.plot(kind='bar', stacked=True)
plt.title('Distribution of race and gender in shootings')
plt.show()

# Count plot of weapon category
sns.countplot(data=data, x='weapon_category').set(title="Count plot of Weapon Category of Individuals Shot")
plt.xticks(rotation=90, fontsize=8)
plt.show()

# Visualize the thread level of each mental illness
sns.countplot(data=data, x='mental_illness', hue='threat').set(title="Count plot of Mental Illness and Threat Level")
plt.show()
# no need

# Line chart of number of deaths by year and month
sns.lineplot(data=deaths_by_year, x='month', y='age', hue='year').set(title="Line chart of Number of Deaths by Year and Month")
plt.show()
"""



# data_race_gender = data.groupby(['year', 'race', 'gender'])['id'].count().reset_index(name='count')
# sns.lineplot(x='year', y='count', hue='race', style='gender', data=data_race_gender)
# plt.title('Number of deaths by race and gender over time')
# plt.yscale('log')
# plt.show()
# better log scale for y axis


# sns.boxplot(x='weapon_category', y='age', hue='race', data=data)
# plt.title('Distribution of age and weapon category by race')
# plt.show()
# bad visualization

# data_race_weapon = data.groupby(['race', 'weapon_category'])['id'].count().reset_index(name='count')
# pivot_data = data_race_weapon.pivot(index='race', columns='weapon_category', values='count')
# sns.heatmap(pivot_data, cmap='Blues', annot=True)
# plt.title('Number of deaths by race and weapon category')
# plt.show()
# number overflow

# data_race_threat = data.groupby(['race', 'threat'])['id'].count().reset_index(name='count')
# pivot_data = data_race_threat.pivot(index='race', columns='threat', values='count')
# pivot_data.plot(kind='bar', stacked=True)
# plt.title('Number of deaths by race and threat level')
# plt.show()
# no meaning

# sns.scatterplot(x='age', y='id', hue='race', data=data)
# plt.title('Age versus number of deaths by race')
# plt.show()
# smaller dots

# sns.jointplot(x='age', y=data['race'].value_counts(), data=data, kind='reg')
# plt.title('Correlation between age and number of shootings by race')
# plt.show()
# not working


# Group data by race and gender and count the number of deaths and create stacked bar chart
# data_race_gender = data.groupby(['race', 'gender'])['id'].count().reset_index(name='count')
# sns.barplot(x='race', y='count', hue='gender', data=data_race_gender, palette='muted')
# plt.title('Number of Deaths by Race and Gender')
# plt.show()
# normal

# Group data by year and race and count the number of deaths and create line chart
# data_race_year = data.groupby(['year', 'race'])['id'].count().reset_index(name='count')
# sns.lineplot(x='year', y='count', hue='race', data=data_race_year, palette='muted')
# plt.title('Number of Deaths Over Time by Race')
# plt.show()
# normal

# Create histogram
# sns.histplot(x='age', hue='race', data=data, kde=True, palette='muted')
# plt.title('Age Distribution by Race')
# plt.show()
# good

# Compute correlation matrix and create heatmap
# corr = data.corr(numeric_only=True)
# sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')
# plt.title('Correlation Matrix')
# plt.show()
# good

# Create scatterplot matrix
# sns.pairplot(data, hue='race', palette='muted')
# plt.title('Pairwise Scatterplot Matrix')
# plt.show()
# too crowded, should choose some features


# sns.countplot(data=data, x='gender', order=data['gender'].value_counts(ascending=False, normalize=True).index)

"""

# Data visualization

# Plot a line chart of the number of deaths over the years
# Group the data by year and calculate the total number of deaths for each year
deaths_by_year = data.groupby(['year'])['id'].count()
# Plot the number of deaths over the years and months
deaths_by_year.plot(kind='line')
plt.xlabel('Year')
plt.ylabel('Number of Deaths')
plt.title('Number of Deaths by Year')
plt.show()

# Plot a line chart of the number of deaths over the years and months
# Group the data by year and month and calculate the total number of deaths for each year and month
deaths_by_year_month = data.groupby(['year', 'month'])['id'].count()
# Plot the number of deaths over the years and months
deaths_by_year_month.plot(kind='line')
plt.xlabel('Year and Month')
plt.ylabel('Number of Deaths')
plt.title('Number of Deaths by Year and Month')
plt.show()

# Group the data by age and calculate the total number of deaths for each age
deaths_by_age = data.groupby(['age'])['id'].count()
# Plot the histogram of the number of deaths by age
plt.hist(deaths_by_age, bins=20)
plt.xlabel('Frequency')
plt.ylabel('Number of Deaths')
plt.title('Number of Deaths by Age')
plt.show()

import seaborn as sns

# Group the data by age and calculate the total number of deaths for each age
deaths_by_age = data.groupby(['age'])['id'].count()
# Create a scatter plot of the number of deaths by age
sns.scatterplot(x=deaths_by_age.index, y=deaths_by_age.values)
# Add a regression line to the scatter plot
sns.regplot(x=deaths_by_age.index, y=deaths_by_age.values)
# Add labels and a title to the plot
plt.xlabel('Age')
plt.ylabel('Number of Deaths')
plt.title('Number of Deaths by Age')
# Show the plot
plt.show()

# Group the data by year and race and calculate the total number of deaths for each year and race
deaths_by_race_year = data.groupby(['year', 'race'])['id'].count()
# Reshape the dataframe into a more readable format
deaths_by_race_year = deaths_by_race_year.unstack()
# Plot the bar chart of the number of deaths by race and year
deaths_by_race_year.plot(kind='bar', stacked=True)
plt.xlabel('Year')
plt.ylabel('Number of Deaths')
plt.title('Number of Deaths by Race and Year')
plt.legend(loc='upper right')
plt.show()

import plotly.express as px

# Group the data by state and calculate the total number of deaths for each state
deaths_by_state = data.groupby(['state'])['id'].count().reset_index(name='counts')

# Create a choropleth map of the statewise shooting data
fig = px.choropleth(deaths_by_state,
                    locations='state',  # DataFrame column with locations
                    locationmode = 'USA-states', # set of locations match entries in `locations`
                    color='counts',  # DataFrame column with color values
                    title='Statewise Shooting Data')
fig.show()

"""

# TODO: Relationship between variables
# Median polish
# Statistical model
# Data analysis

