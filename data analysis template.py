# EDA (Exploratory Data Analysis) - Data Analysis Template

# US Police Shootings Dataset

# https://colab.research.google.com/drive/10pcm9FA6uuboKCPNjlk4HHKX0_MnpfTZ?usp=sharing
# write me python code for basic data exploration and visualization of the dataset
# date column in my dataset is in object datatype, convert it to datetime format column
# what are the count of deaths by age, show in visualization
# can we use seaborn library with a regression line to plot count of deaths by age as a visualization


# Import libraries
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Load the dataset
loc_file = "https://raw.githubusercontent.com/MuhammadAli437/ChatGPT/main/shootings.csv"
data = pd.read_csv(loc_file)

"""
data:
id, name, date, manner_of_death, armed, age, gender, race, city, state, signs_of_mental_illness, threat_level, flee, body_camera, arms_category
3, Tim Elliot, 2015-01-02, shot, gun, 53.0, M, Asian, Shelton, WA, True, attack, Not fleeing, False, Guns
"""

"""
# Merge the two DataFrames on the 'key' column
merged_df = df1.merge(df2, on='key') # returns key column
merged_df = df1.merge(df2, on=['key1','key2']) # returns key1 and key2 columns
merged_df = merged_df.loc[:, ~df_double.columns.duplicated()].copy() # remove duplicated columns
# how: Specifies the type of join to perform. Can be 'inner', 'outer', 'left', or 'right'.
# on: Specifies the column to join on. If not specified, the intersection of the columns is used.
# surffixes: Specifies the suffixes to use for overlapping column names. Defaults to ('_x', '_y').
"""

"""
# Get summary statistics of the data
print("Summary statistics of the data:")
print(data.describe())
# Get the data types of the columns
print("Data types of the columns:")
print(data.info())
# Get number of rows and columns
print("Number of rows and columns:")
print(data.shape[0], "rows and", data.shape[1], "columns")
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
# Create a pivot table that shows the mean, median, standard deviation, minimum, maximum, sum of age of individuals shot, grouped by race and gender
print("Pivot table that shows statistics of age grouped by age and gender:")
pivot_table = pd.pivot_table(data, values='age', index=['race', 'gender'], aggfunc=[np.mean, np.median, np.std, np.min, np.max, np.sum])
print(pivot_table)
data_grouped = [data.groupby(classes).size() for classes in data.columns]
# plot the class imbalance in subplot
plt.figure(figsize=(16, 16))
for i in range(len(data_grouped)):
    plt.subplot(math.ceil(len(data_grouped)/4), 4, i+1)
    plt.bar(data_grouped[i].index, data_grouped[i].values)
    plt.title(str(data.columns[i]))
# hide all the x labels and y labels
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
# show the plot
plt.tight_layout()
plt.show()
"""


# Data cleaning
# Reset index, drop means the old index is dropped
# data = data.reset_index(drop=True)
# show me the code for adding the index column
# data = data.set_index("id")
# Select specific rows using .iloc
# print(data.iloc[0:5, 0:5]) # rows index, columns index
# Select specific rows using .loc
# print(data.loc[data['age'] == 40]) # row label, column label
# Getting a slice of strings from a column
# data['name'] = data['name'].str.slice(0, 5) # first 5 characters
# data['name'] = data['name'].str[-5:] # last 5 characters
# Sample 5 rows without replacement with specific column to be chosen
# df_sample = data.sample(n=5, replace=False, random_state=1, axis=1).copy() # axis=0 for rows, axis=1 for columns, frac=0.1 for fraction of the whole dataset
# Drop rows with na values
print("Drop rows with missing values:")
data = data.dropna(axis=0)
print("Number of rows after dropping rows with missing values:", data.shape[0])
# Drop columns that are not useful for the analysis
print("Drop columns that are not useful for the analysis:")
data = data.drop(["name"], axis=1)
print("Number of columns after dropping columns that are not useful for the analysis:", data.shape[1])
# Drop duplicate rows
print("Drop duplicate rows:")
data = data.drop_duplicates()
print("Number of rows after dropping duplicate rows:", data.shape[0])
# Fill missing value in numerical column with the mean value
print("Fill missing value in numerical column with the mean value:")
numeric_cols = data.select_dtypes(include="number").columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
print("Number of rows after filling missing value in numerical column with the mean value:", data.shape[0])
# Fill missing value in categorical column with the mode value
print("Fill missing value in categorical column with the mode value:")
categorical_cols = data.select_dtypes(include="object").columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
print("Number of rows after filling missing value in categorical column with the mode value:", data.shape[0])
# Drop rows with outliers using z-score
# data[numeric_cols] = data[numeric_cols and (np.abs(stats.zscore(data)) < 5).all(axis=1)]
print("Drop rows with outliers using z-score:")
data.loc[(np.abs(stats.zscore(data[numeric_cols])) < 5).all(axis=1), numeric_cols] = data[numeric_cols]
print("Number of rows after dropping rows with outliers using z-score:", data.shape[0])

# Data wrangling

# Turn the date column into a datetime object
data["date"] = pd.to_datetime(data["date"])
# Extract the year from the date column
data["year"] = data["date"].dt.year
# Extract the month from the date column
data["month"] = data["date"].dt.month
# Extract the day from the date column
data["day"] = data["date"].dt.day
# Converts the time column into a timedelta object
# pd.to_timedelta()
# Change the data type of the age column to integer
data["age"] = pd.to_numeric(data["age"], downcast="integer")
# Rename columns
data = data.rename(columns={
        "manner_of_death": "death_manner",
        "signs_of_mental_illness": "mental_illness",
        "threat_level": "threat",
        "body_camera": "bodycam",
        "arms_category": "weapon_category",
    })
# Change the data type of the mental_illness column to boolean
data["mental_illness"] = data["mental_illness"].astype(bool)
# Change the data type of the bodycam column to boolean
data["bodycam"] = data["bodycam"].astype(bool)
# Create a new column called age_group that categorizes the age of the individuals shot
data = data.assign(age_group=data["age"].apply(lambda x: "child" if x < 18 else "adult" if x >= 18 and x < 65 else "elderly"))
data["age_group"] = pd.cut(data["age"], bins=[0, 17, 64, 100], labels=["child", "adult", "elderly"])
# Use query to filter the data based on whether the individual has a mental illness or not
mental_illness = True
no_mental_illness = False
data_mental_illness = data.query("mental_illness == @mental_illness").copy()
data_no_mental_illness = data.query("mental_illness == @no_mental_illness").copy()
# Use query to filter the data based on whether the individual was armed or not
data_armed = data.query("armed == True").copy()
data_not_armed = data.query("armed == False").copy()
# Group the data by year and month and calculate the total number of deaths for each year
deaths_by_year = data.groupby(["year", "month"]).sum(numeric_only=True)
# Sort values by year
deaths_by_year = deaths_by_year.sort_values(by="year", ascending=True)
# find the top 10 states with the highest number of deaths
top_10_states = data.groupby("state").size().sort_values(ascending=False).head(10)
# find the top 10 states with the highest number of deaths over the years
top_10_states_mean = (data.groupby("state").sum(numeric_only=True).sort_values(by="id", ascending=False).head(10))
# calculate mean age and number of deaths for each age group, race, and gender
mean_age_count_race_age = data.groupby(["age_group", "race", "gender"]).agg({"age": ["mean", "std"], "id": "count"}) # min, max, median, mode
mean_age_count_race_age.columns = ["mean_age", "std_age", "count"]


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
descriptive analysis
univariate analysis
bivariate analysis
multivariate analysis
variable transformation
factor Analysis
dimensionality reduction
"""
"""
classification model
regression model
clustering model
time series model
"""



"""
# Line chart of Number of Deaths by Year and Month: shows the number of deaths over the years and months.
sns.lineplot(data=deaths_by_year, x='month', y='age', hue='year').set(title="Line chart of Number of Deaths by Year and Month")
plt.show()
# good

# Create a line chart that shows the trend of number of deaths over time
sns.lineplot(x='year', y='id', data=data, estimator=np.mean)
plt.title('Trend of number of deaths over time')
plt.show()
# normal

# Histogram of Age of Individuals Shot: shows a distribution of ages of individuals shot.
sns.histplot(data=data, x='age', kde=True, bins=20).set(title="Histogram of Age of Individuals Shot")
plt.show()
# good

# Count plot of Race of Individuals Shot: shows the number of deaths for each race over the years.
data_race_year = data.groupby(['year', 'race'])['id'].count().unstack()
sns.barplot(x='year', y='total', hue='race', data=data_race_year.reset_index().melt(id_vars=['year'], var_name='race', value_name='total'), palette='muted').set(title="Count plot of Race of Individuals Shot")
# data_race_year = data.groupby(['year', 'race'])['id'].count().reset_index(name='count')
# sns.barplot(x='year', y='count', hue='race', data=data_race_year, palette='muted')
plt.show()
# can be done on other columns other than race

# Create JointGrid object and plot a scatterplot and histogram
grid = sns.JointGrid(data=data, x="age", y="race")
grid.plot_joint(sns.histplot, bins=20, cbar=True, cmap='Blues')
grid.plot_marginals(sns.histplot, kde=True, kde_kws={'bw_adjust': 1.5}, palette='muted')
plt.subplots_adjust(top=0.9)
grid.fig.suptitle("Distribution of age and race of deaths")
grid.set_axis_labels(xlabel="Age", ylabel="Race")
plt.show()
# good

# Histogram of Age of Different Race of Individuals Shot: shows a distribution of ages of different race of individuals shot.
sns.histplot(x='age', hue='race', data=data, kde=True, kde_kws={'bw_adjust': 1.5}, palette='muted')
plt.title('Age Distribution by Race')
plt.show()
# replaced but good

# Group the data by year, race, and gender and show the total number of deaths of each group over years
data_race_gender = data.groupby(['year', 'race', 'gender'])['id'].count().reset_index(name='count')
sns.lineplot(x='year', y='count', hue='race', style='gender', data=data_race_gender)
plt.title('Number of deaths by race and gender over time')
plt.yscale('log')
plt.show()
# good

"""
# Pie chart of Number of Deaths by Threat Level: shows the percentage of deaths for each threat level.
data_threat_count = data.groupby('threat')['id'].count().reset_index(name='count')
plt.pie(x=data_threat_count['count'], labels=data_threat_count['threat'].tolist(), autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white'})
plt.title('Number of deaths by threat level')
plt.show()
# normal

# Bar Plot of Number of Deaths by Mental Illness and Threat Level:
sns.countplot(data=data, x='mental_illness', hue='threat')
plt.title('Bar Plot of Number of Deaths by Mental Illness and Threat Level')
plt.show()
# good

# Heatmap of Gender and Weapon category:
data_gender_weapon = data.groupby(['gender', 'weapon_category'])['id'].count().reset_index(name='count')
pivot_data = data_gender_weapon.pivot(index='gender', columns='weapon_category', values='count')
sns.heatmap(pivot_data, cmap='Blues', linewidth=0.5, annot=True, fmt=".0f")
plt.title('Number of deaths by gender and weapon category')
plt.show()
# replaced

# Boxplot of Age and Weapon category
sns.boxplot(x='weapon_category', y='age', data=data)
plt.xticks(rotation=90, fontsize=8)
plt.title('Distribution of age and weapon category by race')
plt.show()
# no meaning
"""

# Count plot of Weapon Category of Individuals Shot: shows the number of deaths for each weapon category.
sns.countplot(data=data, x='weapon_category').set(title="Count plot of Weapon Category of Individuals Shot")
plt.xticks(rotation=90, fontsize=8)
plt.show()
# good

# sns.jointplot(x='age', y=data['race'].value_counts(), data=data, kind='reg')
# plt.title('Correlation between age and number of shootings by race')
# plt.show()
# not working

# Compute correlation matrix and create heatmap
corr = data.corr(numeric_only=True)
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
# good

# Create scatterplot matrix
# sns.pairplot(data, hue='race', palette='muted')
# plt.title('Pairwise Scatterplot Matrix')
# plt.show()
# too crowded, should choose some features

# Create a FacetGrid and map a histplot to the grid
grid = sns.FacetGrid(data, row='death_manner', col='race', margin_titles=True, palette='muted')
grid.map(sns.histplot, 'age', bins=20)
plt.show()
# replaced

# Create a lmplot of age and number of deaths
sns.regplot(x='age', y='id', data=data, scatter=True, fit_reg=True) # fit_reg: whether to fit a regression line
plt.title('Trend of age and number of deaths')
plt.show()
# use lmplot to plot trend line across a FacetGrid

# Regression plot of age and number of deaths
sns.lmplot(x='age', y='id', data=data, x_estimator=np.mean) # x_bins: number of bins to use when computing the estimate
plt.title('Regression plot of age and number of deaths')
plt.show()


# sns.scatterplot(x='age', y='id', hue='race', data=data, s=5)
# sns.countplot(data=data, x='gender', order=data['gender'].value_counts(ascending=False, normalize=True).index)
# sns.kdeplot(df['feature1'], shade=True, multiple='stack')
# sns.rugplot(x='age', data=data)
# using built-in trend function like pct_change, rolling, and diff
# use value_counts() to count the number of occurrences of each value in a column and show in sorted order


# df.plot(x='column_name1', y='column_name2', kind='line')
# kind: specifies the type of plot you want to create. Available options are line, bar, barh, hist, box, kde, density, area, pie, scatter, and hexbin.
# title: is used to set the title of the plot.
# xlabel: is used to set the label of the x-axis.
# ylabel: is used to set the label of the y-axis.
# legend: is used to specify whether to show the legend or not.
# grid: is used to specify whether to show the grid in the plot or not.
# xlim: is used to set the limits of the x-axis.
# ylim: is used to set the limits of the y-axis.
# xticks: is used to set the ticks of the x-axis.
# yticks: is used to set the ticks of the y-axis.
"""

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

# Hypothesis testing

"""
# One-sample t-test
from scipy import stats

# Calculate the mean and standard deviation of the age of the victims
mean_age = data['age'].mean()
std_age = data['age'].std()
print('Mean age of the victims: ', mean_age)
print('Standard deviation of age of the victims: ', std_age)

# Calculate the t-statistic and p-value
t_stat, p_value = stats.ttest_1samp(data['age'], 40)
# t-statistic is the number of standard deviations away from the mean our sample mean is
print('t-statistic: ', t_stat)
# p-value is the probability of getting a t-statistic as extreme as the one we got
print('p-value: ', p_value)


# Two-sample t-test
# Divide the data into two groups based on gender
male_age = data[data['gender'] == 'M']['age']
female_age = data[data['gender'] == 'F']['age']

# Calculate the means and standard deviations of the two groups
male_mean_age = male_age.mean()
female_mean_age = female_age.mean()
male_std_age = male_age.std()
female_std_age = female_age.std()

# Print the means and standard deviations
print('Mean age of male victims: ', male_mean_age)
print('Mean age of female victims: ', female_mean_age)
print('Standard deviation of age of male victims: ', male_std_age)
print('Standard deviation of age of female victims: ', female_std_age)

# Conduct a two-sample t-test
t_stat, p_value = stats.ttest_ind(male_age, female_age)

# Print the results
print('t-statistic: ', t_stat)
print('p-value: ', p_value)


# Paired t-test
# Assume we want to compare the number of records for each race and gender before and after the year 2017
# Create two related samples: count of records before and after 2017 for each race and gender
# Group the data by race, gender, and year, and count the number of records for each group
data_race_gender_year = data.groupby(['race', 'gender', 'year'])['id'].count().reset_index(name='count')

# Calculate the total number of records before and after 2017 for each race and gender
before_2017 = data_race_gender_year[data_race_gender_year['year'] < 2017].groupby(['race', 'gender'])['count'].sum().values
after_2017 = data_race_gender_year[data_race_gender_year['year'] >= 2017].groupby(['race', 'gender'])['count'].sum().values

# Conduct a paired t-test
t_stat, p_value = stats.ttest_rel(before_2017, after_2017)

# Print the results
print('t-statistic: ', t_stat)
print('p-value: ', p_value)


# One-way ANOVA
import scipy.stats as stats

data_race_gender = data.groupby(['year', 'race', 'gender'])['id'].count().reset_index(name='count')
# Assume we have a DataFrame called data_race_gender with columns 'year', 'race', 'gender', and 'count'
group1 = data_race_gender[data_race_gender['race'] == 'Asian']['count']
group2 = data_race_gender[data_race_gender['race'] == 'Black']['count']
group3 = data_race_gender[data_race_gender['race'] == 'Hispanic']['count']
group4 = data_race_gender[data_race_gender['race'] == 'Native']['count']
group5 = data_race_gender[data_race_gender['race'] == 'Other']['count']
group6 = data_race_gender[data_race_gender['race'] == 'White']['count']

# Conduct a one-way ANOVA
f_stat, p_value = stats.f_oneway(group1, group2, group3, group4, group5, group6)

# Print the results
# F-statistic: the ratio of the variance between groups to the variance within groups
print('F-statistic: ', f_stat)
# p-value: the probability of getting an F-statistic as extreme as the one we got
print('p-value: ', p_value)


# Chi-squared test
# Create a contingency table of the race and gender of the victims
contingency_table = pd.crosstab(data['race'], data['gender'])

# Perform a chi-squared test
chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)

# Print the results
print('Chi-squared statistic: ', chi2_stat)
print('p-value: ', p_val)


# Correlation
# Calculate the Pearson correlation coefficient
# corr, p_value = stats.pearsonr(data['age'], data['death_manner'])
# print('Pearson correlation coefficient:', corr)
# print('p-value:', p_value)

# Calculate the Spearman rank correlation coefficient
# corr, p_value = stats.spearmanr(data['race'], data['threat'])
# print('Spearman rank correlation coefficient:', corr)
# print('p-value:', p_value)

# Calculate Kendall's tau coefficient
# corr, p_value = stats.kendalltau(data['mental_illness'], data['flee'])
# print("Kendall's tau coefficient:", corr)
# print('p-value:', p_value)


# Linear regression
# stats.linregress(df['feature1'], df['feature2'])


# Non-parametric tests

# group the data by year, race, and gender and count the number of observations
data_race_gender = data.groupby(['year', 'race', 'gender'])['id'].count().reset_index(name='count')

# perform Mann-Whitney U test to compare counts of two groups (e.g. Male and Female)
result_mw = stats.mannwhitneyu(data_race_gender[data_race_gender['gender'] == 'M']['count'],
                            data_race_gender[data_race_gender['gender'] == 'F']['count'])

# statistics: U-value, represents the probability that a randomly selected observation from one group will have a higher value than a randomly selected observation from the other group
# the larger the U value, the greater the difference between the two groups
# p-value: the probability of getting a U-value as extreme as the one we got
print('Mann-Whitney U test result: ', result_mw)

# perform Kruskal-Wallis H test to compare counts of more than two groups (e.g. race)
result_kw = stats.kruskal(data_race_gender[data_race_gender['race'] == 'Asian']['count'],
                        data_race_gender[data_race_gender['race'] == 'Black']['count'],
                        data_race_gender[data_race_gender['race'] == 'Hispanic']['count'],
                        data_race_gender[data_race_gender['race'] == 'White']['count'])

# statistics: H-value, represents the degree of difference between the groups
# the larger the H-value, the greater the difference between the groups
# p-value: the probability of getting a H-value as extreme as the one we got
print('Kruskal-Wallis H test result: ', result_kw)

"""


# Predictive modeling
# Linear regression
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
# Logistic regression
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()
# KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
# Decision tree
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
# Random forest
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
# Naive Bayes
from sklearn.naive_bayes import GaussianNB

naive_bayes = GaussianNB()
# SVM
from sklearn.svm import SVC

svm = SVC(kernel="linear")
# Neural network
from sklearn.neural_network import MLPClassifier

neural_network = MLPClassifier()

# USe these models to predict the target variable
model = linear_regression
# model = logistic_regression
# model = knn
# model = decision_tree
# model = random_forest
# model = naive_bayes
# model = svm
# model = neural_network

"""
# encode the categorical variables


# Specify the input and output variables
data_race_gender = data.groupby(['year', 'race', 'gender'])['id'].count().reset_index(name='count')
X = data_race_gender.drop('count', axis=1)
X_scaled = preprocessing.scale(X)
y = data_race_gender['count']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict the target variable on the test data
y_pred = model.predict(X_test)
"""

# Cross-validation
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
# print(scores.mean())

# Hyperparameter tuning
# from sklearn.model_selection import GridSearchCV
# param_grid = dict(n_neighbors=np.arange(1, 50))
# grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
# grid.fit(X, y)
# print(grid.best_score_)
# print(grid.best_params_)
# print(grid.best_estimator_)
"""
# Model evaluation
from sklearn.metrics import accuracy_score, mean_squared_error
acc = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_squared = model.score(X_test, y_test)
print('Accuracy: ', acc)
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r_squared)

# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
"""
# Classification report
# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))

# ROC curve
# from sklearn.metrics import roc_curve
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr, label='KNN')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('KNN ROC Curve')
# plt.show()

# AUC
# from sklearn.metrics import roc_auc_score
# roc_auc_score(y_test, y_pred_prob)

# Feature selection
# Recursive feature elimination
# from sklearn.feature_selection import RFE
# rfe = RFE(model, 3)
# rfe = rfe.fit(X, y)
# print(rfe.support_)
# print(rfe.ranking_)
# Principal component analysis
# from sklearn.decomposition import PCA
# pca = PCA(n_components=3)
# X = pca.fit_transform(X)
# print(pca.explained_variance_ratio_)
# Feature importance
# model.feature_importances_

# Feature engineering
# Binning
# df['age'] = pd.cut(df['age'], bins=[0, 18, 25, 40, 60, 100], labels=['child', 'young adult', 'adult', 'middle-aged', 'senior'])
# df['age'] = pd.qcut(df['age'], q=4, labels=['child', 'young adult', 'adult', 'middle-aged', 'senior'])
# One-hot encoding
# df = pd.get_dummies(df, columns=['column1', 'column2'], drop_first=True)
# Label encoding
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# df['column1'] = label_encoder.fit_transform(df['column1'])
# df['column2'] = label_encoder.fit_transform(df['column2'])
# Polynomial features
# from sklearn.preprocessing import PolynomialFeatures
# polynomial_features = PolynomialFeatures(degree=2)
# X = polynomial_features.fit_transform(X)
# Interaction features
# df['column1*column2'] = df['column1'] * df['column2']
# df['column1^2'] = df['column1'] ** 2

# Dimensionality reduction
# PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components=3)
# X = pca.fit_transform(X)
# print(pca.explained_variance_ratio_)
# LDA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# lda = LinearDiscriminantAnalysis(n_components=2)
# X = lda.fit_transform(X, y)
# print(lda.explained_variance_ratio_)
# t-SNE
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2)
# X = tsne.fit_transform(X)

# Clustering
# K-means
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
# y_kmeans = kmeans.fit_predict(X)
# plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
# plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
# plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
# plt.title('Clusters of customers')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.legend()
# plt.show()
# Hierarchical clustering
# from sklearn.cluster import AgglomerativeClustering
# hierarchical_clustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
# y_hierarchical_clustering = hierarchical_clustering.fit_predict(X)
# plt.scatter(X[y_hierarchical_clustering == 0, 0], X[y_hierarchical_clustering == 0, 1], s=100, c='red', label='Cluster 1')
# plt.scatter(X[y_hierarchical_clustering == 1, 0], X[y_hierarchical_clustering == 1, 1], s=100, c='blue', label='Cluster 2')
# plt.scatter(X[y_hierarchical_clustering == 2, 0], X[y_hierarchical_clustering == 2, 1], s=100, c='green', label='Cluster 3')
# plt.title('Clusters of customers')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.legend()
# plt.show()
# DBSCAN
# from sklearn.cluster import DBSCAN
# dbscan = DBSCAN(eps=3, min_samples=4)
# y_dbscan = dbscan.fit_predict(X)
# plt.scatter(X[y_dbscan == 0, 0], X[y_dbscan == 0, 1], s=100, c='red', label='Cluster 1')
# plt.scatter(X[y_dbscan == 1, 0], X[y_dbscan == 1, 1], s=100, c='blue', label='Cluster 2')
# plt.scatter(X[y_dbscan == 2, 0], X[y_dbscan == 2, 1], s=100, c='green', label='Cluster 3')
# plt.title('Clusters of customers')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.legend()
# plt.show()

# Time Series Analysis
# from statsmodels.tsa.seasonal import seasonal_decompose
# decomposition = seasonal_decompose(df['column_name'], model='additive', freq=1)
# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid
# plt.subplot(411)
# plt.plot(df['column_name'], label='Original')
# plt.legend(loc='best')
# plt.subplot(412)
# plt.plot(trend, label='Trend')
# plt.legend(loc='best')
# plt.subplot(413)
# plt.plot(seasonal, label='Seasonality')
# plt.legend(loc='best')
# plt.subplot(414)
# plt.plot(residual, label='Residuals')
# plt.legend(loc='best')
# plt.tight_layout()

# Cluster analysis
# K-means clustering
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X)
# df['segment'] = kmeans.predict(X)

# Sentiment analysis
# TextBlob
# from nltk.sentiment import SentimentIntensityAnalyzer
# sentiment = SentimentIntensityAnalyzer()
# df['sentiment'] = df['text'].apply(lambda x: sentiment.polarity_scores(x)['compound'])

# Association rules
# Apriori
# from apyori import apriori
# rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)
# results = list(rules)
# print(results)
# Eclat
# from pyECLAT import ECLAT
# eclat = ECLAT()
# eclat.fit(transactions)
# results = eclat.transform(min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)
# print(results)

# Reinforcement learning
# Upper confidence bound
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
# N = 10000
# d = 10
# ads_selected = []
# numbers_of_selections = [0] * d
# sums_of_rewards = [0] * d
# total_reward = 0
# for n in range(0, N):
#     ad = 0
#     max_upper_bound = 0
#     for i in range(0, d):
#         if (numbers_of_selections[i] > 0):
#             average_reward = sums_of_rewards[i] / numbers_of_selections[i]
#             delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
#             upper_bound = average_reward + delta_i
#         else:
#             upper_bound = 1e400
#         if upper_bound > max_upper_bound:
#             max_upper_bound = upper_bound
#             ad = i
#     ads_selected.append(ad)
#     numbers_of_selections[ad] = numbers_of_selections[ad] + 1
#     reward = dataset.values[n, ad]
#     sums_of_rewards[ad] = sums_of_rewards[ad] + reward
#     total_reward = total_reward + reward
# plt.hist(ads_selected)
# plt.title('Histogram of ads selections')
# plt.xlabel('Ads')
# plt.ylabel('Number of times each ad was selected')
# plt.show()
# Thompson sampling
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
# N = 10000
# d = 10
# ads_selected = []
# numbers_of_rewards_1 = [0] * d
# numbers_of_rewards_0 = [0] * d
# total_reward = 0
# for n in range(0, N):
#     ad = 0
#     max_random = 0
#     for i in range(0, d):
#         random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
#         if random_beta > max_random:
#             max_random = random_beta
#             ad = i
#     ads_selected.append(ad)
#     reward = dataset.values[n, ad]
#     if reward == 1:
#         numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
#     else:
#         numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
#     total_reward = total_reward + reward
# plt.hist(ads_selected)
# plt.title('Histogram of ads selections')
# plt.xlabel('Ads')
# plt.ylabel('Number of times each ad was selected')
# plt.show()

# Natural language processing
# Bag of words model
# import re
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# corpus = []
# for i in range(0, 1000):
#     review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
#     review = review.lower()
#     review = review.split()
#     ps = PorterStemmer()
#     review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#     review = ' '.join(review)
#     corpus.append(review)
# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer(max_features=1500)
# X = cv.fit_transform(corpus).toarray()
# y = dataset.iloc[:, 1].values
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)

# TF-IDF
# import re
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# corpus = []
# for i in range(0, 1000):
#     review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
#     review = review.lower()
#     review = review.split()
#     ps = PorterStemmer()
#     review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#     review = ' '.join(review)
#     corpus.append(review)
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(max_features=1500)
# X = tfidf.fit_transform(corpus).toarray()
# y = dataset.iloc[:, 1].values
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)

# Word2Vec


# Insight generation

# Export the processed DataFrame to a CSV file
# data.to_csv('processed_dataset.csv', index=False)
"""
index: specifies whether to include the index of the DataFrame in the CSV file or not.
header: specifies whether to include the column names of the DataFrame in the CSV file or not.
columns: is used to specify the columns to be included in the CSV file.
sep: is used to specify the separator to be used in the CSV file.
na_rep: is used to specify the string to be used for missing values in the CSV file.
"""
# data.to_excel('data.xlsx', index=False)
# data.to_json('data.json', orient='records', lines=True)
# data.to_pickle('data.pkl')
# data.to_html('data.html', index=False)

# TODO: Relationship between variables
# Median polish
# Statistical model
# Data analysis

"""
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Create a logistic regression model and fit it to the training data
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Make predictions on the test data
predictions = lr.predict(X_test)

# Evaluate the model's accuracy
accuracy = lr.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")
"""

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
"""
Here are the basic steps I would follow for a data analysis project in Python:
Get the data - This could be from a file, API, database, etc. Load it into a pandas DataFrame for convenience.
Explore the data - Look at summaries, distributions, spot check for errors or anomalies, look at correlations, etc. This is to get a feel for the data and spot any major issues.
Prepare the data - Clean, transform, encode, scale, etc. the data as needed to get it into a state suitable for modeling.
Model the data - Fit machine learning models or statistical models and evaluate their performance.
Evaluate and optimize - Tune hyperparameters, try different models, engineering features, etc. to optimize the results.
Present and interpret - Create data visualizations, summaries, and reports to present your findings and interpret the results.
"""
"""
Here is a general outline for designing a data analysis project in Python:
Define the problem statement and research questions: Start by clearly defining the problem you want to solve and the research questions you want to answer through data analysis.
Gather and clean the data: Identify the data sources you need to answer your research questions and gather the data. Clean and preprocess the data to ensure it is in a usable format for analysis.
Exploratory data analysis (EDA): Conduct a thorough EDA to get a better understanding of the data and identify any patterns, trends, or outliers that may affect your analysis. Use visualizations and descriptive statistics to summarize the data.
Feature engineering: If necessary, create new features or transform existing features to better represent the data and improve your analysis.
Model building: Select the appropriate statistical or machine learning models to answer your research questions. Train and validate the models using the data.
Model evaluation: Evaluate the performance of the models and interpret the results. Determine whether the models are reliable and accurate enough to answer the research questions.
Communication and visualization: Communicate the results of your analysis through clear and concise visualizations and reports. Use storytelling techniques to effectively communicate the insights and implications of your analysis.
Iteration: Iterate through the analysis process as necessary to refine your models or explore new research questions.
"""
"""
Data loading module: This module will contain functions to load historical stock data from different sources such as Yahoo Finance, Google Finance, etc. This module will be responsible for providing a Pandas dataframe object containing the historical data.

Feature engineering module: This module will contain functions to create new features from the historical data that can help improve the accuracy of the model. For example, creating moving averages, rolling standard deviation, or technical indicators.

Data processing module: This module will contain functions to prepare the data for the model. This includes defining the target and feature variables, splitting the data into training and testing sets, and normalizing the data.

Modeling module: This module will contain functions to train and evaluate the model. This includes defining the model architecture, training the model, evaluating the performance of the model on the test set, and making predictions on new data.

Visualization module: This module will contain functions to visualize the data and model results. This includes creating plots of historical stock prices, feature engineering output, model performance, and prediction results.
"""
"""
text cleaning: remove punctuation, numbers, and special characters, remove stopwords, and perform stemming or lemmatization
text vectorization: convert text to numbers, using bag-of-words, TF-IDF, or word embeddings, with libraries such as scikit-learn, NLTK, or spaCy
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()

# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(max_features=1500)
# X = tfidf.fit_transform(corpus).toarray()

df['Label'].replace({'ham': 1, 'spam': 0}, inplace=True)

from sklearn.model_selection import train_test_split
from data analysis template import data_race_year
X_train, X_test, y_train, y_test = train_test_split(X, df['Label'], test_size=0.20, random_state=0)
"""