
# EDA (Exploratory Data Analysis) - Data Analysis Template

# US Police Shootings Dataset

# https://colab.research.google.com/drive/10pcm9FA6uuboKCPNjlk4HHKX0_MnpfTZ?usp=sharing
# write me python code for basic data exploration and visualization of the dataset
# date column in my dataset is in object datatype, convert it to datetime format column
# what are the count of deaths by age, show in visualization
# can we use seaborn library with a regression line to plot count of deaths by age as a visualization

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load the dataset
loc_file = 'https://raw.githubusercontent.com/MuhammadAli437/ChatGPT/main/shootings.csv'
data = pd.read_csv(loc_file)



# Get summary statistics of the data
print(data.describe())

# Get the data types of the columns
print(data.info())

# Get the first 5 rows of the data
print(data.head())

# Get the number of missing values in each column
print(data.isnull().sum())

# Get the number of missing values in each column
print(data.isna().sum())

# Get the number of duplicated rows
print(data.duplicated().sum())

# Get the number of unique values in each column
print(data.nunique())

# Check for outliers
print(data.boxplot())

# Check for correlations
print(data.corr())

# Check for class imbalance
print(data.groupby('target').size())



# Data cleaning
# Turn the date column into a datetime object
data["date"] = pd.to_datetime(data["date"])
# Turn the age column into an integer
data['age'] = data['age'].astype(int)

# Plot a histogram of the age of the individuals shot
plt.hist(data['age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age of Individuals Shot')
plt.show()

# Plot a bar chart of the race of the individuals shot
plt.bar(data['race'].value_counts().index, data['race'].value_counts())
plt.xlabel('Race')
plt.ylabel('Frequency')
plt.title('Bar Chart of Race of Individuals Shot')
plt.show()

# Plot a line chart of the number of deaths over the years
# Extract the year from the date column
data['year'] = data['date'].dt.year
# Group the data by year and calculate the total number of deaths for each year
deaths_by_year = data.groupby(['year'])['name'].count()
# Plot the number of deaths over the years and months
deaths_by_year.plot(kind='line')
plt.xlabel('Year')
plt.ylabel('Number of Deaths')
plt.title('Number of Deaths by Year')
plt.show()

# Plot a line chart of the number of deaths over the years and months
# Extract the month from the date column
data['month'] = data['date'].dt.month
# Group the data by year and month and calculate the total number of deaths for each year and month
deaths_by_year_month = data.groupby(['year', 'month'])['name'].count()
# Plot the number of deaths over the years and months
deaths_by_year_month.plot(kind='line')
plt.xlabel('Year and Month')
plt.ylabel('Number of Deaths')
plt.title('Number of Deaths by Year and Month')
plt.show()

# Group the data by age and calculate the total number of deaths for each age
deaths_by_age = data.groupby(['age'])['name'].count()
# Plot the histogram of the number of deaths by age
plt.hist(deaths_by_age, bins=20)
plt.xlabel('Frequency')
plt.ylabel('Number of Deaths')
plt.title('Number of Deaths by Age')
plt.show()

import seaborn as sns

# Group the data by age and calculate the total number of deaths for each age
deaths_by_age = data.groupby(['age'])['name'].count()
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
deaths_by_race_year = data.groupby(['year', 'race'])['name'].count()
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
deaths_by_state = data.groupby(['state'])['name'].count().reset_index(name='counts')

# Create a choropleth map of the statewise shooting data
fig = px.choropleth(deaths_by_state,
                    locations='state',  # DataFrame column with locations
                    locationmode = 'USA-states', # set of locations match entries in `locations`
                    color='counts',  # DataFrame column with color values
                    title='Statewise Shooting Data')
fig.show()

# TODO: Relationship between variables



