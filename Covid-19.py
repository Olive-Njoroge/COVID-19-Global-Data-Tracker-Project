import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('owid-covid-data.csv')

# Check columns
print(df.columns)

# Preview rows
print(df.head())

# Identify missing values
missing_values = df.isnull().sum()
print(missing_values)

# Filter for countries of interest
countries = ['Angola', 'United States', 'India']
filtered_df = df[df['location'].isin(countries)]

# Show the result
print(filtered_df)

# Drop rows with missing values in critical columns
critical_columns = ['date', 'location', 'total_cases', 'total_deaths', 'new_cases', 'new_deaths', 'total_vaccinations']
df_cleaned = df.dropna(subset=critical_columns)

# Show result
print(df_cleaned.head())

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Check the result
print(df['date'].dtype)
print(df.head())

# Fill remaining missing values with zeros
df = df.fillna(0)

# --- Begin added analysis and visualizations ---

# Work with filtered_df for consistent processing: filter again after filling NAs
filtered_df = df[df['location'].isin(countries)].copy()
filtered_df.fillna(0, inplace=True)

# Calculate death rate safely (avoid division by zero)
filtered_df['death_rate'] = filtered_df.apply(
    lambda row: row['total_deaths'] / row['total_cases'] if row['total_cases'] > 0 else 0,
    axis=1
)

# 1. Plot total cases over time for selected countries
plt.figure(figsize=(12,6))
for country in countries:
    data = filtered_df[filtered_df['location'] == country]
    plt.plot(data['date'], data['total_cases'], label=country)
plt.title('Total COVID-19 Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Plot total deaths over time
plt.figure(figsize=(12,6))
for country in countries:
    data = filtered_df[filtered_df['location'] == country]
    plt.plot(data['date'], data['total_deaths'], label=country)
plt.title('Total COVID-19 Deaths Over Time')
plt.xlabel('Date')
plt.ylabel('Total Deaths')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Compare daily new cases between countries
plt.figure(figsize=(12,6))
for country in countries:
    data = filtered_df[filtered_df['location'] == country]
    plt.plot(data['date'], data['new_cases'], label=country)
plt.title('Daily New COVID-19 Cases')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Bar chart: top 10 countries by total cases on the latest date
latest_date = df['date'].max()
latest_data = df[df['date'] == latest_date].copy()
latest_data = latest_data.sort_values(by='total_cases', ascending=False).head(10)

plt.figure(figsize=(12,6))
sns.barplot(data=latest_data, x='location', y='total_cases', palette='viridis')
plt.title(f'Top 10 Countries by Total COVID-19 Cases (as of {latest_date.date()})')
plt.xlabel('Country')
plt.ylabel('Total Cases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Optional: Heatmap of correlations among numeric columns for filtered countries
numeric_cols = ['total_cases', 'total_deaths', 'new_cases', 'new_deaths', 'total_vaccinations']
corr_df = filtered_df[numeric_cols].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of COVID-19 Metrics')
plt.tight_layout()
plt.show()

# 6. Descriptive statistics for filtered countries
print("\nDescriptive Statistics for Selected Countries:")
print(filtered_df.describe())
