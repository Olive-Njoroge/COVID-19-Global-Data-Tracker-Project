import pandas as pd

#Load the dataset
df = pd.read_csv('owid-covid-data.csv')

#Check columns
print(df.columns)

#Preview rows
print(df.head())

#Identify missing values
missing_values = df.isnull().sum()
print(missing_values)

# Filter for countries of interest
countries = ['Kenya', 'United States', 'India']
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

#Fill remaining missing values with zeros.
df = df.fillna(0)