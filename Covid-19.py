import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # <-- Added for choropleth map

# Load the dataset
df = pd.read_csv('owid-covid-data.csv')

# Check columns and preview data
print("Columns:", df.columns)
print(df.head())

# Identify missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:\n", missing_values)

# Filter for countries of interest
countries = ['United States', 'India', 'Italy']
filtered_df = df[df['location'].isin(countries)].copy()

print("\nFiltered Data for countries of interest:")
print(filtered_df.head())

# Drop rows with missing values in critical columns
critical_columns = ['date', 'location', 'total_cases', 'total_deaths', 'new_cases', 'new_deaths', 'total_vaccinations']
df_cleaned = df.dropna(subset=critical_columns)
print("\nData after dropping rows with missing critical columns:")
print(df_cleaned.head())

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])
filtered_df['date'] = pd.to_datetime(filtered_df['date'])

# Fill remaining missing values with zeros
df.fillna(0, inplace=True)
filtered_df.fillna(0, inplace=True)

# Calculate death rate safely (avoid division by zero)
filtered_df['death_rate'] = filtered_df.apply(
    lambda row: (row['total_deaths'] / row['total_cases']) if row['total_cases'] > 0 else 0,
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

# 5. Heatmap of correlations among numeric columns for filtered countries
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

# --- Vaccination rollout analysis ---

# 7. Plot cumulative vaccinations over time for selected countries
plt.figure(figsize=(12,6))
for country in countries:
    data = filtered_df[filtered_df['location'] == country]
    plt.plot(data['date'], data['total_vaccinations'], label=country)
plt.title('Cumulative COVID-19 Vaccinations Over Time')
plt.xlabel('Date')
plt.ylabel('Total Vaccinations (doses)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Calculate % vaccinated population (approximate)
filtered_df['percent_vaccinated'] = filtered_df.apply(
    lambda row: (row['total_vaccinations'] / row['population']) * 100 if row['population'] > 0 else 0,
    axis=1
)

# Get latest vaccination % per country
latest_vax = filtered_df.groupby('location').apply(lambda x: x.loc[x['date'].idxmax()])

plt.figure(figsize=(8,6))
sns.barplot(data=latest_vax.reset_index(drop=True), x='location', y='percent_vaccinated', palette='magma')
plt.title('Percentage of Population Vaccinated (based on total doses)')
plt.xlabel('Country')
plt.ylabel('Percent Vaccinated (%)')
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

# Optional: Pie chart of vaccinated vs unvaccinated for each country (latest date)
for country in countries:
    latest = latest_vax.loc[country]
    vaccinated = latest['total_vaccinations']
    population = latest['population']
    unvaccinated = max(population - vaccinated, 0)
    plt.figure(figsize=(6,6))
    plt.pie(
        [vaccinated, unvaccinated],
        labels=['Vaccinated', 'Unvaccinated'],
        colors=['#4CAF50', '#F44336'],
        autopct='%1.1f%%',
        startangle=140
    )
    plt.title(f'Vaccination Coverage in {country} (Latest Data)')
    plt.show()

# --- Choropleth Map for total vaccinations ---

# Prepare latest data by country with iso_code (remove missing iso_code)
latest_data = df[df['date'] == latest_date].copy()
latest_data = latest_data[latest_data['iso_code'].notnull() & (latest_data['iso_code'] != '')]

fig = px.choropleth(
    latest_data,
    locations='iso_code',       # ISO 3-letter codes
    color='total_vaccinations',
    hover_name='location',
    color_continuous_scale='Viridis',
    title=f'COVID-19 Total Vaccinations by Country (as of {latest_date.date()})',
    labels={'total_vaccinations': 'Total Vaccinations'}
)

fig.update_layout(
    geo=dict(showframe=False, showcoastlines=True)
)

fig.show()
