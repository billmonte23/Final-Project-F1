# Main Title
## Section Title
### Subsection

This project focuses on data preprocessing and feature engineering using Formula 1 datasets.The objective of this study is to predict Formula 1 race performance (points scored) and identify the most influential factors affecting race outcomes using historical race dat

# The following datasets are used in this project:
- drivers.csv  
- constructors.csv  
- races.csv  
- results.csv  
- qualifying.csv

# Import Libraries
import pandas as pd
# Load Datasets

# Replace '\N' values (missing values in dataset)
df = df.replace('\\N', None)

drivers = pd.read_csv('/content/drivers.csv')
constructors = pd.read_csv('/content/constructors.csv')
races = pd.read_csv('/content/races.csv')
results = pd.read_csv('/content/results.csv')
qualifying = pd.read_csv('/content/qualifying.csv')

# Merge Datasets
## We combine datasets step-by-step using common keys.
df = results.merge(races[['raceId', 'year']], on='raceId', how='left')

df = df.merge(
    qualifying[['raceId', 'driverId', 'q1', 'q2', 'q3']],
    on=['raceId', 'driverId'],
    how='left'
)
## Select Relevant Features
df = df[['raceId', 'year', 'driverId', 'constructorId',
         'grid', 'positionOrder', 'q1', 'q2', 'q3']]

# Data Cleaning
## Remove missing or invalid values
## Handle \N values
## Drop null rows
df = df.replace('\\N', None)
df = df.dropna()

# Data Type Conversion
## Convert columns to numeric format for analysis.
df['final_position'] = pd.to_numeric(df['positionOrder'])
df['grid'] = pd.to_numeric(df['grid'])

# Feature Engineering
## Create a new feature:

top3 → 1 if driver finished in top 3, else 0
df['top3'] = df['final_position'].apply(lambda x: 1 if x <= 3 else 0)

# Drop Unnecessary Columns
df = df.drop(columns=['positionOrder'])
## Save Clean Dataset
df.to_csv("f1_clean_dataset_final.csv", index=False)

# Exploratory Data Analysis (EDA)
# Plot relationship between grid position and final position

plt.scatter(df['grid'], df['position'])

plt.xlabel("Starting Grid Position")
plt.ylabel("Final Position")
plt.title("Grid Position vs Final Race Position")

plt.show()

### Insight- Drivers starting in better grid positions tend to finish in better positions, showing a strong relationship between qualifying performance and race outcome.

# Feature Engineering
## Calculate position gain (performance improvement)
df['position_gain'] = df['grid'] - df['position']

# Create podium flag (top 3 finishers)
df['podium'] = df['position'].apply(lambda x: 1 if x <= 3 else 0)

# Create performance score
df['performance_score'] = df['points'] / (df['position'] + 1)

### Insight-Position gain helps identify drivers who outperform expectations, while podium classification simplifies performance evaluation.

# Machine Learning Preparation
# Select features
X = df[['grid', 'laps', 'fastestLapSpeed', 'position_gain']]

# Target variable
y = df['points']

# Split dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model Training
## Import Random Forest model
from sklearn.ensemble import RandomForestRegressor

# Initialize model
model = RandomForestRegressor()

# Train model
model.fit(X_train, y_train)

# Model Evaluation

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R2 Score:", r2)

### Insight-The model evaluates how accurately race performance (points) can be predicted based on key race features.

# Feature Importance
# Extract feature importance
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': importance
}).sort_values(by='importance', ascending=False)

# Plot feature importance
feature_importance.plot(x='feature', y='importance', kind='bar')

plt.title("Feature Importance")
plt.show()
