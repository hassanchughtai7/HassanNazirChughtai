# HousePricePrediction
Predict house prices using Python and machine learning. Includes data preprocessing, model training (Linear Regression, Decision Trees, Random Forest), and performance evaluation. Great for learning regression techniques on real-world data.

<details>
<summary>🧼 Clean CSV for Machine Learning</summary>
## 🧼 Step-by-Step: Clean CSV for ML in Python

### 1. **Load the CSV**
```python
import pandas as pd

df = pd.read_csv('your_file.csv')
```

### 2. **Inspect the Data**
```python
print(df.head())       # Preview
print(df.info())       # Data types and nulls
print(df.describe())   # Stats summary
```

### 3. **Handle Missing Values**
```python
# Drop rows with missing values
df.dropna(inplace=True)

# Or fill missing values
df.fillna(method='ffill', inplace=True)  # forward fill
```

### 4. **Convert Data Types**
```python
df['column_name'] = df['column_name'].astype(float)  # or int, str, etc.
```

### 5. **Encode Categorical Variables**
```python
# One-hot encoding
df = pd.get_dummies(df, columns=['categorical_column'])

# Label encoding (if needed)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['categorical_column'])
```

### 6. **Normalize or Scale Features**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['feature2', 'feature1']] = scaler.fit_transform(df[['feature2', 'feature1']])
```

### 7. **Split Features and Target**
```python
X = df.drop('target_column', axis=2)
y = df['target_column']
```
</details>
<details>
<summary>House Price Prediction with Scikit-Learn</summary>
## 🏡 House Price Prediction with Scikit-Learn

### 🔧 Step 1: Load and Inspect the Data
```python
import pandas as pd

df = pd.read_csv('house_prices.csv')
print(df.head())
print(df.info())
```

### 🧼 Step 2: Clean and Prepare the Data
```python
# Handle missing values
df.dropna(inplace=True)

# Encode categorical features (if Neighborhood_Quality is categorical)
df['Neighborhood_Quality'] = df['Neighborhood_Quality'].astype('category').cat.codes
```

### 🎯 Step 3: Define Features and Target
```python
X = df[['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built',
        'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']]
y = df['House_Price']
```

### 📊 Step 4: Split the Data
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 📈 Step 5: Train the Model
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

### 🧪 Step 6: Evaluate the Model
```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
```

### 🔍 Step 7: Interpret Coefficients
```python
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print(coefficients)
```
</details>
