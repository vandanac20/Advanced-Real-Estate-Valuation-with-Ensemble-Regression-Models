import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
#Read the Data
data = pd.read_csv('RealEstate_Valuation.csv')
data['Total_Area'].head()
# Convert 'Price' column to numeric
def convert_price(price):
    price = price.replace('₹', '').replace(' Cr', 'e7').replace(' L', 'e5').replace('k', 'e3')
    return eval(price)
# Separate features and target variable
X = data.drop(['Price', 'Name', 'Property Title'], axis=1)  # Drop irrelevant columns
y = data['Price']

data['Price'] = data['Price'].apply(convert_price)
data['Price'].head()
'''
# Before preprocessing, make sure X and y are aligned
print(f"Shape of X before preprocessing: {X.shape}")
print(f"Shape of y before preprocessing: {y.shape}")
'''

X_cleaned = X.dropna()
y_cleaned = y[X_cleaned.index]  # Ensure y is aligned with the rows in X
'''
# After any preprocessing steps, make sure the number of rows in X and y match:
print(f"Shape of X after preprocessing: {X_cleaned.shape}")
print(f"Shape of y after preprocessing: {y_cleaned.shape}")
'''
#Find Missing values
missing_values = data.isnull().sum()
'''
print("Missing Values:")
print(missing_values)
'''
# Separate features and target variable
X = data.drop(['Price', 'Name', 'Property Title'], axis=1)  # Drop irrelevant columns
y = data['Price']
from scipy.stats import zscore

# List of columns for which to calculate Z-scores
columns_to_check = ['Price', 'Total_Area', 'Baths', 'Price_per_SQFT']

# Calculate Z-scores for the specified columns
z_scores = data[columns_to_check].apply(zscore)
'''
# Display the Z-scores
print("Z-scores for selected columns:")
print(z_scores)
'''
# Identify rows with outliers (e.g., Z-score > 3 or < -3)
outliers = (z_scores.abs() > 3).any(axis=1)
'''
print("\nRows with outliers in the specified columns:")
'''
data[outliers].head()
threshold = 3

# Keep rows where all Z-scores are within the threshold
data_cleaned_zscore = data[(z_scores.abs() <= threshold).all(axis=1)]
'''
print(f"Original data size: {data.shape}")
print(f"Data size after removing outliers using Z-score: {data_cleaned_zscore.shape}")
'''
# Identify numerical and categorical columns
numerical_cols = ['Total_Area', 'Baths','Price_per_SQFT']
categorical_cols = ['Location', 'Description','Balcony']
standard_scaler = StandardScaler()
data[numerical_cols] = standard_scaler.fit_transform(data[numerical_cols])
'''
# Display the transformed data
print("Standardized Features:")
print(data[numerical_cols].head())
'''
from sklearn.decomposition import PCA

# Apply PCA to numerical data
pca = PCA(n_components=2)  # Reduce to 2 components
pca_result = pca.fit_transform(data[['Total_Area', 'Price', 'Baths']])
# print(f"Explained Variance: {pca.explained_variance_ratio_}")
# Combine preprocessing steps into a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # Apply StandardScaler to numerical columns
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)  # One-Hot Encode categorical variables, set sparse_output=False to get a dense array
    ])
# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)
preprocessor.get_feature_names_out()
# Convert the preprocessed features to a DataFrame for better interpretability
feature_names = preprocessor.get_feature_names_out()
X_preprocessed_df = pd.DataFrame(X_preprocessed,columns=feature_names)
# Assuming X and y are preprocessed
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)


# print(f"Training Set: {X_train.shape}, Validation Set: {X_valid.shape}, Test Set: {X_test.shape}")
# Train-test split using the preprocessed data
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed_df, y, test_size=0.9, random_state=42)
pipeline = Pipeline(steps=[
    ('model', RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=2, random_state=42))
])
# Fit the model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluate performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
'''
print(f"📊 Random Forest Performance:")
print(f"🔹 **RMSE**: {rmse:.4f}")
print(f"🔹 **MAE**: {mae:.4f}")
print(f"🔹 **R² Score**: {r2:.4f}")
'''
gb_model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model on the training data
gb_model.fit(X_train, y_train)

# Predict using the trained Gradient Boosting model
y_pred_gb = gb_model.predict(X_test)
X_train.columns = X_train.columns.str.replace(r'[^\w\s]', '', regex=True)
X_test.columns = X_test.columns.str.replace(r'[^\w\s]', '', regex=True)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, tree_method='hist', random_state=42)

# Train the model on the training data
xgb_model.fit(X_train, y_train)

# Predict using the trained XGBoost model
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate model performance (RMSE, MAE, R²)
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
'''
    print(f"\n📊 {model_name} Performance:")
    print(f"🔹 **RMSE**: {rmse:.4f}")
    print(f"🔹 **MAE**: {mae:.4f}")
    print(f"🔹 **R² Score**: {r2:.4f}")
'''
# Evaluate all models
evaluate_model(y_test, y_pred, "Random Forest")
evaluate_model(y_test, y_pred_gb, "Gradient Boosting")
evaluate_model(y_test, y_pred_xgb, "XGBoost")

def predict():

        # Get input from the frontend


        # Input features
        # Use example values for prediction
        location = 'Kanakapura Road'
        total_area = 1200
        num_baths = 2
        balcony = 'Yes'
        description = '2 BHK Apartment'
        price_per_sqft = 6000

        # Create a DataFrame with the same structure as the training data
        input_data = pd.DataFrame([[total_area, num_baths, price_per_sqft, location, description, balcony]],
                                  columns=['Total_Area', 'Baths', 'Price_per_SQFT', 'Location', 'Description', 'Balcony'])

        # Preprocess the input data using the same preprocessor
        input_data_preprocessed = preprocessor.transform(input_data)

        # Convert to DataFrame with correct feature names
        input_data_preprocessed_df = pd.DataFrame(input_data_preprocessed, columns=feature_names)

        # Predict using the trained model
        prediction = pipeline.predict(input_data_preprocessed_df)[0]

        print(f"Predicted Price: ₹{prediction:,.2f}")

joblib.dump(pipeline, "pipeline.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(feature_names, "feature_names.pkl")