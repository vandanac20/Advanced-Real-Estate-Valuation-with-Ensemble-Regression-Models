# Advanced-Real-Estate-Valuation-with-Ensemble-Regression-Models
This is a real estate price prediction app that uses machine learning models to predict property prices based on input features such as location, area, number of bathrooms, price per square foot, and other details. The app leverages various models including Random Forest, Gradient Boosting, and XGBoost to provide accurate price predictions for a given property.

# Features
- Real Estate Price Prediction: Predict the price of a property based on features like location, total area, number of bathrooms, and price per square foot.
- Streamlit Interface: A simple and intuitive web interface built using Streamlit for easy interaction.
- Model Integration: Models trained with real estate data and saved for future predictions.

# Technologies Used
- Streamlit: For building the web application interface.
- Scikit-learn: For machine learning models and preprocessing.
- XGBoost: For enhanced gradient boosting model.
- Pandas & Numpy: For data manipulation and analysis.
- Joblib: For saving and loading trained models.
- Matplotlib: For visualizations (if used in the data processing phase).

# Requirements
To run this app, you need to install the following dependencies:

Python 
Streamlit
Pandas
Numpy
Scikit-learn
XGBoost
Matplotlib
Joblib

You can install the necessary packages using pip:
pip install streamlit pandas numpy scikit-learn xgboost matplotlib joblib

# How to Run the Application
Clone the Repository
Run the Streamlit App:
- streamlit run index.py
- This will start the Streamlit app and open it in your default web browser.

# How It Works
Data Preprocessing: The real estate data is cleaned and preprocessed, including handling missing values, scaling numerical features, encoding categorical features, and performing dimensionality reduction using PCA (if necessary).

# Model Training:
Various regression models (Random Forest, Gradient Boosting, XGBoost) are trained using the cleaned and preprocessed data. The best model is chosen for making predictions.

# Prediction: 
Users can input the details of a property (location, area, number of bathrooms, price per square foot, etc.) and the trained model will predict the price.

# Model Performance
The models have been evaluated using standard regression metrics like:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score (R-Squared)

Each model's performance is displayed in the terminal when the prediction is made.

# Model Training and Prediction
The model training process is handled in main.py, where we load the dataset, preprocess the data, and train the model using various algorithms.
The trained model and preprocessing steps are saved using joblib into .pkl files, which are later loaded in index.py for predictions in the Streamlit app.

# Snapshot
![Model 1](https://github.com/user-attachments/assets/67f0f87e-351a-4801-9e6d-b2283089c1d2)

# Contributions
Feel free to fork this repository and submit pull requests if you find improvements or bug fixes.
