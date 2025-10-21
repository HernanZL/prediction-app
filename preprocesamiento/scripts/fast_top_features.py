import warnings
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Change the default font to Arial
# rcParams['font.family'] = 'Arial'

# Load the preprocessed training data
data = pd.read_csv('../preprocessed-data/preprocessed_data.csv')

# Optimize memory usage
X = data.drop('remuneracion', axis=1).astype('float32')  # Features
y = data['remuneracion'].astype('float32')              # Target variable

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to DMatrix to ensure it's placed on the GPU
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Define the XGBoost model with GPU support
xgb_model = xgb.train(
    params={
        'objective': 'reg:squarederror',  # Use squared error for regression
        # 'tree_method': 'hist',           # Use GPU-compatible "hist" method
        # 'device': 'cuda',                # Explicitly use GPU
        'learning_rate': 0.05,           # Reduce learning rate for stability
        'max_depth': 15,                 # Allow deeper trees
        'subsample': 0.9,                # Use 90% of samples per tree
        'colsample_bytree': 0.9,         # Use 90% of features per tree
        'lambda': 1.0,                   # L2 regularization
        'alpha': 0.1,                    # L1 regularization
    },
    dtrain=dtrain,
    num_boost_round=500,  # Increase the number of boosting iterations
    evals=[(dval, 'validation')],
    early_stopping_rounds=10  # Stop early if no improvement
)

# Predict on the validation set
y_val_pred = xgb_model.predict(dval)
mae_val = mean_absolute_error(y_val, y_val_pred)
print(f"Mean Absolute Error (MAE) on the validation set: {mae_val}")

# Load the MinMaxScaler used during preprocessing to backscale predictions
scaler = joblib.load('../preprocessed-data/minmax_scaler.pkl')

# Reverse the scaling for MAE calculation
y_val_original = scaler.inverse_transform(y_val.values.reshape(-1, 1)).flatten()
y_val_pred_original = scaler.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()

# Compute MAE and Relative Error on the original scale
mae_original = mean_absolute_error(y_val_original, y_val_pred_original)
relative_error_original = mae_original / y_val_original.mean()
print(f"Mean Absolute Error (Original Scale): {mae_original}")
print(f"Relative Error (Original Scale): {relative_error_original * 100:.2f}%")

# Save the best model for future use
xgb_model.save_model('./advanced_json_model.json')
print("Advanced XGBoost Model saved successfully.")


warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

# Use default font and reduce font size
rcParams['font.family'] = 'sans-serif'
rcParams['axes.titlesize'] = 10
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8

# Define a function to plot feature importance
# Define a function to plot top 10 feature importance
def plot_top10_feature_importance(importance_type, title):
    # Get feature importance
    feature_importance = xgb_model.get_score(importance_type=importance_type)
    
    # Convert to DataFrame
    feature_importance_df = pd.DataFrame(
        list(feature_importance.items()), 
        columns=['Feature', 'Importance']
    ).sort_values(by='Importance', ascending=False).head(20)  # Limit to top 10
    
    # Clean feature names to avoid rendering issues
    feature_importance_df['Feature'] = feature_importance_df['Feature'].apply(
        lambda x: x.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII characters
    )
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.gca().invert_yaxis()
    
    # Reduce y-axis font size to prevent overlap
    plt.yticks(fontsize=8)
    
    plt.show()

# Plot top 10 feature importances
plot_top10_feature_importance('weight', 'Top 10 Feature Importance by Weight')
plot_top10_feature_importance('gain', 'Top 10 Feature Importance by Gain')
plot_top10_feature_importance('cover', 'Top 10 Feature Importance by Cover')

