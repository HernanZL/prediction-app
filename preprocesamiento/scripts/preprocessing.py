
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
dataset_path = '../datasets/EmpleosSectorPrivado.csv'  # Replace with your file path
dataset = pd.read_csv(dataset_path, delimiter=';')

# Columns to use for processing
selected_columns = [
    'rango_edad', 'regimen_laboral', 'nivel_educativo', 'regimen_salud',
    'tamaão_empresa', 'sexo', 'departamento', 'actividad_economica', 
    'regimen_pension', 'ocupacion', 'remuneracion'
]
data = dataset[selected_columns]

# Handle 'NO DETERMINADO' values by treating them as a separate category
def handle_no_determinado(column):
    if column.dtype == 'object' or column.dtype.name == 'category':
        return column.fillna('NO DETERMINADO').replace('NO DETERMINADO', 'NoDeterminado')
    return column.fillna(column.median())  # Fill numerical columns with median

# Apply the 'NO DETERMINADO' handling to all categorical columns
for col in data.columns:
    data[col] = handle_no_determinado(data[col])

# Separate features (X) and target (y)
X = data.drop(columns=['remuneracion'])  # All columns except 'remuneracion'
y = data['remuneracion']  # Target variable

# Ensure that 'remuneracion' is not in X before applying transformations
print("Columns in X:", X.columns)

# Define categorical columns
categorical_columns = [
    'rango_edad', 'regimen_laboral', 'nivel_educativo', 'regimen_salud', 
    'tamaão_empresa', 'sexo', 'departamento', 'actividad_economica', 
    'regimen_pension', 'ocupacion'
]

# Impute missing values in categorical columns with the most frequent category
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Preprocessor for categorical data: Imputation + OneHotEncoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline([
            ('imputer', categorical_imputer),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # OneHot for nominal
        ]), categorical_columns)
    ])

# Apply transformations to the features (X) only
X_preprocessed = preprocessor.fit_transform(X)

# MinMaxScaler normalization of the target variable (remuneracion) to ensure consistency
scaler = MinMaxScaler()
y_normalized = scaler.fit_transform(y.values.reshape(-1, 1))  # Reshape for MinMaxScaler

# Get the feature names after one-hot encoding
cat_columns_transformed = preprocessor.transformers_[0][1].named_steps['onehot'].get_feature_names_out(categorical_columns)

# Combine the transformed features (numerical + one-hot encoded categorical) into one DataFrame
X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=cat_columns_transformed)

# Add the normalized target variable (remuneracion) to the final DataFrame
X_preprocessed_df['remuneracion'] = y_normalized  # Add normalized target variable to features

# Save the preprocessed data to a new CSV (with feature names)
X_preprocessed_df.to_csv('../preprocessed-data/preprocessed_data.csv', index=False, encoding='utf-8')

# Save the scaler for future use (to inverse transform on predictions)
joblib.dump(scaler, 'minmax_scaler.pkl')

print("Preprocessing complete and saved to 'train_preprocesada3.csv'. MinMaxScaler saved as 'minmax_scaler.pkl'.")

