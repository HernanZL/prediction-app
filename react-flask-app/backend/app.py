from flask import Flask, request, jsonify, render_template
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Cargar modelo y scaler
model = xgb.Booster()
model.load_model('./model/advanced_json_model.json')
scaler = joblib.load('./model/minmax_scaler.pkl')
train_columns = pd.read_csv('./model/preprocessed_data.csv').drop('remuneracion', axis=1).columns

# Dataset base para obtener categorías
raw_data_path = './dataset/2020.csv'
raw_data = pd.read_csv(raw_data_path, delimiter=';')

categorical_columns = [
    'rango_edad', 'regimen_laboral', 'nivel_educativo', 'regimen_salud',
    'tamaão_empresa', 'sexo', 'departamento', 'actividad_economica',
    'regimen_pension', 'ocupacion'
]

# Pipeline para categorías
categorical_imputer = SimpleImputer(strategy='most_frequent')
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline([
            ('imputer', categorical_imputer),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_columns)
    ]
)
preprocessor.fit(raw_data[categorical_columns])

def handle_no_determinado(column):
    if column.dtype == 'object' or column.dtype.name == 'category':
        return column.fillna('NO DETERMINADO').replace('NO DETERMINADO', 'NoDeterminado')
    return column.fillna(column.median())

# ✅ Ruta principal: muestra formulario y procesa el envío
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Recibir datos del formulario
        user_input = {col: request.form.get(col) for col in categorical_columns}
        input_df = pd.DataFrame([user_input])

        for col in input_df.columns:
            input_df[col] = handle_no_determinado(input_df[col])

        preprocessed_input = preprocessor.transform(input_df)
        ohe_columns = preprocessor.transformers_[0][1].named_steps['onehot'].get_feature_names_out(categorical_columns)
        aligned_df = pd.DataFrame(preprocessed_input, columns=ohe_columns)

        # Alinear columnas con las del entrenamiento
        missing_columns = [col for col in train_columns if col not in aligned_df.columns]
        for col in missing_columns:
            aligned_df[col] = 0
        aligned_df = aligned_df[train_columns]

        # Predicción
        dmatrix = xgb.DMatrix(aligned_df)
        predictions = model.predict(dmatrix)
        predictions_original_scale = scaler.inverse_transform(predictions.reshape(-1, 1))
        prediction = round(float(predictions_original_scale.flatten()[0]), 2)

    return render_template('index.html', prediction=prediction)

# (Mantener /predict por compatibilidad con API JSON)
@app.route('/predict', methods=['POST'])
def predict_json():
    user_input = request.get_json()
    input_df = pd.DataFrame([user_input])
    for col in input_df.columns:
        input_df[col] = handle_no_determinado(input_df[col])
    preprocessed_input = preprocessor.transform(input_df)
    ohe_columns = preprocessor.transformers_[0][1].named_steps['onehot'].get_feature_names_out(categorical_columns)
    aligned_df = pd.DataFrame(preprocessed_input, columns=ohe_columns)
    missing_columns = [col for col in train_columns if col not in aligned_df.columns]
    for col in missing_columns:
        aligned_df[col] = 0
    aligned_df = aligned_df[train_columns]
    dmatrix = xgb.DMatrix(aligned_df)
    predictions = model.predict(dmatrix)
    predictions_original_scale = scaler.inverse_transform(predictions.reshape(-1, 1))
    return jsonify({"prediction": float(predictions_original_scale.flatten()[0])})

@app.route('/get-options', methods=['GET'])
def get_options():
    relevant_columns = [
        "rango_edad", "regimen_laboral", "nivel_educativo", "regimen_salud",
        "tamaão_empresa", "sexo", "departamento", "actividad_economica",
        "regimen_pension", "ocupacion"
    ]

    options_per_column = {}
    for col in relevant_columns:
        # Reemplaza valores faltantes y convierte a texto
        options = raw_data[col].fillna('NoDeterminado').unique().tolist()
        options_per_column[col] = sorted([str(option) for option in options])

    return jsonify(options_per_column)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

