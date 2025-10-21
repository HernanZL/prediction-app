import React, { useState, useEffect } from "react";

interface Options {
  [key: string]: string[];
}

interface FormData {
  rango_edad: string;
  regimen_laboral: string;
  nivel_educativo: string;
  regimen_salud: string;
  tamaão_empresa: string;
  sexo: string;
  departamento: string;
  actividad_economica: string;
  regimen_pension: string;
  ocupacion: string;
}

const App: React.FC = () => {
  const [formData, setFormData] = useState<FormData>({
    rango_edad: "",
    regimen_laboral: "",
    nivel_educativo: "",
    regimen_salud: "",
    tamaão_empresa: "",
    sexo: "",
    departamento: "",
    actividad_economica: "",
    regimen_pension: "",
    ocupacion: "",
  });

  const [options, setOptions] = useState<Options | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const response = await fetch("http://localhost:5000/get-options");
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data: Options = await response.json();
        setOptions(data);
      } catch (err) {
        console.error("Error fetching options:", err);
        setError("No se pudo cargar las opciones. Inténtelo de nuevo más tarde.");
      }
    };

    fetchOptions();
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setPrediction(null);
    setLoading(true);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        setPrediction(data.prediction);
      }
    } catch (err) {
      setError("Error al conectar con el servidor.");
    } finally {
      setLoading(false);
    }
  };

  if (!options) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100">
        <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="flex flex-col md:flex-row min-h-screen w-full bg-gray-100">
      {/* Left Side: Form */}
      <div className="md:w-2/3 w-full bg-white shadow-lg rounded-lg p-12 flex flex-col justify-between">
        <div className="space-y-12">
          <h1 className="text-5xl font-bold text-gray-700 text-center">
            Calculadora de Sueldos
          </h1>
          <p className="text-lg text-gray-600 text-center">
            ¡Descubre cuánto podrías ganar! Ingresa tus datos y obtén una predicción aproximada de tu sueldo en soles (PEN), basado en tus características laborales y demográficas.
          </p>
          <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.keys(formData).map((field) => (
              <div key={field}>
                <label className="block text-xl text-gray-700 font-medium mb-3">
                  {field.replace("_", " ").toUpperCase()}
                </label>
                <select
                  name={field}
                  value={formData[field as keyof FormData]}
                  onChange={handleChange}
                  className="w-full border-gray-300 rounded-lg p-4 text-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">(Opcional) Seleccione</option>
                  {options[field]?.map((option) => (
                    <option key={option} value={option} title={option}>
                      {option.length > 50 ? option.slice(0, 50) + "..." : option}
                    </option>
                  ))}
                </select>
              </div>
            ))}

            <div className="md:col-span-2">
              <button
                type="submit"
                disabled={loading}
                className={`w-full bg-gradient-to-r from-gray-500 to-gray-700 text-white font-bold mt-16 py-6 px-8 text-xl rounded-lg ${
                  loading
                    ? "opacity-50 cursor-not-allowed"
                    : "hover:from-gray-600 hover:to-gray-800"
                }`}
              >
                {loading ? (
                  <span className="flex items-center justify-center">
                    <span className="animate-spin h-6 w-6 mr-3 border-t-2 border-white"></span>
                    Procesando...
                  </span>
                ) : (
                  "Calcular"
                )}
              </button>
            </div>
          </form>
        </div>
        {error && <p className="text-red-500 mt-4 text-center">{error}</p>}
      </div>

{/* Right Side: Prediction */}
<div className="md:w-1/3 bg-gray-800 text-white flex flex-col rounded-lg m-8 p-6">
  <h2 className="text-4xl font-bold mb-8 text-center">SUELDO</h2>
  <div className="flex-grow flex flex-col items-center justify-center">
    {prediction !== null ? (
      <div className="text-center">
        <p className="text-6xl font-extrabold mb-4">{prediction.toFixed(2)} PEN</p>
        <p className="text-6xl font-medium text-gray-300 my-4">
          ${(prediction / 3.7).toFixed(2)} USD
        </p>
      </div>
    ) : (
      <p className="text-xl text-center">
        Tu predicción aparecerá aqui al llenar el formulario.
      </p>
    )}
  </div>
</div>
    </div>
  );
};

export default App;

