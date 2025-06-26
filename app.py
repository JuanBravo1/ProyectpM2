from flask import Flask, request, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar modelo y escalador
modelo = joblib.load('modelo_red_neuronal.pkl')
escalador = joblib.load('scaler.pkl')

# HTML + CSS en una sola plantilla
TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Predicción de Balance</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f2f5f7;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 400px;
        }
        h2 {
            text-align: center;
            margin-bottom: 20px;
        }
        form label {
            display: block;
            margin: 10px 0 5px;
        }
        form input,
        form select {
            width: 100%;
            padding: 8px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        button {
            margin-top: 15px;
            width: 100%;
            padding: 10px;
            background-color: #008cba;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #005f7f;
        }
        .resultado {
            margin-top: 20px;
            text-align: center;
            font-size: 18px;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Predicción de Balance</h2>
        <form method="POST">
            <label>Ingreso mensual (Income)</label>
            <input type="number" name="income" step="any" required>

            <label>Límite de crédito (Limit)</label>
            <input type="number" name="limit" step="any" required>

            <label>Calificación crediticia (Rating)</label>
            <input type="number" name="rating" step="any" required>

            <label>¿Es estudiante?</label>
            <select name="student" required>
                <option value="1">Sí</option>
                <option value="0">No</option>
            </select>

            <label>Edad</label>
            <input type="number" name="age" step="any" required>

            <button type="submit">Predecir</button>
        </form>

        {% if prediccion is not none %}
            <div class="resultado">
                <strong>Balance estimado:</strong> {{ prediccion }}
            </div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    prediccion = None

    if request.method == 'POST':
        try:
            # Capturar datos del formulario
            income = float(request.form['income'])
            limit = float(request.form['limit'])
            rating = float(request.form['rating'])
            student = int(request.form['student'])
            age = float(request.form['age'])

            # Crear DataFrame
            entrada = pd.DataFrame([{
                'Income': income,
                'Limit': limit,
                'Rating': rating,
                'Student': student,
                'Age': age
            }])

            # Escalar y predecir
            entrada_scaled = escalador.transform(entrada)
            resultado = modelo.predict(entrada_scaled)
            prediccion = round(resultado[0], 2)

        except Exception as e:
            prediccion = f"Error: {str(e)}"

    return render_template_string(TEMPLATE, prediccion=prediccion)

if __name__ == '__main__':
    app.run(debug=True)
