from flask import Flask, render_template, request
from joblib import load

# Load the saved model
try:
    model = load('random_forest_model.joblib')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialize the Flask app
app = Flask(__name__)

def validate_input(value, min_val, max_val):
    """Validate if input is within acceptable range"""
    try:
        val = int(value)
        return min_val <= val <= max_val
    except ValueError:
        return False

def get_input_constraints():
    """Define acceptable ranges for each input"""
    return {
        'Sex': (0, 1),
        'Age': (1, 120),
        'Height': (0, 350),
        'Overweight_Obese_Family': (1, 2),
        'Consumption_of_Fast_Food': (1, 2),
        'Frequency_of_Consuming_Vegetables': (1, 3),
        'Number_of_Main_Meals_Daily': (1, 3),
        'Food_Intake_Between_Meals': (1, 4),
        'Smoking': (1, 2),
        'Liquid_Intake_Daily': (1, 3),
        'Calculation_of_Calorie_Intake': (1, 2),
        'Physical_Excercise': (1, 5),
        'Schedule_Dedicated_to_Technology': (1, 3),
        'Type_of_Transportation_Used': (1, 5)
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    
    if request.method == 'POST':
        try:
            # Get input constraints
            constraints = get_input_constraints()
            
            # Validate and collect features
            features = []
            for field, (min_val, max_val) in constraints.items():
                value = request.form.get(field)
                if not value or not validate_input(value, min_val, max_val):
                    raise ValueError(f"Invalid input for {field}. Must be between {min_val} and {max_val}.")
                features.append(int(value))
            
            # Check if model is loaded
            if model is None:
                raise Exception("Model not loaded. Please check the model file.")
            
            # Make prediction
            prediction = int(model.predict([features])[0]) +1 # Convert to int to avoid numpy type issues
            
            # Map prediction to class description
            class_descriptions = {
                0: "Underweight",
                1: "Normal Weight",
                2: "Overweight",
                3: "Obesity"
            }
            
            prediction_text = f"{prediction} - {class_descriptions.get(prediction, 'Unknown')}"
            
        except ValueError as ve:
            error = str(ve)
        except Exception as e:
            error = f"An error occurred: {str(e)}"
    
    return render_template('index.html', prediction=prediction_text if prediction is not None else None, error=error)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html', error="Internal server error"), 500

if __name__ == '__main__':
    app.run(debug=True)