from flask import Flask, request, render_template, url_for, redirect
from src.pipeline.predict_pipeline import PredictPipeline  # Assuming CustomData handles the image data
import os
import random
import string
from werkzeug.utils import secure_filename

application = Flask(__name__)
app = application

# Set up the upload folder and allowed file extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the prediction model once when the app starts
predict_pipeline = PredictPipeline()

# Route for home page
@app.route('/')
def index():
    return render_template('home.html')  # Render home.html for the main page

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Utility to generate a unique filename to prevent overwriting files
def generate_unique_filename(filename):
    # Extract file extension
    file_ext = filename.rsplit('.', 1)[1].lower()
    # Generate a random string to append to the filename
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    return f"{random_str}.{file_ext}"

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_img():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('home.html', error="No file uploaded")  # Use home.html for errors

        file = request.files['file']

        # If no file is selected
        if file.filename == '':
            return render_template('home.html', error="No selected file")  # Use home.html for errors

        # Check if the uploaded file is of allowed type
        if file and allowed_file(file.filename):
            # Generate a unique filename to avoid conflicts
            unique_filename = generate_unique_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            try:

                # Debugging: Print file path
                print(f"File received: {file_path}")

                # Use the pre-loaded model to make a prediction
                class_name, confidence_score = predict_pipeline.predict(file_path)

                # Debugging: Print prediction results
                print(f"Prediction: {class_name}, Confidence: {confidence_score}%")

                # Returning the result to the front-end
                return render_template('home.html',  # Use home.html to return the result
                                       prediction=class_name,
                                       confidence_score=confidence_score,
                                       image_path=f'uploads/{unique_filename}')  # Path to static/uploads

            except Exception as e:
                print(f"Error: {e}")  # Debugging: Print the error in the console
                return render_template('home.html', error=str(e))  # Return error if something fails

        else:
            return render_template('home.html', error="Invalid file format. Only images are allowed.")  # Invalid file format error

    # If it's a GET request, render the home page (for the initial page load)
    return render_template('home.html')  # Render the home page to display the form

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
