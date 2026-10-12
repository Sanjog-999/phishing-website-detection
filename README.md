# Phishing Website Detection

Machine Learning (Random Forest) aur Flask se bana phishing URL detector.

## Features
- URL se 31 lexical features nikalta hai
- Result ke saath phishing probability % dikhata hai
- Flask web interface aur JSON API (/api/predict)

## How to Run
1. Install the required dependencies:
   `pip install -r requirements.txt`
2. Download the PhiUSIIL Phishing URL Dataset from UCI and place the CSV file in this folder.
3. Train the model:
   `python train_model.py dataset.csv`
4. Run the application:
   `python app.py`
5. Open the following URL in your browser:
   `http://127.0.0.1:5000`

## Tech Stack
Python, Flask, Scikit-learn, Pandas, HTML/CSS
