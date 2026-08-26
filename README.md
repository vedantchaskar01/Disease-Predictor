Disease Predictor

This is a machine learning project I built for my AI and ML lab evaluation. It predicts diseases based on symptoms that the user selects.

The project contains two versions:
1. disease_predictor.py - A simple terminal based version where you answer yes or no to symptom questions.
2. app.py - A web interface built with Streamlit that lets you check boxes for your symptoms and gives you a nice visual diagnosis.

The model uses Logistic Regression from scikit-learn. It is trained on a small dataset created directly in the code, covering 8 common diseases like cold, viral fever, dengue, malaria, covid-19, allergies, migraine, and food poisoning, based on 12 different symptoms.

How to run the project:

First, install the required libraries:
pip install -r requirements.txt

To run the terminal version:
python disease_predictor.py

To run the web UI version:
streamlit run app.py

Note: This is just a simple educational project for my college lab and not for real medical advice.
