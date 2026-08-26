import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import time

st.set_page_config(page_title="Disease Predictor", layout="centered")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    .main .block-container {
        background-color: white;
        padding: 3rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }

    h1 {
        color: #2c3e50;
        text-align: center;
        font-weight: 700;
        letter-spacing: -1px;
        padding-bottom: 10px;
    }

    div[data-testid="stCheckbox"] label span {
        font-size: 1.1rem;
        color: #34495e;
    }

    .stButton>button { 
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        color: white; 
        border: none;
        border-radius: 50px; 
        width: 100%; 
        font-size: 1.2rem; 
        font-weight: bold; 
        padding: 0.8rem;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stButton>button:hover { 
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.6);
        color: white;
    }

    .stButton>button:active {
        transform: translateY(1px);
    }

    .pred-box { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px; 
        border-radius: 15px; 
        text-align: center;
        margin-top: 30px; 
        box-shadow: 0 10px 20px rgba(118, 75, 162, 0.3);
        animation: fadeIn 0.5s ease-out;
    }

    .pred-box h3 {
        color: rgba(255,255,255,0.9);
        margin: 0;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .pred-box p {
        color: white;
        font-size: 2.2rem;
        font-weight: bold;
        margin: 10px 0 0 0;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    data = {
        'fever':        [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        'cough':        [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        'nausea':       [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        'stomach_pain': [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'runny_nose':   [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
        'headache':     [0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        'fatigue':      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'loss_of_taste':[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'chills':       [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        'joint_pain':   [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
        'rash':         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'sneezing':     [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
        'disease':      [
            'Cold', 'Viral Fever', 'Food Poisoning', 'COVID-19', 'Malaria', 'Dengue', 'Allergies', 
            'Cold', 'Food Poisoning', 'COVID-19', 'Malaria', 'Migraine', 'Allergies', 'Malaria', 'Dengue', 
            'COVID-19', 'Allergies', 'Cold', 'Migraine'
        ]
    }
    
    df = pd.DataFrame(data)
    x = df.drop('disease', axis=1)
    y = df['disease']
    
    model = LogisticRegression(max_iter=200)
    model.fit(x, y)
    return model

clf = get_model()

st.title("Disease Predictor")
st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 1.1rem; margin-bottom: 2rem;'>Select your symptoms below to receive an instant AI diagnosis.</p>", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    fever = st.checkbox("Fever")
    cough = st.checkbox("Cough")
    fatigue = st.checkbox("Fatigue")
    chills = st.checkbox("Chills")
    headache = st.checkbox("Headache")
    joint_pain = st.checkbox("Joint Pain")

with c2:
    loss_of_taste = st.checkbox("Loss of Taste")
    nausea = st.checkbox("Nausea")
    stomach_pain = st.checkbox("Stomach Pain")
    runny_nose = st.checkbox("Runny Nose")
    sneezing = st.checkbox("Sneezing")
    rash = st.checkbox("Rash")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Predict"):
    with st.spinner("Analyzing..."):
        time.sleep(1) 
        
        user_input = [
            int(fever), int(cough), int(nausea), int(stomach_pain), 
            int(runny_nose), int(headache), int(fatigue), int(loss_of_taste),
            int(chills), int(joint_pain), int(rash), int(sneezing)
        ]
        
        res = clf.predict([user_input])[0]
        
        st.markdown(f"""
        <div class="pred-box">
            <h3>Diagnosis Result</h3>
            <p>{res}</p>
        </div>
        """, unsafe_allow_html=True)
