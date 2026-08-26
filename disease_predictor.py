import pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

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

x_train = df.drop('disease', axis=1)
y_train = df['disease']

clf = LogisticRegression(max_iter=200)
clf.fit(x_train, y_train)

def get_diagnosis():
    print("--- Disease Predictor ---")
    print("Answer with 1 for Yes, 0 for No\n")
    
    symps = [
        'fever', 'cough', 'nausea', 'stomach_pain', 'runny_nose',
        'headache', 'fatigue', 'loss_of_taste', 'chills', 'joint_pain',
        'rash', 'sneezing'
    ]
    user_data = []
    
    for s in symps:
        while True:
            val = input(f"Do you have {s.replace('_', ' ')}? (1/0): ")
            if val in ['0', '1']:
                user_data.append(int(val))
                break
            else:
                print("please enter 1 or 0")
                
    pred = clf.predict([user_data])[0]
    
    print("\nResult:")
    print(f"Predicted disease: {pred}")

if __name__ == "__main__":
    get_diagnosis()
