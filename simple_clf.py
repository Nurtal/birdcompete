import glob
import extract_features
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def run_svm_clf():
    """
    Simple exemple case, extract features from scats and used it to train a SVM
    """

    # params
    J = 6 
    Q = 8
    duration = 16000

    # load data
    file_list_1 = glob.glob("data/126247/*.ogg")
    file_list_2 = glob.glob("data/21211/*.ogg")

    X = []
    y = []
    for fl in file_list_1:
        x = extract_features.extract_features(fl, J, Q, duration)
        features = x.numpy().flatten()
        X.append(features)
        y.append("Poison_Frog")
    for fl in file_list_2:
        x = extract_features.extract_features(fl, J, Q, duration)
        features = x.numpy().flatten()
        X.append(features)
        y.append("Foam-nest_Frog")
    
    # Division en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle SVM
    clf = SVC(kernel="linear", C=1.0)
    clf.fit(X_train, y_train)

    # Prédiction et évaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[CLF][SVM] ACC : {accuracy * 100:.2f}%")
    

    





if __name__ == "__main__":


    run_svm_clf()
