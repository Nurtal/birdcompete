import pandas as pd
import os
import extract_features
import craft_clf_figure
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from pydub import AudioSegment

def train_classic_clf(label_file:str, result_folder:str):
    """ """

    # params
    J = 6 
    Q = 8

    # init result folder
    if os.path.isdir(result_folder):
        shutil.rmtree(result_folder)
    os.mkdir(result_folder)

    # load manifest
    df = pd.read_csv(label_file)

    # get duration
    test_file = list(df['ID'])[0]
    audio = AudioSegment.from_ogg(test_file)
    duration = len(audio)  

    # load training dataset
    X = []
    y = []
    for index, row in df.iterrows():

        # extract infos
        audio_file_path = row['ID']
        label = row['LABEL']

        # load audio
        x = extract_features.extract_features(audio_file_path, J, Q, duration)
        features = x.numpy().flatten()
        X.append(features)

        # load label
        y.append(label)
    
    # Division en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle SVM
    clf = SVC(kernel="linear", C=1.0)
    clf.fit(X_train, y_train)

    # Prédiction et évaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[CLF] ACC : {accuracy * 100:.2f}%")

    # craft confusion matrix
    craft_clf_figure.craft_confusion_matrix(y_test, y_pred, f"{result_folder}/confusion_matrix.png")

    # craft histogram repartition in train
    craft_clf_figure.craft_class_histogram(y_train, f"{result_folder}/train_class_distribution.png")

    # craft histogram repartition in test
    craft_clf_figure.craft_class_histogram(y_test, f"{result_folder}/test_class_distribution.png")

    # save prediction
    df = pd.DataFrame({"TRUE_LABEL": y_test, "PREDICTED_LABEL": y_pred})
    df.to_csv(f"{result_folder}/predicted_labels.csv", index=False)


    

if __name__ == "__main__":

    train_classic_clf("data/small_labels.csv", "/tmp/zogzog")
    
