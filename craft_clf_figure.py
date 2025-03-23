from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


def craft_confusion_matrix(y_test, y_pred, save_file_name):
    """ """

    # Obtenir la liste des labels uniques
    labels = sorted(set(y_test))

    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Affichage avec Seaborn
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Prédictions")
    plt.ylabel("Vérité terrain")
    plt.title("Matrice de Confusion")
    plt.savefig(save_file_name)
    plt.close()

def craft_class_histogram(x, save_file_name):
    """ """

    # Compter les occurrences des classes
    x = list(map(str, x))  
    count_labels = Counter(x)

    # Extraire les classes et leurs fréquences
    classes = list(count_labels.keys())
    frequences = list(count_labels.values())

    # Tracer l'histogramme
    plt.figure(figsize=(6,4))
    plt.bar(classes, frequences, color="royalblue")

    # Labels et titre
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'occurrences")
    plt.title("Distribution des classes")
    plt.xticks(rotation=45)
    plt.savefig(save_file_name)
    plt.close()






if __name__ == "__main__":

    print("Tardis")
    x = [21,21,3,5,5,4,4,4,4]
    craft_class_histogram(x, "test.png")    
