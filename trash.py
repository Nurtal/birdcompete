import matplotlib.pyplot as plt
from collections import Counter

# Exemple y_test (remplace avec tes données réelles)
y_test = ["chien", "chat", "chien", "oiseau", "chat", "chien", "oiseau", "oiseau", "chat", "chien"]

# Compter les occurrences des classes
count_labels = Counter(y_test)

# Extraire les classes et leurs fréquences
classes = list(count_labels.keys())
frequences = list(count_labels.values())

# Tracer l'histogramme
plt.figure(figsize=(6,4))
plt.bar(classes, frequences, color="royalblue")

# Labels et titre
plt.xlabel("Classe")
plt.ylabel("Nombre d'occurrences")
plt.title("Distribution des classes dans y_test")
plt.xticks(rotation=45)  # Rotation si nécessaire
plt.show()
