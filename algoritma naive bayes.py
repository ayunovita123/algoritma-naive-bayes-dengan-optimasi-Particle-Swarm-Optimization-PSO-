# Mount google drive 
from google.colab import drive
drive.mount('/content/drive')

# Baca data dari drive
import pandas as pd
dataset = pd.read_csv("/content/drive/MyDrive/data1.csv", sep=";")

# Pisahkan feature dan target
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, -1].values

# Split data training dan testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=32)

# Standardisasi fitur
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training Naive bayes model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Prediksi data testing
y_pred = classifier.predict(X_test) 

# Evaluasi Akurasi
from sklearn.metrics import accuracy_score
print("Akurasi: ", accuracy_score(y_test, y_pred))
