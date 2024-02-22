# Install library yang diperlukan
!pip install pyswarm
!pip install scikit-learn

# Import library yang diperlukan
import pandas as pd
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from pyswarm import pso

# Mount Google Drive
drive.mount('/content/drive')

# Baca dataset dari Google Drive
dataset_path = '/content/drive/My Drive/data1.csv'
dataset = pd.read_csv(dataset_path, sep=';')

# Hapus duplikat
dataset.drop_duplicates(inplace=True)

# Pisahkan fitur dan label
X = dataset.drop(columns=['output'])
y = dataset['output']

# Bagi dataset menjadi data latih dan data uji (90:10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=32)


# Fungsi objektif untuk PSO
def objective_function(params):
    alpha, beta = params
    naive_bayes = GaussianNB(var_smoothing=alpha)
    naive_bayes.fit(X_train, y_train)
    y_pred = naive_bayes.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return -accuracy  # Negatif karena kita ingin memaksimalkan akurasi

# Batasan untuk parameter PSO
lb = [1e-10, 1e-10]  # Lower bound untuk alpha dan beta
ub = [1e-3, 1e-3]    # Upper bound untuk alpha dan beta

# Lakukan optimasi PSO
best_params, _ = pso(objective_function, lb, ub)

# Cetak parameter terbaik yang ditemukan oleh PSO
print("Parameter terbaik (alpha, beta):", best_params)

# Buat model Naive Bayes dengan parameter terbaik
alpha, beta = best_params
naive_bayes = GaussianNB(var_smoothing=alpha)
naive_bayes.fit(X_train, y_train)

# Lakukan prediksi
y_pred = naive_bayes.predict(X_test)

# Cetak prediksi
print("Prediksi kelas untuk data uji:")
print(y_pred)

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi Naive Bayes setelah optimasi:", accuracy)

from sklearn.metrics import recall_score, precision_score

# Lakukan prediksi menggunakan model
y_pred = naive_bayes.predict(X_test)

# Hitung recall dan precision
recall = recall_score(y_test, y_pred) # Recall
precision = precision_score(y_test, y_pred) # Precision

# Cetak hasil recall dan precision
print("Recall:", recall)
print("Precision:", precision)
