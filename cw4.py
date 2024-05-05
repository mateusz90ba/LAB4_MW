import numpy as np
from keras import Sequential
from keras._tf_keras.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

file = pd.read_csv('wynik.csv', sep=',')
data = file.to_numpy() #zamiana na tablicę numpy

X = data[:,:-1].astype('float') #wyodrębnienie wektorów cech do macierzy X
y = data[:,-1]                  #wyodrębnienie etykiety kategorii do osobnej kolumny (wektora Y)

label_encoder = LabelEncoder() #obiekt, zamieniający etykiety nominalne na intigery
integer_encoded = label_encoder.fit_transform(y)
#binarne enkodowanie
onehot_encoder = OneHotEncoder(sparse_output= False) #wykonanie kodowania 1 z n dla wektora
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded) #tutaj jest już kategoria, klasa, a nie słowa i jest zakodowana binarnie

X_train, X_test, y_train, y_test = train_test_split(X, onehot_encoded, test_size=0.3) #ustawienie części testowej na 30% (dlatego test_size=0.3), reszta idzie do treningu

#Tworzenie modelu sieci neuronowej
#Można dzięki bibliotece keras stworzyć modele sieci za pomocą dwóch ścieżek: sekwencyjna albo funkcjonalna, w tym zadaniu wykorzystano sekwencyjną
model = Sequential() #utworzenie obiektu sieci typu Sequential
model.add(Dense(10, input_dim=72, activation='sigmoid')) #stworzenie i ustawienie pierwszej warstwy (liczby perceptronów na 10, funkcja aktywacji sigmoid i wymiar wyjściowy na 72)
model.add(Dense(3, activation='softmax')) #stworzenie i ustawienie drugiej warstwy
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy']) #kompilacja i zbudowanie modelu z funkcją categorical_crossentropy i algorytmem optymalizacji sgd (stochastic gradient descent)
model.summary() #wypisze podsumowanie sieci na ekran
model.fit(X_train, y_train, epochs = 100, batch_size = 10, shuffle = True) #X_train i y_train to zbiór treningowy, epochs to liczba iteracji, ile razy przejść przez zbiór danych, shuffle - powoduje wymieszanie wektorów danych, co jest bardzo ważne, bo tak to sieć nie będzie się uczyła jednej klasy (tekstury np gres)

#Tu jest przejście z reprezentacji binarnej (onehot_encoder) na intigera
y_pred = model.predict(X_test)
y_pred_int = np.argmax(y_pred, axis=1)
y_test_int = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test_int, y_pred_int)
print(cm)



