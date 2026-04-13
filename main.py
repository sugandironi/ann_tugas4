from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

app = Flask(__name__)

# =============================
# Load & Training Model
# =============================
df = pd.read_csv("daftar_umkm.csv", sep=';')
df = df[['tahun', 'jumlah umkm']]
df = df.dropna()

X = df[['tahun']].values
Y = df[['jumlah umkm']].values

scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

model = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(10, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_scaled, Y_scaled, epochs=500, verbose=0)

# =============================
# Route
# =============================
@app.route('/', methods=['GET','POST'])
def index():
    hasil = None

    if request.method == 'POST':
        tahun = int(request.form['tahun'])

        tahun_scaled = scaler_X.transform([[tahun]])
        pred_scaled = model.predict(tahun_scaled)
        pred = scaler_Y.inverse_transform(pred_scaled)

        hasil = int(pred[0][0])

        # =============================
        # Buat Grafik
        # =============================
        plt.figure(figsize=(8,5))
        plt.scatter(df['tahun'], df['jumlah umkm'], label="Data Aktual")
        plt.scatter(tahun, hasil, label="Prediksi", marker='x')
        plt.xlabel("Tahun")
        plt.ylabel("Jumlah UMKM")
        plt.legend()
        plt.title("Prediksi UMKM dengan ANN")

        if not os.path.exists("static"):
            os.makedirs("static")

        plt.savefig("static/grafik.png")
        plt.close()

    return render_template("index.html", hasil=hasil)

if __name__ == "__main__":
    app.run(debug=True)