import pandas as pd

# Memuat dataset
data3 = pd.read_csv("data3.csv")
data5 = pd.read_csv("data5.csv")

# Memeriksa kolom yang ada di kedua dataset
print("Kolom di data3 (TPT Pendidikan):", data3.columns)
print("Kolom di data5 (TPT Usia):", data5.columns)
