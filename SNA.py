import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load dataset (contoh)
data = pd.read_csv(r'D:\BigData\Dataset')

# Membuat graph kosong
G = nx.Graph()

# Menambahkan node dan edge berdasarkan data pengangguran
for idx, row in data.iterrows():
    pendidikan = row['jenis_pendidikan']  # Kategori pendidikan
    usia = row['kelompok_usia']  # Kategori usia
    kota = row['nama_kabupaten_kota']  # Nama kabupaten/kota
    pengangguran = row['jumlah']
    
    # Menambahkan node untuk pendidikan, usia, dan kota
    G.add_node(pendidikan, type='pendidikan', population=pengangguran)
    G.add_node(usia, type='usia')
    G.add_node(kota, type='kota')
    
    # Menambahkan edge antara kategori pendidikan, usia, dan kota
    G.add_edge(pendidikan, kota, weight=pengangguran)
    G.add_edge(usia, kota, weight=pengangguran)


# Visualisasi jaringan
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.2)  # Penataan posisi node
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
plt.title("Social Network Analysis Pengangguran di Kota Bekasi")
plt.show()

# Menganalisis Jaringan
# Misalnya menghitung derajat setiap node (berapa banyak hubungan yang dimiliki node tersebut)
degree = dict(G.degree())
print("Degree dari setiap node:", degree)

# Bisa juga melakukan analisis lebih lanjut, misalnya centrality
centrality = nx.degree_centrality(G)
print("Centrality dari setiap node:", centrality)

# Menampilkan edge dengan berat (pengangguran)
edges = G.edges(data=True)
for edge in edges:
    print(f"Edge {edge[0]} - {edge[1]} memiliki pengangguran: {edge[2]['weight']}")
