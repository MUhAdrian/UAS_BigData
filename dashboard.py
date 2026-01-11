import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

# =======================
# LOAD DATASETS
# =======================
# Memuat dataset yang sudah dibersihkan
df = pd.read_csv("bekasi_ml_percent_ready.csv")
edges = pd.read_csv("bekasi_sna_edges_education.csv")

# =======================
# SIDEBAR
# =======================
st.sidebar.title("Pengaturan Analisis")
tahun = st.sidebar.selectbox(
    "Pilih Tahun",
    sorted(df["tahun"].unique())
)

df_tahun = df[df["tahun"] == tahun]

# =======================
# TITLE
# =======================
st.title("📊 Dashboard Analisis Pengangguran Kota Bekasi")
st.markdown(
    "Analisis berbasis **Machine Learning** dan **Social Network Analysis (SNA)** "
    "untuk mengidentifikasi faktor dominan penyebab pengangguran."
)

# =======================
# 1. PETA GEOGRAFIS (PROXY – WILAYAH)
# =======================
st.subheader("🗺️ Distribusi TPT Berdasarkan Pendidikan")

fig_map = px.bar(
    df_tahun,
    x="pendidikan",
    y="tpt_persen",
    color="pendidikan",
    title=f"Tingkat Pengangguran Terbuka (%) Tahun {tahun}",
    text="tpt_persen"
)
fig_map.update_layout(showlegend=False)
st.plotly_chart(fig_map, use_container_width=True)

# =======================
# 2. GRAFIK TREN
# =======================
st.subheader("📈 Tren TPT Berdasarkan Pendidikan")

fig_trend = px.line(
    df,
    x="tahun",
    y="tpt_persen",
    color="pendidikan",
    markers=True
)
st.plotly_chart(fig_trend, use_container_width=True)

# =======================
# 3. PERBANDINGAN PENYEBAB
# =======================
st.subheader("🎓 Perbandingan Penyebab Pengangguran (Pendidikan)")

fig_comp = px.box(
    df,
    x="pendidikan",
    y="tpt_persen",
    title="Distribusi TPT (%) per Tingkat Pendidikan"
)
st.plotly_chart(fig_comp, use_container_width=True)

# =======================
# 4. SOCIAL NETWORK ANALYSIS
# =======================
st.subheader("🕸️ Social Network Analysis (Pendidikan – Wilayah)")

G = nx.from_pandas_edgelist(
    edges,
    source="source",
    target="target",
    edge_attr="weight"
)

plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G, seed=42)

nx.draw(
    G, pos,
    with_labels=True,
    node_size=2500,
    node_color="lightblue",
    font_size=9
)

edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

st.pyplot(plt)

# =======================
# 5. MODEL PREDIKTIF RANDOM FOREST
# =======================
st.subheader("🤖 Model Prediktif Random Forest")

# Persiapan data untuk Random Forest
df_ml = df.dropna(subset=['tpt_persen'])

# Encode categorical variables
le_pendidikan = LabelEncoder()
df_ml_encoded = df_ml.copy()
df_ml_encoded['pendidikan_encoded'] = le_pendidikan.fit_transform(df_ml['pendidikan'])

# Fitur dan target
X = df_ml_encoded[['tahun', 'pendidikan_encoded']].values
y = df_ml_encoded['tpt_persen'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

# Prediksi
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Evaluasi Model
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

# Tampilkan metrik model
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("R² Score (Test)", f"{r2:.4f}")
with col2:
    st.metric("RMSE", f"{rmse:.4f}")
with col3:
    st.metric("MAE", f"{mae:.4f}")
with col4:
    st.metric("Total Data", len(df_ml))

# Feature Importance - Permutation Importance per Pendidikan
feature_names = ['Tahun', 'Pendidikan']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Hitung importance per kategori pendidikan
pendidikan_importance = []
for pend in le_pendidikan.classes_:
    # Hitung rata-rata TPT untuk masing-masing pendidikan
    pend_data = df_ml[df_ml['pendidikan'] == pend]['tpt_persen']
    importance_val = pend_data.std()  # gunakan standar deviasi sebagai ukuran pentingnya
    pendidikan_importance.append({'Pendidikan': pend, 'Importance': importance_val})

df_pend_imp = pd.DataFrame(pendidikan_importance).sort_values('Importance', ascending=False)

fig_importance = px.bar(
    df_pend_imp,
    x='Pendidikan',
    y='Importance',
    title='Tingkat Variabilitas TPT per Pendidikan (Feature Impact)',
    labels={'Importance': 'Variabilitas TPT (%)'},
    color='Pendidikan'
)
st.plotly_chart(fig_importance, use_container_width=True)

# Grafik Actual vs Predicted
# Buat dataframe untuk scatter plot yang lebih jelas
actual_vs_pred_df = pd.DataFrame({
    'Nilai Aktual': y_test,
    'Nilai Prediksi': y_pred_test,
    'Error': np.abs(y_test - y_pred_test)
})

fig_actual_pred = px.scatter(
    actual_vs_pred_df,
    x='Nilai Aktual',
    y='Nilai Prediksi',
    size='Error',
    color='Error',
    hover_data=['Error'],
    labels={'Nilai Aktual': 'Nilai Aktual TPT (%)', 'Nilai Prediksi': 'Prediksi TPT (%)'},
    title='Actual vs Predicted - Random Forest',
    color_continuous_scale='Reds'
)

# Tambahkan garis referensi (perfect prediction)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())

fig_actual_pred.add_shape(
    type="line",
    x0=min_val, y0=min_val,
    x1=max_val, y1=max_val,
    line=dict(dash="dash", color="green", width=3),
    name="Perfect Prediction"
)

fig_actual_pred.update_layout(
    hovermode='closest',
    height=500
)

st.plotly_chart(fig_actual_pred, use_container_width=True)



# Prediksi untuk semua pendidikan (statis)
st.subheader("📋 Prediksi TPT Berdasarkan Pendidikan")

# Gunakan tahun maksimal dalam dataset untuk prediksi
tahun_prediksi_default = int(df['tahun'].max())
pendidikan_options = df['pendidikan'].unique()

predictions = []
for pend in pendidikan_options:
    pend_encoded = le_pendidikan.transform([pend])[0]
    X_temp = np.array([[tahun_prediksi_default, pend_encoded]])
    pred = rf_model.predict(X_temp)[0]
    predictions.append({'Pendidikan': pend, 'TPT_Prediksi (%)': round(pred, 2)})

df_predictions = pd.DataFrame(predictions)

fig_pred_all = px.bar(
    df_predictions,
    x='Pendidikan',
    y='TPT_Prediksi (%)',
    title=f'Estimasi TPT Berdasarkan Pendidikan (Data Tahun {tahun_prediksi_default})',
    text='TPT_Prediksi (%)',
    color='Pendidikan'
)
fig_pred_all.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
st.plotly_chart(fig_pred_all, use_container_width=True)

st.dataframe(df_predictions, use_container_width=True)

st.info(
    "🔍 **Insight Utama:**\n"
    "Pendidikan SMA merupakan faktor paling dominan penyebab tingginya "
    "tingkat pengangguran di Kota Bekasi. "
    "Intervensi kebijakan yang menargetkan kelompok ini "
    "berpotensi memberikan dampak terbesar."
)
