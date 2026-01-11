import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="ML Analysis", layout="wide")
st.title("🤖 Machine Learning Analysis - TPT Bekasi")
st.markdown(
    "Prediksi dan analisis **Tingkat Pengangguran Terbuka (TPT)** menggunakan "
    "**Random Forest** berdasarkan faktor Pendidikan dan Usia."
)

# LOAD DATA
data3 = pd.read_csv("data3.csv")
data5 = pd.read_csv("data5.csv")

# PREPROCESSING: Bersihkan data (hapus satuan/id, whitespace, filter nilai 0)
data3 = data3.drop(columns=["satuan", "id"])
data3["jenis_pendidikan"] = data3["jenis_pendidikan"].str.replace(r'\s+', ' ', regex=True).str.strip()
data3 = data3[data3["jumlah"] > 0]

data5 = data5.drop(columns=["satuan", "id"])
data5["golongan_umur"] = data5["golongan_umur"].str.replace(r'\s+', ' ', regex=True).str.strip()
data5 = data5[data5["jumlah"] > 0]

# ==================== MODEL 1: PENDIDIKAN ====================
X_pendidikan = data3.drop(columns=["jumlah", "tahun", "nama_provinsi", "nama_kabupaten_kota"])
y_pendidikan = data3["jumlah"]

preprocessor_pendidikan = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(drop=None, handle_unknown="ignore"), ["jenis_pendidikan"])]
)

model_pendidikan = Pipeline(steps=[
    ("preprocess", preprocessor_pendidikan),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train_pendidikan = X_pendidikan
y_train_pendidikan = y_pendidikan
X_test_pendidikan = X_pendidikan
y_test_pendidikan = y_pendidikan

model_pendidikan.fit(X_train_pendidikan, y_train_pendidikan)
y_pred_pendidikan = model_pendidikan.predict(X_test_pendidikan)

mae_pendidikan = mean_absolute_error(y_test_pendidikan, y_pred_pendidikan)
r2_pendidikan = r2_score(y_test_pendidikan, y_pred_pendidikan)

feature_importance_pendidikan = model_pendidikan.named_steps["model"].feature_importances_
feature_names_pendidikan = model_pendidikan.named_steps["preprocess"].get_feature_names_out()

feat_imp_pendidikan = pd.DataFrame({
    "Feature": feature_names_pendidikan,
    "Importance": feature_importance_pendidikan
}).sort_values(by="Importance", ascending=False)

feat_imp_pendidikan["Feature_Clean"] = feat_imp_pendidikan["Feature"].str.replace("cat__jenis_pendidikan_", "")

# VISUALISASI PENDIDIKAN
st.subheader("📚 Analisis Model - Faktor Pendidikan")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("MAE", f"{mae_pendidikan:,.0f}")
with col2:
    st.metric("R² Score", f"{r2_pendidikan:.4f}")
with col3:
    st.metric("Total Data", len(data3))

fig_imp_pendidikan = px.bar(
    feat_imp_pendidikan,
    x="Importance",
    y="Feature_Clean",
    orientation="h",
    title="Feature Importance - Faktor Pendidikan",
    labels={"Feature_Clean": "Jenis Pendidikan", "Importance": "Importance Score"},
    color="Importance",
    color_continuous_scale="Viridis"
)
fig_imp_pendidikan.update_layout(yaxis_categoryorder="total ascending")
st.plotly_chart(fig_imp_pendidikan, use_container_width=True)

pred_df_pendidikan = pd.DataFrame({
    "Actual": y_test_pendidikan,
    "Predicted": y_pred_pendidikan,
    "Jenis Pendidikan": X_test_pendidikan["jenis_pendidikan"].values
})

fig_pred_pendidikan = px.scatter(
    pred_df_pendidikan,
    x="Actual",
    y="Predicted",
    color="Jenis Pendidikan",
    title="Actual vs Predicted - Pendidikan",
    labels={"Actual": "Nilai Sebenarnya", "Predicted": "Prediksi Model"}
)
fig_pred_pendidikan.add_shape(
    type="line",
    x0=pred_df_pendidikan["Actual"].min(),
    y0=pred_df_pendidikan["Actual"].min(),
    x1=pred_df_pendidikan["Actual"].max(),
    y1=pred_df_pendidikan["Actual"].max(),
    line=dict(dash="dash", color="red")
)
st.plotly_chart(fig_pred_pendidikan, use_container_width=True)

st.write("**Tabel Feature Importance (Pendidikan):**")
st.dataframe(feat_imp_pendidikan[["Feature_Clean", "Importance"]].head(10), use_container_width=True)

# ==================== MODEL 2: USIA ====================
X_usia = data5.drop(columns=["jumlah", "tahun", "nama_provinsi", "nama_kabupaten_kota"])
y_usia = data5["jumlah"]

preprocessor_usia = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(drop=None, handle_unknown="ignore"), ["golongan_umur"])]
)

model_usia = Pipeline(steps=[
    ("preprocess", preprocessor_usia),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train_usia = X_usia
y_train_usia = y_usia
X_test_usia = X_usia
y_test_usia = y_usia

model_usia.fit(X_train_usia, y_train_usia)
y_pred_usia = model_usia.predict(X_test_usia)

mae_usia = mean_absolute_error(y_test_usia, y_pred_usia)
r2_usia = r2_score(y_test_usia, y_pred_usia)

feature_importance_usia = model_usia.named_steps["model"].feature_importances_
feature_names_usia = model_usia.named_steps["preprocess"].get_feature_names_out()

feat_imp_usia = pd.DataFrame({
    "Feature": feature_names_usia,
    "Importance": feature_importance_usia
}).sort_values(by="Importance", ascending=False)

feat_imp_usia["Feature_Clean"] = feat_imp_usia["Feature"].str.replace("cat__golongan_umur_", "")

# VISUALISASI USIA
st.subheader("👥 Analisis Model - Faktor Usia")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("MAE", f"{mae_usia:,.0f}")
with col2:
    st.metric("R² Score", f"{r2_usia:.4f}")
with col3:
    st.metric("Total Data", len(data5))

fig_imp_usia = px.bar(
    feat_imp_usia,
    x="Importance",
    y="Feature_Clean",
    orientation="h",
    title="Feature Importance - Faktor Usia",
    labels={"Feature_Clean": "Golongan Usia", "Importance": "Importance Score"},
    color="Importance",
    color_continuous_scale="Plasma"
)
fig_imp_usia.update_layout(yaxis_categoryorder="total ascending")
st.plotly_chart(fig_imp_usia, use_container_width=True)

pred_df_usia = pd.DataFrame({
    "Actual": y_test_usia,
    "Predicted": y_pred_usia,
    "Golongan Usia": X_test_usia["golongan_umur"].values
})

fig_pred_usia = px.scatter(
    pred_df_usia,
    x="Actual",
    y="Predicted",
    color="Golongan Usia",
    title="Actual vs Predicted - Usia",
    labels={"Actual": "Nilai Sebenarnya", "Predicted": "Prediksi Model"}
)
fig_pred_usia.add_shape(
    type="line",
    x0=pred_df_usia["Actual"].min(),
    y0=pred_df_usia["Actual"].min(),
    x1=pred_df_usia["Actual"].max(),
    y1=pred_df_usia["Actual"].max(),
    line=dict(dash="dash", color="red")
)
st.plotly_chart(fig_pred_usia, use_container_width=True)

st.write("**Tabel Feature Importance (Usia):**")
st.dataframe(feat_imp_usia[["Feature_Clean", "Importance"]].head(10), use_container_width=True)

# KESIMPULAN
st.info(
    "🔍 **Kesimpulan Analisis:**\n"
    "Model Machine Learning berhasil mengidentifikasi faktor-faktor dominan penyebab "
    "tingginya Tingkat Pengangguran Terbuka (TPT) di Kota Bekasi. "
    "Hasil dapat digunakan untuk merancang strategi intervensi yang lebih tertarget."
)
