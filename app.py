import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMAResults

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Harga Saham",
    page_icon="📈",
    layout="wide"
)

# --- Fungsi untuk Memuat Model ---
@st.cache_resource
def load_model(model_path):
    """Memuat model dari file pickle."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please make sure 'forecast_saham.sav' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Fungsi untuk Memuat Data ---
@st.cache_data
def load_data(data_path):
    """Memuat dan memproses data historis saham."""
    try:
        data = pd.read_csv(data_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        return data
    except FileNotFoundError:
        st.warning(f"Data file not found at {data_path}. Historical data plot will be unavailable.")
        return pd.DataFrame() # Mengembalikan dataframe kosong jika file tidak ada

# --- Memuat Aset ---
model = load_model('forecast_saham.sav')
# Ganti 'data_saham_BBCA.csv' dengan nama file data Anda
data = load_data('data_saham_BBCA.csv') 

# --- Antarmuka Aplikasi ---
st.title('📈 Aplikasi Prediksi Harga Saham')
st.write("Gunakan aplikasi ini untuk memprediksi pergerakan harga saham ke depan menggunakan model ARIMA.")

# --- Sidebar untuk Input Pengguna ---
st.sidebar.header("Panel Kontrol")
st.sidebar.write("Pilih jumlah hari untuk diprediksi.")
days_to_forecast = st.sidebar.slider("Jumlah Hari Prediksi", 1, 90, 30, step=1)

# --- Tombol Prediksi ---
if st.sidebar.button("🚀 Lakukan Prediksi"):
    if model is not None:
        with st.spinner('Melakukan prediksi... Mohon tunggu sebentar.'):
            # Melakukan forecast
            forecast = model.forecast(steps=days_to_forecast)
            
            # Membuat DataFrame dari hasil forecast
            forecast_df = pd.DataFrame({
                'Tanggal': pd.to_datetime(forecast.index),
                'Harga Prediksi (IDR)': forecast.values.round(2)
            })
            forecast_df.set_index('Tanggal', inplace=True)

            st.success("Prediksi berhasil dibuat!")
            
            # --- Menampilkan Hasil ---
            col1, col2 = st.columns((1, 2))

            with col1:
                st.subheader(f"Prediksi untuk {days_to_forecast} Hari ke Depan")
                st.dataframe(forecast_df, use_container_width=True)

            with col2:
                st.subheader("Visualisasi Prediksi")
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot data historis jika tersedia
                if not data.empty:
                    # Ambil 200 hari terakhir untuk konteks
                    ax.plot(data['Close'].tail(200), label='Data Historis', color='blue')
                
                # Plot forecast
                ax.plot(forecast_df['Harga Prediksi (IDR)'], label='Hasil Prediksi', color='orange', linestyle='--')
                
                ax.set_title('Prediksi Harga Saham vs Data Historis')
                ax.set_xlabel('Tanggal')
                ax.set_ylabel('Harga Penutupan (IDR)')
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                st.pyplot(fig)
    else:
        st.error("Model tidak dapat dimuat. Prediksi tidak dapat dilakukan.")

st.sidebar.info("Aplikasi ini menggunakan model ARIMA yang telah dilatih pada data historis saham.")

# --- Tampilan Awal ---
if 'last_run' not in st.session_state:
    st.info("Silakan pilih jumlah hari di sidebar dan klik tombol 'Lakukan Prediksi' untuk memulai.")
    if not data.empty:
        st.subheader("Data Historis Harga Saham (5 Hari Terakhir)")
        st.dataframe(data.tail(), use_container_width=True)
    st.session_state['last_run'] = True

