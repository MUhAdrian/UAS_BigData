import pandas as pd
import numpy as np
import os
from pathlib import Path

# =======================
# DATA INGESTION & CLEANING
# =======================

class DataProcessor:
    """
    Kelas untuk melakukan data ingestion dan cleaning pada dataset pengangguran Kota Bekasi
    """
    
    def __init__(self, input_dir='.', output_dir='.'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.df = None
        self.original_df = None
        
    def load_data(self, filename):
        """Load data dari CSV file"""
        try:
            filepath = os.path.join(self.input_dir, filename)
            self.original_df = pd.read_csv(filepath)
            self.df = self.original_df.copy()
            print(f"✓ Data berhasil dimuat: {filename}")
            print(f"  - Baris: {len(self.df)}, Kolom: {len(self.df.columns)}")
            return self
        except FileNotFoundError:
            print(f"✗ File tidak ditemukan: {filename}")
            return None
    
    def remove_duplicates(self):
        """Hapus data duplikat"""
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_rows - len(self.df)
        print(f"✓ Duplikat dihapus: {removed} baris")
        return self
    
    def handle_missing_values(self, strategy='drop'):
        """
        Handle nilai yang hilang (missing values)
        strategy: 'drop' (hapus) atau 'mean'/'median' (isi dengan rata-rata/median)
        """
        missing_before = self.df.isnull().sum().sum()
        
        if missing_before == 0:
            print("✓ Tidak ada missing values")
            return self
        
        if strategy == 'drop':
            self.df = self.df.dropna()
            print(f"✓ Missing values dihapus: {missing_before} cell")
        else:
            # Isi missing values untuk kolom numerik
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    if strategy == 'mean':
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    elif strategy == 'median':
                        self.df[col].fillna(self.df[col].median(), inplace=True)
            print(f"✓ Missing values diisi dengan {strategy}")
        
        return self
    
    def clean_whitespace(self):
        """Bersihkan whitespace pada kolom string"""
        string_cols = self.df.select_dtypes(include=['object']).columns
        for col in string_cols:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].str.strip()
                self.df[col] = self.df[col].str.replace(r'\s+', ' ', regex=True)
        
        print(f"✓ Whitespace dibersihkan pada {len(string_cols)} kolom string")
        return self
    
    def standardize_column_names(self):
        """Standardisasi nama kolom ke lowercase dengan underscore"""
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        print("✓ Nama kolom distandardisasi")
        return self
    
    def remove_outliers(self, columns=None, method='iqr', threshold=1.5):
        """
        Remove outliers menggunakan IQR atau Z-score
        method: 'iqr' atau 'zscore'
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        initial_rows = len(self.df)
        
        if method == 'iqr':
            for col in columns:
                if col in self.df.columns:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
            
            removed = initial_rows - len(self.df)
            print(f"✓ Outliers dihapus (IQR method): {removed} baris")
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(self.df[columns].select_dtypes(include=[np.number])))
            self.df = self.df[(z_scores < 3).all(axis=1)]
            removed = initial_rows - len(self.df)
            print(f"✓ Outliers dihapus (Z-score method): {removed} baris")
        
        return self
    
    def convert_data_types(self, type_dict=None):
        """Konversi tipe data kolom"""
        if type_dict:
            for col, dtype in type_dict.items():
                if col in self.df.columns:
                    try:
                        self.df[col] = self.df[col].astype(dtype)
                    except ValueError:
                        print(f"⚠ Tidak bisa convert {col} ke {dtype}")
            print("✓ Tipe data dikonversi")
        return self
    
    def filter_rows(self, condition_dict):
        """Filter baris berdasarkan kondisi"""
        initial_rows = len(self.df)
        
        for col, value in condition_dict.items():
            if isinstance(value, list):
                self.df = self.df[self.df[col].isin(value)]
            else:
                self.df = self.df[self.df[col] == value]
        
        removed = initial_rows - len(self.df)
        print(f"✓ Data difilter: {removed} baris dihapus, {len(self.df)} baris tersisa")
        return self
    
    def drop_columns(self, columns):
        """Hapus kolom yang tidak diperlukan"""
        cols_to_drop = [col for col in columns if col in self.df.columns]
        self.df = self.df.drop(columns=cols_to_drop)
        print(f"✓ Kolom dihapus: {cols_to_drop}")
        return self
    
    def get_summary(self):
        """Dapatkan ringkasan data"""
        print("\n" + "="*50)
        print("RINGKASAN DATA")
        print("="*50)
        print(f"Dimensi: {self.df.shape}")
        print(f"\nTipe Data:\n{self.df.dtypes}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        print(f"\nDeskripsi Statistik:\n{self.df.describe()}")
        print("="*50 + "\n")
        return self
    
    def save_to_csv(self, filename, index=False):
        """Simpan data ke CSV"""
        filepath = os.path.join(self.output_dir, filename)
        self.df.to_csv(filepath, index=index, encoding='utf-8')
        print(f"✓ Data berhasil disimpan: {filepath}")
        return self
    
    def display_preview(self, rows=5):
        """Tampilkan preview data"""
        print(f"\nPreview Data (5 baris pertama):")
        print(self.df.head(rows))
        print(f"\n✓ Total baris: {len(self.df)}, Total kolom: {len(self.df.columns)}\n")
        return self


# =======================
# EXAMPLE USAGE
# =======================

if __name__ == "__main__":
    
    # Contoh 1: Process data TPT
    print("="*50)
    print("PROCESSING: DATA TPT BEKASI")
    print("="*50 + "\n")
    
    processor = DataProcessor(input_dir='.', output_dir='.')
    
    processor.load_data('bekasi_ml_percent_ready.csv') \
             .display_preview() \
             .remove_duplicates() \
             .handle_missing_values(strategy='drop') \
             .clean_whitespace() \
             .standardize_column_names() \
             .get_summary() \
             .save_to_csv('bekasi_cleaned.csv')
    
    
    # Contoh 2: Process data3.csv dengan filtering dan cleaning
    print("\n" + "="*50)
    print("PROCESSING: DATA3.CSV")
    print("="*50 + "\n")
    
    processor2 = DataProcessor(input_dir='.', output_dir='.')
    
    processor2.load_data('data3.csv') \
              .display_preview() \
              .remove_duplicates() \
              .clean_whitespace() \
              .standardize_column_names() \
              .handle_missing_values(strategy='drop') \
              .get_summary() \
              .save_to_csv('data3_cleaned.csv')
    
    
    # Contoh 3: Process data5.csv
    print("\n" + "="*50)
    print("PROCESSING: DATA5.CSV")
    print("="*50 + "\n")
    
    processor3 = DataProcessor(input_dir='.', output_dir='.')
    
    processor3.load_data('data5.csv') \
              .display_preview() \
              .remove_duplicates() \
              .clean_whitespace() \
              .standardize_column_names() \
              .handle_missing_values(strategy='drop') \
              .get_summary() \
              .save_to_csv('data5_cleaned.csv')
    
    
    print("\n✓ SEMUA PROSES SELESAI!")
