import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

def load_fruit_sales_data():
    """Membuat/Memuat dataset simulasi penjualan dan kuantitas buah harian."""
    np.random.seed(42)
    n_rows = 1000
    
    data = {
        'Transaction_ID': [f'TRX-{i:04d}' for i in range(1, n_rows + 1)],
        'Fruit_Name': np.random.choice(['Apel', 'Jeruk', 'Pisang', 'Mangga', 'Anggur'], n_rows),
        'Quantity_Sold': np.random.exponential(scale=50, size=n_rows).astype(int) + 1, # Distribusi miring (skewed)
        'Price_Per_Kg': np.random.uniform(15000, 45000, n_rows),
        'Temperature': np.random.uniform(20, 35, n_rows),
        'Is_Weekend': np.random.choice([0, 1], n_rows, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(n_rows, 20), 'Quantity_Sold'] = np.nan
    df.loc[np.random.choice(n_rows, 15), 'Price_Per_Kg'] = np.nan
    
    # Duplikat beberapa baris transaksi
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    
    # Menambahkan outlier ekstrem pada kuantitas terjual
    df.loc[5, 'Quantity_Sold'] = 1200
    df.loc[12, 'Quantity_Sold'] = 950
    
    return df

def check_missing_values(df):
    """Menghitung jumlah missing values."""
    return df.isnull().sum().sort_values(ascending=False)

def handle_duplicates(df):
    """Menghapus data duplikat berdasarkan Transaction_ID."""
    duplicates = df[df.duplicated(['Transaction_ID'])]
    print(f"Jumlah baris duplikat ditemukan: {len(duplicates)}")
    cleaned_df = df.drop_duplicates(subset=['Transaction_ID'])
    return cleaned_df

def log_transform_quantity(df, column_name='Quantity_Sold'):
    """Transformasi log untuk mengatasi skewness pada kuantitas penjualan."""
    valid_data = df[column_name].dropna()
    print(f"Skewness awal {column_name}: {valid_data.skew():.4f}")
    
    log_transformed = np.log(valid_data)
    print(f"Skewness setelah log transform: {log_transformed.skew():.4f}")
    return log_transformed

def impute_missing_values(df):
    """Imputasi missing values pada Quantity_Sold (median) dan Price_Per_Kg (mean)."""
    if "Quantity_Sold" in df.columns:
        med_qty = df["Quantity_Sold"].median()
        df["Quantity_Sold"] = df["Quantity_Sold"].fillna(med_qty)
        print(f"Quantity_Sold diisi dengan median: {med_qty}")
        
    if "Price_Per_Kg" in df.columns:
        mean_price = df["Price_Per_Kg"].mean()
        df["Price_Per_Kg"] = df["Price_Per_Kg"].fillna(mean_price)
        print(f"Price_Per_Kg diisi dengan mean: {mean_price:.2f}")
        
    return df

def scale_numerical_features(df):
    """Standardisasi fitur numerik menggunakan StandardScaler."""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[num_cols])
    return scaled_data, num_cols

def remove_outliers_zscore(df, column_name='Quantity_Sold', threshold=3):
    """Menghapus outlier menggunakan Z-score."""
    z_scores = np.abs(stats.zscore(df[column_name]))
    filtered_df = df[z_scores < threshold]
    print(f"Jumlah baris sebelum filter outlier: {len(df)}")
    print(f"Jumlah baris setelah filter outlier (Z < {threshold}): {len(filtered_df)}")
    return filtered_df