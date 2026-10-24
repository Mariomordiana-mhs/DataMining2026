# Data Mining Workspace

Repositori ini berisi kumpulan modul praktikum dan eksperimen **Data Mining** menggunakan Python (`.py`) dan Jupyter Notebook (`.ipynb`).

## 1. Struktur File

Berikut adalah struktur lengkap direktori dan file di dalam *workspace* **Data Mining**:

```text
Data Mining/
├── 01_Data_Preprocessing/
│   ├── __pycache__/
│   ├── data1.py                                
│   └── experiment1.ipynb                       
│
├── 02_home_prediction/
│   ├── prediction.ipynb                        
│   └── prediction.py                           
│
├── 03_polynomial_regression_project/
│   ├── __pycache__/
│   ├── plots/                                 
│   ├── polynomial_regression_demo.ipynb        
│   └── polynomial_regression.py               
│
└── README.md                                   
```

### Keterangan Modul
* **`01_Data_Preprocessing/`**: Modul persiapan dan pembersihan data awal (`data1.py` dan `experiment1.ipynb`).
* **`02_home_prediction/`**: Modul pemodelan regresi untuk memprediksi harga rumah (`prediction.py` dan `prediction.ipynb`).
* **`03_polynomial_regression_project/`**: Modul analisis **Polynomial Regression** pada dataset harga mobil (*Car Sales*), mencakup visualisasi hubungan non-linear (`curbweight`, `carlength`, `horsepower` terhadap `price`), evaluasi `Pipeline` + `StandardScaler`, transformasi `PolynomialFeatures(degree=2)`, serta pencarian *hyperparameter* terbaik menggunakan `GridSearchCV`.

## 2. Cara Menjalankan Project

### Persiapan Awal (Instalasi Library)
Buka terminal di VS Code (`Ctrl + `` ` `` atau menu **Terminal > New Terminal**), lalu instal seluruh pustaka Python yang dibutuhkan:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn jupyter ipykernel
```

### A. Menjalankan File Script Python (`.py`)

Kamu bisa menjalankan masing-masing modul langsung dari *root* folder `Data Mining` di terminal VS Code:

**1. Menjalankan Project 01 — Data Preprocessing:**
```bash
python 01_Data_Preprocessing/data1.py
```

**2. Menjalankan Project 02 — Home Prediction:**
```bash
python 02_home_prediction/prediction.py
```

**3. Menjalankan Project 03 — Polynomial Regression:**
```bash
python 03_polynomial_regression_project/polynomial_regression.py
```

*(Alternatif: Buka file `.py` yang diinginkan di editor VS Code, lalu klik tombol **Run Python File (▶)** di pojok kanan atas).*

### B. Menjalankan File Jupyter Notebook (`.ipynb`)

1. Buka file `.ipynb` yang ingin dijalankan di VS Code:
   * `03_polynomial_regression_project/polynomial_regression_demo.ipynb`
   * `02_home_prediction/prediction.ipynb`
   * `01_Data_Preprocessing/experiment1.ipynb`
2. Klik **Select Kernel** di pojok kanan atas *notebook*, lalu pilih *environment* **Python 3** (misalnya `Python 3.13.x 64-bit`).
3. Klik tombol **Run All** di bagian atas *notebook* untuk mengeksekusi seluruh *cell* sekaligus, atau tekan `Shift + Enter` pada tiap *cell* secara berurutan.
