import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

# Menonaktifkan peringatan agar output konsisten dan rapi
warnings.filterwarnings("ignore")

# Direktori untuk menyimpan hasil plot visualisasi
OUTPUT_PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)


def load_car_data() -> pd.DataFrame:
    
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/encoded_car_data.csv"
    local_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoded_car_data.csv")

    if os.path.exists(local_file):
        print(f"[INFO] Memuat data dari file lokal: {local_file}")
        return pd.read_csv(local_file)

    try:
        print(f"[INFO] Mengunduh dataset dari: {url}")
        df = pd.read_csv(url)
        return df
    except Exception as e:
        print(f"[PERINGATAN] Tidak dapat mengunduh dari internet ({e}).")
        print("[INFO] Menggunakan dataset simulasi (fallback) dengan struktur fitur yang sama...")
        np.random.seed(42)
        n_samples = 205
        curbweight = np.random.uniform(1500, 4000, n_samples)
        carlength = np.random.uniform(140, 210, n_samples)
        horsepower = np.random.uniform(50, 280, n_samples)
        enginesize = np.random.uniform(60, 320, n_samples)
        highwaympg = np.random.uniform(16, 54, n_samples)
        citympg = np.random.uniform(13, 49, n_samples)
        wheelbase = np.random.uniform(86, 120, n_samples)
        carwidth = np.random.uniform(60, 72, n_samples)
        boreratio = np.random.uniform(2.5, 4.0, n_samples)

        # Hubungan non-linear (kuadratik) untuk harga mobil
        price = (
            2000
            + 1.8 * curbweight
            + 0.0012 * (curbweight ** 2)
            + 35.0 * horsepower
            + 0.15 * (horsepower ** 2)
            + 25.0 * carlength
            - 120.0 * highwaympg
            + np.random.normal(0, 1500, n_samples)
        )

        df = pd.DataFrame({
            "curbweight": curbweight,
            "carlength": carlength,
            "horsepower": horsepower,
            "enginesize": enginesize,
            "highwaympg": highwaympg,
            "citympg": citympg,
            "wheelbase": wheelbase,
            "carwidth": carwidth,
            "boreratio": boreratio,
            "three": np.random.randint(0, 2, n_samples),
            "price": price,
        })
        return df


def get_R2_features(model, X: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, plot_name: str = "r2_features.png"):
    """
    Menghitung dan memplot nilai R^2 untuk setiap fitur individual pada data train dan test.
    """
    features = list(X.columns)
    if "three" in features:
        features.remove("three")

    R_2_train = []
    R_2_test = []

    for feature in features:
        model.fit(X_train[[feature]], y_train)
        R_2_train.append(model.score(X_train[[feature]], y_train))
        R_2_test.append(model.score(X_test[[feature]], y_test))

    plt.figure(figsize=(10, 5))
    plt.bar(features, R_2_train, label="Train", alpha=0.8)
    plt.bar(features, R_2_test, label="Test", alpha=0.8)
    plt.xticks(rotation=90)
    plt.ylabel("$R^2$")
    plt.title(f"Perbandingan $R^2$ Tiap Fitur ({plot_name})")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_PLOTS_DIR, plot_name)
    plt.savefig(save_path)
    plt.close()

    print(f"  -> Rata-rata R^2 Training: {np.mean(R_2_train):.4f} | Rata-rata R^2 Testing: {np.mean(R_2_test):.4f}")
    print(f"  -> Maksimum R^2 Training : {np.max(R_2_train):.4f} | Maksimum R^2 Testing : {np.max(R_2_test):.4f}")
    print(f"  -> Plot disimpan di      : {save_path}")


def plot_dis(y, yhat, plot_name: str = "distribution.png"):
    """
    Memplot perbandingan distribusi nilai Aktual (Actual Value) vs Prediksi (Fitted Values).
    """
    plt.figure(figsize=(8, 5))
    sns.kdeplot(y, color="r", label="Actual Value", linewidth=2)
    sns.kdeplot(yhat, color="b", label="Fitted Values", linewidth=2)
    plt.title("Actual vs Fitted Values for Price")
    plt.xlabel("Price (in dollars)")
    plt.ylabel("Proportion of Cars")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_PLOTS_DIR, plot_name)
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Plot distribusi disimpan di: {save_path}")


def visualize_nonlinear_relationships(data: pd.DataFrame):
    """
    Visualisasi hubungan polinomial orde-2 untuk curbweight, carlength, dan horsepower (Exercise 1).
    """
    print("\n--- 1. Visualisasi Hubungan Non-Linear Fitur terhadap Price ---")
    for feat in ["curbweight", "carlength", "horsepower"]:
        if feat in data.columns:
            sns.lmplot(x=feat, y="price", data=data, order=2, height=4, aspect=1.3)
            plt.title(f"Polynomial Fit (Order 2): {feat} vs price")
            plt.tight_layout()
            save_path = os.path.join(OUTPUT_PLOTS_DIR, f"lmplot_{feat}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"  -> Grafik {feat} vs price disimpan di: {save_path}")


def main():
    print("=" * 65)
    print("   PROJECT: POLYNOMIAL REGRESSION - PREDIKSI HARGA MOBIL")
    print("=" * 65)

    # 1. Load Data
    data = load_car_data()
    print(f"\nDimensi Dataset: {data.shape[0]} baris, {data.shape[1]} kolom")
    print("5 Baris Pertama Data:")
    print(data.head())

    # 2. Visualisasi awal (termasuk Exercise 1: horsepower vs price)
    visualize_nonlinear_relationships(data)

    # 3. Data Preparation & Train-Test Split
    print("\n--- 2. Persiapan Data & Train-Test Split (70% Train, 30% Test) ---")
    X = data.drop("price", axis=1)
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"  -> Jumlah sampel Training : {X_train.shape[0]}")
    print(f"  -> Jumlah sampel Testing  : {X_test.shape[0]}")

    # 4. Multiple Linear Regression (Baseline)
    print("\n--- 3. Baseline: Multiple Linear Regression ---")
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    predicted_lr = lm.predict(X_test)
    print(f"  -> R^2 pada data Training : {lm.score(X_train, y_train):.4f}")
    print(f"  -> R^2 pada data Testing  : {lm.score(X_test, y_test):.4f}")
    plot_dis(y_test, predicted_lr, "dist_linear_regression.png")

    # Evaluasi R^2 fitur individual dengan Linear Regression
    print("\n--- 4. Evaluasi Fitur Individual (Linear Regression) ---")
    get_R2_features(lm, X, X_train, X_test, y_train, y_test, "r2_individual_linear.png")

    # 5. Exercise 2 & 3: Pipeline (StandardScaler + LinearRegression)
    print("\n--- 5. Pipeline: StandardScaler + LinearRegression (Exercise 2 & 3) ---")
    pipe_lr = Pipeline([
        ("ss", StandardScaler()),
        ("lr", LinearRegression())
    ])
    pipe_lr.fit(X_train, y_train)
    predicted_pipe_lr = pipe_lr.predict(X_test)
    print(f"  -> R^2 pada data Training : {pipe_lr.score(X_train, y_train):.4f}")
    print(f"  -> R^2 pada data Testing  : {pipe_lr.score(X_test, y_test):.4f}")
    plot_dis(y_test, predicted_pipe_lr, "dist_scaled_linear.png")
    get_R2_features(pipe_lr, X, X_train, X_test, y_train, y_test, "r2_individual_scaled_linear.png")

    # 6. Polynomial Features (Multiple Features - Degree 2)
    print("\n--- 6. Polynomial Features (Degree=2) pada Semua Fitur ---")
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    print(f"  -> Jumlah fitur setelah transformasi polinomial (degree=2): {X_train_poly.shape[1]}")

    lm_poly = LinearRegression()
    lm_poly.fit(X_train_poly, y_train)
    print(f"  -> R^2 pada data Training : {lm_poly.score(X_train_poly, y_train):.4f}")
    print(f"  -> R^2 pada data Testing  : {lm_poly.score(X_test_poly, y_test):.4f} (Overfitting jika bernilai rendah/negatif)")

    # 7. Pipeline Polynomial Features pada Fitur Individual
    print("\n--- 7. Pipeline Polynomial (Degree=2) pada Fitur Individual ---")
    pipe_poly = Pipeline([
        ("polynomial", PolynomialFeatures(include_bias=False, degree=2)),
        ("model", LinearRegression())
    ])
    get_R2_features(pipe_poly, X, X_train, X_test, y_train, y_test, "r2_individual_poly_deg2.png")

    # 8. GridSearchCV dan Pipeline (Hyperparameter Tuning)
    print("\n--- 8. GridSearchCV + Pipeline (Mencari Degree Polinomial Terbaik) ---")
    pipe_grid = Pipeline([
        ("scaler", StandardScaler()),
        ("polynomial", PolynomialFeatures(include_bias=False, degree=2)),
        ("model", LinearRegression())
    ])
    param_grid = {
        "polynomial__degree": [1, 2, 3]
    }
    search = GridSearchCV(pipe_grid, param_grid, n_jobs=1, cv=5)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print(f"  -> Parameter Terbaik (Best Params) : {search.best_params_}")
    print(f"  -> R^2 Terbaik pada Data Testing   : {best_model.score(X_test, y_test):.4f}")
    predicted_best = best_model.predict(X_test)
    plot_dis(y_test, predicted_best, "dist_best_gridsearch.png")

    # 9. Exercise 4: GridSearchCV untuk setiap fitur individual
    print("\n--- 9. GridSearchCV pada Masing-Masing Fitur Individual (Exercise 4) ---")
    features = [f for f in X.columns if f != "three"]
    R_2_train_gs = []
    R_2_test_gs = []

    for feature in features:
        param_grid_indiv = {
            "polynomial__degree": [1, 2, 3, 4, 5],
            "model__positive": [True, False]
        }
        pipe_indiv = Pipeline([
            ("polynomial", PolynomialFeatures(include_bias=False, degree=2)),
            ("model", LinearRegression())
        ])
        search_indiv = GridSearchCV(pipe_indiv, param_grid_indiv, n_jobs=1, cv=5)
        search_indiv.fit(X_train[[feature]], y_train)
        best_indiv = search_indiv.best_estimator_

        R_2_train_gs.append(best_indiv.score(X_train[[feature]], y_train))
        R_2_test_gs.append(best_indiv.score(X_test[[feature]], y_test))

    print(f"  -> Rata-rata R^2 Training (GridSearch Individual): {np.mean(R_2_train_gs):.4f}")
    print(f"  -> Rata-rata R^2 Testing  (GridSearch Individual): {np.mean(R_2_test_gs):.4f}")
    print(f"  -> Maksimum R^2 Training  (GridSearch Individual): {np.max(R_2_train_gs):.4f}")
    print(f"  -> Maksimum R^2 Testing   (GridSearch Individual): {np.max(R_2_test_gs):.4f}")

    print("\n" + "=" * 65)
    print("Selesai! Seluruh hasil evaluasi dan grafik telah dibuat.")
    print("=" * 65)


if __name__ == "__main__":
    main()