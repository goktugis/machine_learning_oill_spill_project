import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Sonuçların tutarlılığı için seed
np.random.seed(42)

def calculate_metrics(y_true, y_pred, y_probs, model_name):
    """Metrikleri hesaplayan yardımcı fonksiyon"""
    return {
        'Model': model_name,
        'Accuracy': round(accuracy_score(y_true, y_pred), 3),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 3),
        'Recall': round(recall_score(y_true, y_pred), 3),
        'F1_Score': round(f1_score(y_true, y_pred, zero_division=0), 3),
        'ROC_AUC': round(roc_auc_score(y_true, y_probs), 3)
    }

def train_models(X_train, y_train, X_test, y_test):
    """Farklı modelleri eğiten ana fonksiyon"""
    models = {
        'SVM': SVC(kernel='rbf', probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'DecisionTree': DecisionTreeClassifier(),
        'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500),
        'RandomForest': RandomForestClassifier(n_estimators=50)
    }
    
    results = []
    for name, mdl in models.items():
        try:
            mdl.fit(X_train, y_train)
            y_pred = mdl.predict(X_test)
            y_probs = mdl.predict_proba(X_test)[:, 1]
            results.append(calculate_metrics(y_test, y_pred, y_probs, name))
        except Exception as e:
            print(f"Hata ({name}): {e}")
            results.append({'Model': name, 'Accuracy': 0})
    
    return pd.DataFrame(results)

## 1. VERİ YÜKLEME VE HAZIRLIK
print(">>> Veri Yükleniyor...")
try:
    data = pd.read_csv('oil_spill.csv')
except FileNotFoundError:
    print("HATA: 'oil_spill.csv' dosyası bulunamadı.")
    exit()

# Veriyi Ayır
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Eksik Veri Temizliği
if X.isnull().values.any() or y.isnull().any():
    print("UYARI: NaN değerler temizleniyor...")
    data = data.dropna()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

# Stratified Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Normalizasyon (Z-Score)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Eğitim Seti: {len(y_train)}")
print(f"Test Seti: {len(y_test)}")

## SENARYO 1: TÜM ÖZELLİKLER
print("\n" + "="*58)
print("TABLO 1: TÜM ÖZELLİKLER İLE MODEL PERFORMANSI")
print("="*58)
results1 = train_models(X_train_scaled, y_train, X_test_scaled, y_test)
print(results1)

## SENARYO 2: SEÇİLEN 10 ÖZELLİK
print("\n" + "="*58)
print("TABLO 2: SEÇİLEN 10 ÖZELLİK VE NORMALLİK TESTİ")
print("="*58)

target_feats = ['f_47', 'f_1', 'f_2', 'f_3', 'f_25', 'f_46', 'f_6', 'f_48', 'f_35', 'f_32']
# Mevcut olan özellikleri seç (Hata almamak için)
available_feats = [f for f in target_feats if f in X.columns]

print("--- Normallik Testi (Lilliefors/Kolmogorov-Smirnov) ---")
for feat in available_feats:
    # Python'da lilliefors statsmodels içinde yer alır, burada kstest benzer işlev görür
    stat, p = stats.kstest(X[feat], 'norm', args=(X[feat].mean(), X[feat].std()))
    res = 'Normal Değil' if p < 0.05 else 'Normal'
    print(f"{feat} : p={p:.5f} ({res})")

X_train_sel = scaler.fit_transform(X_train[available_feats])
X_test_sel = scaler.transform(X_test[available_feats])

print("\n--- Seçilen Özelliklerle Sonuçlar ---")
results2 = train_models(X_train_sel, y_train, X_test_sel, y_test)
print(results2)

## SENARYO 3: SMOTE
print("\n" + "="*58)
print("TABLO 3: SMOTE SONRASI PERFORMANS")
print("="*58)

print(f"SMOTE Öncesi Sınıf Dağılımı: {np.bincount(y_train)}")
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train_scaled, y_train)
print(f"SMOTE Sonrası Sınıf Dağılımı: {np.bincount(y_train_sm)}")

print("\n--- SMOTE Uygulanmış Sonuçlar ---")
results3 = train_models(X_train_sm, y_train_sm, X_test_scaled, y_test)
print(results3)