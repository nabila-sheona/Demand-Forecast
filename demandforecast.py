
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
import pickle
warnings.filterwarnings('ignore')

print("="*90)
print("SEASONALITY-FOCUSED DEMAND FORECAST MODEL")
print("="*90)

# =====================================================================
# STEP 1: LOAD DATA
# =====================================================================
print("\nSTEP 1: LOADING DATA")
print("-"*90)

encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
df = None

for encoding in encodings:
    try:
        df = pd.read_csv('./prediction/Copy of forecastalldata (1).csv', encoding=encoding)
        print(f"✓ File loaded with {encoding} encoding")
        break
    except:
        continue

if df is None:
    print("✗ Error: Could not load file")
    exit()

# Convert to datetime FIRST before checking date range
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

print(f"✓ Loaded {len(df):,} records")
print(f"  Date range: {df['Order Date'].min().strftime('%m/%d/%Y')} to {df['Order Date'].max().strftime('%m/%d/%Y')}")


# =====================================================================
# STEP 2: AGGREGATE TO MONTHLY LEVEL
# =====================================================================

# =====================================================================
# STEP 3: CREATE TEMPORAL FEATURES
# =====================================================================



# =====================================================================
# STEP 4: CREATE SEASONALITY PATTERNS (EMPHASIZED!)
# =====================================================================

# =====================================================================
# STEP 5: CREATE LAG FEATURES
# =====================================================================

# =====================================================================
# STEP 6: ENCODE CATEGORICALS
# =====================================================================


# =====================================================================
# STEP 7: PREPARE FEATURES - PRIORITIZE SEASONALITY
# =====================================================================


# Clean data

# =====================================================================
# STEP 8: TIME-BASED SPLIT
# =====================================================================

# =====================================================================
# STEP 9: TRAIN MODELS (TUNED FOR SEASONALITY)
# =====================================================================

#model 3: random forest
# Model 4: Neural Network

# =====================================================================
# STEP 10: SELECT BEST MODEL
# =====================================================================

# Feature importance


# =====================================================================
# STEP 11: PATTERN ANALYSIS
# =====================================================================


# =====================================================================
# STEP 12: SAVE MODEL
# =====================================================================


model_package = {
    'models': models,
    'best_model_name': best_model_name,
    'label_encoders': label_encoders,
    'feature_columns': feature_columns,
    'results': results,
    'monthly_pattern': monthly_pattern,
    'season_pattern': season_pattern,
    'combo_baseline': combo_baseline,
    'season_map': season_map
}



print(f"✓ Saved model: monthly_demand_forecast_model.pkl")
print(f"✓ Saved data: monthly_aggregated_data.csv")

print("\n" + "="*90)
print("SUMMARY")
print("="*90)
print(f"✓ Best Model: {best_model_name}")
print(f"✓ MAPE: {results[best_model_name]['MAPE']:.2f}%")
print(f"✓ R² Score: {results[best_model_name]['R2']:.4f}")
print(f"\n✓ Key Features:")
print(f"  - Prioritized Historical_Month_Avg & Historical_Season_Avg")
print(f"  - Added seasonality indices (Month_Index, Season_Index)")
print(f"  - Included 12-month lag for year-over-year comparison")
print(f"  - Tuned model to emphasize seasonal patterns")
print(f"  - Reduced overfitting with appropriate regularization")
print("="*90)
