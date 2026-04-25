# ==============================================
# TRAIN AND SAVE MODEL - RUN THIS ONCE
# ==============================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("=" * 60)
print("TRAINING AND SAVING MODEL")
print("=" * 60)

# -------------------------------------------------
# 1. Load Data
# -------------------------------------------------
print("\n📂 Loading data...")
train_df = pd.read_csv('data/train.csv')
print(f"   Training data: {train_df.shape}")

# -------------------------------------------------
# 2. Prepare Features and Target
# -------------------------------------------------
print("\n🔧 Preparing data...")
X = train_df.drop(columns=['price_range'])
y = train_df['price_range']
print(f"   Features: {X.shape}")
print(f"   Target: {y.shape}")

# -------------------------------------------------
# 3. Scale Features
# -------------------------------------------------
print("\n📏 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   ✅ Scaling complete")

# -------------------------------------------------
# 4. Train Model
# -------------------------------------------------
print("\n🤖 Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_scaled, y)
print("   ✅ Training complete")

# Quick accuracy check
y_pred = model.predict(X_scaled)
accuracy = accuracy_score(y, y_pred)
print(f"   Training Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# -------------------------------------------------
# 5. Save Model and Scaler
# -------------------------------------------------
print("\n💾 Saving model and scaler...")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save model
model_path = 'models/logistic_regression_model.pkl'
joblib.dump(model, model_path)
print(f"   ✅ Model saved to: {model_path}")

# Save scaler
scaler_path = 'models/standard_scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"   ✅ Scaler saved to: {scaler_path}")

# -------------------------------------------------
# 6. Verify Saved Files
# -------------------------------------------------
print("\n🔍 Verifying saved files...")
model_size = os.path.getsize(model_path)
scaler_size = os.path.getsize(scaler_path)
print(f"   Model file size:  {model_size:,} bytes")
print(f"   Scaler file size: {scaler_size:,} bytes")

# Test loading
loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)
print("   ✅ Both files load correctly!")

# Test prediction
test_input = np.array([[1250, 1, 1.5, 1, 10, 1, 32, 0.5, 140, 4, 10, 1000, 1250, 2000, 12, 7, 10, 1, 1, 1]])
test_scaled = loaded_scaler.transform(test_input)
test_pred = loaded_model.predict(test_scaled)[0]
print(f"   ✅ Test prediction works! Predicted price range: {test_pred}")

print("\n" + "=" * 60)
print("✅ ALL DONE! You can now run: streamlit run app.py")
print("=" * 60)