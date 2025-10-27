# car.py — train and save Random Forest Model

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 1️⃣ Load your dataset
df = pd.read_csv("car.csv")   # <-- make sure serial number column is present
print("✅ Data loaded successfully")

# 2️⃣ Drop unnecessary column(s)
# Replace 'serial_number' with the exact column name from your CSV (e.g., 'S.No', 'Unnamed: 0')
if 'serial_number' in df.columns:
    df = df.drop(columns=['serial_number'])
elif 'S.No' in df.columns:
    df = df.drop(columns=['S.No'])
elif 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# 3️⃣ Encode categorical columns
from sklearn.preprocessing import LabelEncoder

cat_cols = ['car_name', 'brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']
le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# 4️⃣ Define input (X) and output (Y)
X = df.drop(columns=['selling_price'])   # <-- target column
Y = df['selling_price']

# 5️⃣ Split data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# 6️⃣ Train Random Forest Model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)

# 7️⃣ Test Accuracy (R² score)
from sklearn.metrics import r2_score
Y_pred = rf.predict(X_test)
acc = r2_score(Y_test, Y_pred)
print("✅ Model trained successfully with R2 Score:", round(acc, 2))

# 8️⃣ Save model as car.pkl
import pickle
with open("car.pkl", "wb") as f:
    pickle.dump(rf, f)

print("✅ Model saved successfully as car.pkl")
