import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Amazon_Sales_Report.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Target
y = df["status"]

# Remove leakage columns
X = df.drop(columns=["status","courier status"], errors="ignore")

# Numerical columns
num_cols = X.select_dtypes(include=["int64","float64"]).columns

# Categorical columns
cat_cols = X.select_dtypes(include=["object"]).columns

# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Preprocessor
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Train
model.fit(X_train,y_train)

# Save
joblib.dump(model,"model.pkl")

print("Model Saved Successfully")