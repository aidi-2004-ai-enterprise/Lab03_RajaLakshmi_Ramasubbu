"""
train.py
Train XGBoost model on Seaborn Penguins dataset and save model, columns, and label encoder for deployment.
"""
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import joblib
import os

def train_and_save_model(model_path: str = "app/data/model.pkl") -> None:
    """
    Loads the penguins dataset, preprocesses it, trains an XGBoost classifier,
    evaluates performance, and saves the model, feature columns, and label encoder.
    
    Args:
        model_path (str): File path to save the model and artifacts.
    Returns:
        None
    """
    # Load penguins dataset
    df = sns.load_dataset("penguins").dropna()

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=["sex", "island"])
    
    # Label encode the target
    le = LabelEncoder()
    df["species"] = le.fit_transform(df["species"])

    # Features and target
    X = df.drop(columns=["species"])
    y = df["species"]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Model with parameters to avoid overfitting
    model = XGBClassifier(
        max_depth=3,
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print(f"Train F1 Score: {f1_score(y_train, y_pred_train, average='macro'):.4f}")
    print(f"Test F1 Score: {f1_score(y_test, y_pred_test, average='macro'):.4f}")

    # Save model, columns, label encoder
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({
        "model": model,
        "columns": list(X.columns),
        "label_encoder": le
    }, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
