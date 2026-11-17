
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def prepare_data(df):

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Split dataset into training (70%) and temporary (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Split temporary data into validation (15%) and testing (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # Scale Time and Amount features
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), ["Time", "Amount"])
        ],
        remainder="passthrough"
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        preprocessor
    )
