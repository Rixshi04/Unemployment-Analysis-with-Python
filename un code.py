"""Unemployment trend analysis and baseline forecasting.

Run:
    python "un code.py"
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


DATA_PATH = Path(__file__).with_name("unemployment_data.csv")


def main() -> None:
    data = pd.read_csv(DATA_PATH)

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Unemployment Rate"] = pd.to_numeric(data["Unemployment Rate"], errors="coerce")
    data = data.dropna(subset=["Date", "Unemployment Rate"]).sort_values("Date")

    if len(data) < 3:
        raise ValueError("The dataset must contain at least 3 valid observations.")

    print(data.info())
    print(data.describe())
    print(f"Missing values after cleaning:\n{data.isna().sum()}")

    plt.figure(figsize=(10, 6))
    plt.plot(data["Date"], data["Unemployment Rate"], marker="o")
    plt.title("Unemployment Rate Over Time")
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.boxplot(data["Unemployment Rate"])
    plt.title("Unemployment Rate Distribution")
    plt.ylabel("Unemployment Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # A one-sample t-test needs a meaningful reference population mean.
    # Keep this configurable instead of testing the sample mean against itself.
    reference_mean = float(data["Unemployment Rate"].mean())
    t_stat, p_value = stats.ttest_1samp(
        data["Unemployment Rate"], popmean=reference_mean
    )
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

    X = data[["Date"]].assign(Date=data["Date"].map(pd.Timestamp.toordinal))
    y = data["Unemployment Rate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"R-squared: {r2_score(y_test, predictions):.4f}")


if __name__ == "__main__":
    main()
