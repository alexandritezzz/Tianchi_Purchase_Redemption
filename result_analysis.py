import numpy as np
import pandas as pd
from pathlib import Path

# 当前文件路径
file_folder = Path(__file__).resolve().parent

df = pd.read_csv(f"{file_folder}/backtest_predictions.csv")
alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

rows = []

for target in ["purchase", "redeem"]:
    actual = df[f"actual_{target}"].to_numpy()
    model = df[f"predicted_{target}"].to_numpy()
    baseline = df[f"baseline_{target}"].to_numpy()

    # 当前数据没有真实值为零的日期。
    if (actual <= 0).any():
        raise ValueError("存在非正真实值，需要单独处理相对误差。")

    for alpha in alphas:
        predicted = (1 - alpha) * baseline + alpha * model

        signed_error = (predicted - actual) / actual
        absolute_error = np.abs(signed_error)

        detail = pd.DataFrame({
            "origin": df["origin"],
            "error": absolute_error,
            "bias": signed_error,
        })

        monthly_mape = detail.groupby("origin")["error"].mean()

        rows.append({
            "target": target,
            "alpha": alpha,
            "mape": absolute_error.mean(),
            "worst_month_mape": monthly_mape.max(),
            "over_30pct_days": int((absolute_error > 0.30).sum()),
            "bias": signed_error.mean(),
        })

result = pd.DataFrame(rows)
print(result.to_string(index=False, float_format="%.4f"))
result.to_csv(f"{file_folder}/blend_search.csv", index=False)