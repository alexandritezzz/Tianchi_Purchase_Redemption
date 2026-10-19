"""Model training, backtesting, prediction and export. Run: python forecast.py."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime

from data_processing import (
    DATA_DIR,
    TRAIN_END,
    HORIZON,
    LOOKBACK,
    TARGETS,
    load_daily_data,
    make_forecast_features,
    make_training_samples,
)

# 配置
CURRENT_TIME = (
    str(datetime.now()).replace(" ", "_")[:-7].replace(":", "").replace("-", "")
)
OUTPUT_DIR = Path(f"output_{CURRENT_TIME}")
SUBMISSION_HEADER = False
SEED = 2026

MODEL_PARAMS = {
    "objective": "regression_l1",
    "n_estimators": 150,
    "learning_rate": 0.03,
    "num_leaves": 10,
    "max_depth": 3,
    "min_child_samples": 300,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.5,
    "reg_lambda": 20.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

# 固定参数后进行月度回测，不在每个测试月上调参。
BACKTEST_ORIGINS = [
    pd.Timestamp("2014-05-31"),
    pd.Timestamp("2014-06-30"),
    pd.Timestamp("2014-07-31"),
]


def fit_models(X, y, model_params=None):
    models = {}
    for target in TARGETS:
        model = lgb.LGBMRegressor(
            **(MODEL_PARAMS if model_params is None else model_params)
        )
        model.fit(X, y[target])
        models[target] = model
    return models


def predict_amounts(models, features, alpha=0.25):
    prediction = pd.DataFrame(index=features.index)

    for target in TARGETS:
        correction = models[target].predict(features)
        correction = np.clip(correction, -3.0, 3.0)

        baseline = features[f"{target}_baseline"].to_numpy()
        model_values = np.maximum(
            np.expm1(np.log1p(baseline) + correction),
            0.0,
        )

        prediction[target] = (1.0 - alpha) * baseline + alpha * model_values

    return prediction


# 回测指标
def evaluate(truth, prediction):
    result = {}

    for target in TARGETS:
        actual = truth[target].to_numpy(dtype=float)
        predicted = prediction[target].to_numpy(dtype=float)

        # 若真实值为零，普通相对误差没有定义。
        # 对这类日期不计算相对指标，并明确记录数量。
        nonzero = actual > 0
        if not nonzero.any():
            raise ValueError(f"{target} 没有正值，无法计算相对误差。")

        relative_error = np.abs(predicted[nonzero] - actual[nonzero]) / actual[nonzero]

        result[f"{target}_mape"] = float(relative_error.mean())
        result[f"{target}_over_30pct_days"] = int((relative_error > 0.30).sum())
        result[f"{target}_zero_actual_days"] = int((~nonzero).sum())

    # 只是参照官方权重构造的误差代理，越小越好。
    # 不是官方积分。
    result["weighted_mape_proxy"] = (
        0.45 * result["purchase_mape"] + 0.55 * result["redeem_mape"]
    )
    return result


def run_backtests(daily, X, y, metadata, model_params=None, output_dir=None):
    output_dir = Path(OUTPUT_DIR if output_dir is None else output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    prediction_blocks = []

    for origin in BACKTEST_ORIGINS:
        forecast_end = origin + pd.Timedelta(days=HORIZON)
        if forecast_end > daily.index.max():
            continue

        # 关键：训练样本的全部标签必须已在 origin 时点可见。
        train_mask = metadata["label_date"] <= origin

        models = fit_models(
            X.loc[train_mask], y.loc[train_mask], model_params=model_params
        )

        features = make_forecast_features(daily, origin)
        truth = daily.loc[features.index, TARGETS]
        prediction = predict_amounts(models, features)

        baseline = pd.DataFrame(
            {target: features[f"{target}_baseline"] for target in TARGETS},
            index=features.index,
        )

        for model_name, values in [
            ("LightGBM", prediction),
            ("weekday_baseline", baseline),
        ]:
            row = {
                "origin": origin.strftime("%Y-%m-%d"),
                "model": model_name,
                **evaluate(truth, values),
            }
            metrics.append(row)

        detail = truth.add_prefix("actual_").join(prediction.add_prefix("predicted_"))
        detail = detail.join(baseline.add_prefix("baseline_"))
        detail["origin"] = origin.strftime("%Y-%m-%d")
        prediction_blocks.append(detail)

    if not metrics:
        raise ValueError("没有可用回测窗口。")

    metrics = pd.DataFrame(metrics)
    metrics.to_csv(output_dir / "backtest_metrics.csv", index=False)

    pd.concat(prediction_blocks).to_csv(
        output_dir / "backtest_predictions.csv", index_label="report_date"
    )

    print("\n月度回测结果：")
    print(metrics.to_string(index=False, float_format="%.4f"))

    summary = metrics.groupby("model")[
        ["purchase_mape", "redeem_mape", "weighted_mape_proxy"]
    ].mean()

    print("\n回测平均误差，越小越好：")
    print(summary.to_string(float_format="%.4f"))

    if (
        summary.loc["LightGBM", "weighted_mape_proxy"]
        >= summary.loc["weekday_baseline", "weighted_mape_proxy"]
    ):
        print(
            "\n注意：LightGBM 尚未超过星期基线。"
            "当前脚本仍输出所选 LightGBM 模型的预测；"
            "应先改进特征并重新回测，再决定正式提交。"
        )

    return metrics


# 最终训练与提交
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily = load_daily_data(data_dir=DATA_DIR, train_end=TRAIN_END)
    print(
        f"数据范围：{daily.index.min():%Y-%m-%d}，至 {daily.index.max():%Y-%m-%d}，共 {len(daily)} 天"
    )

    X, y, metadata = make_training_samples(daily)
    print(f"训练样本形状：{X.shape}")

    run_backtests(daily, X, y, metadata)

    # 截止 8 月 31 日，使用全部已知标签重新训练。
    final_models = fit_models(X, y)
    future_features = make_forecast_features(daily, TRAIN_END)
    prediction = predict_amounts(final_models, future_features)

    submission = pd.DataFrame(
        {
            "report_date": prediction.index.strftime("%Y%m%d"),
            "purchase": np.rint(prediction["purchase"]).astype("int64"),
            "redeem": np.rint(prediction["redeem"]).astype("int64"),
        }
    ).reset_index(drop=True)

    expected_dates = (
        pd.date_range("2014-09-01", "2014-09-30", freq="D").strftime("%Y%m%d").tolist()
    )

    assert submission["report_date"].tolist() == expected_dates
    assert len(submission) == 30
    assert not submission.isna().any().any()
    assert (submission[TARGETS] >= 0).all().all()

    submission_path = OUTPUT_DIR / "tc_comp_predict_table.csv"
    submission.to_csv(submission_path, index=False, header=SUBMISSION_HEADER)

    # 另存带表头的版本，方便人工检查。
    submission.to_csv(OUTPUT_DIR / "prediction_preview.csv", index=False)

    for target, model in final_models.items():
        model.booster_.save_model(str(OUTPUT_DIR / f"{target}_lightgbm.txt"))

        importance = pd.DataFrame(
            {
                "feature": X.columns,
                "gain": model.booster_.feature_importance(importance_type="gain"),
            }
        ).sort_values("gain", ascending=False)

        importance.to_csv(OUTPUT_DIR / f"{target}_feature_importance.csv", index=False)

    config = {
        "train_end": TRAIN_END.strftime("%Y-%m-%d"),
        "horizon": HORIZON,
        "lookback": LOOKBACK,
        "model_params": MODEL_PARAMS,
        "feature_columns": X.columns.tolist(),
    }
    with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    print(f"\n提交文件已生成：{submission_path}")
    print("\n预测前 5 行，金额单位为分：")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
