from pathlib import Path
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
from chinese_calendar import is_workday
from datetime import datetime

# ============================================================
# 配置
# ============================================================

CURRENT_TIME = str(datetime.now()).replace(' ','_')[:-7].replace(':','').replace('-','')

DATA_DIR = Path(".")
OUTPUT_DIR = Path(f"output_{CURRENT_TIME}")

TRAIN_END = pd.Timestamp("2014-08-31")
HORIZON = 30
LOOKBACK = 56

TARGETS = ["purchase", "redeem"]
SUBMISSION_HEADER = False

SEED = 42

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


# ============================================================
# 数据读取与检查
# ============================================================

def getCurrentTime():
        time_now = datetime.now()
        exportDate = str(datetime.date(time_now)).replace('-','')
        exportTime = str(datetime.time(time_now)).replace(':','')[0:4]
        return exportDate + '_' + exportTime

def parse_dates(series):
    """兼容 20140831 和 2014-08-31。"""
    text = series.astype(str).str.strip()
    compact = text.str.fullmatch(r"\d{8}")

    result = pd.Series(
        pd.NaT, index=series.index, dtype="datetime64[ns]"
    )
    result.loc[compact] = pd.to_datetime(
        text.loc[compact], format="%Y%m%d", errors="raise"
    )
    result.loc[~compact] = pd.to_datetime(
        text.loc[~compact], errors="raise"
    )
    return result


def load_daily_data():
    path = DATA_DIR / "user_balance_table.csv"

    columns = [
        "user_id",
        "report_date",
        "tBalance",
        "total_purchase_amt",
        "total_redeem_amt",
        "direct_purchase_amt",
        "share_amt",
        "consume_amt",
        "transfer_amt",
    ]

    raw = pd.read_csv(
        path,
        usecols=columns,
        dtype={"report_date": str},
    )
    raw["report_date"] = parse_dates(raw["report_date"])

    # 防止输入文件意外包含测试期数据。
    raw = raw.loc[raw["report_date"] <= TRAIN_END].copy()

    if raw.empty:
        raise ValueError("没有读取到训练数据。")

    if raw.duplicated(["user_id", "report_date"]).any():
        raise ValueError("存在重复的 用户-日期 记录，请先检查数据。")

    amount_columns = [
        c for c in columns
        if c not in {"user_id", "report_date"}
    ]
    raw[amount_columns] = raw[amount_columns].apply(
        pd.to_numeric, errors="raise"
    )

    if raw[amount_columns].isna().any().any():
        raise ValueError("关键金额字段存在缺失值，请检查输入文件。")

    if (raw[amount_columns] < 0).any().any():
        raise ValueError("金额字段存在负值，请检查输入文件。")

    daily = (
        raw.groupby("report_date")
        .agg(
            purchase=("total_purchase_amt", "sum"),
            redeem=("total_redeem_amt", "sum"),
            direct_purchase=("direct_purchase_amt", "sum"),
            share=("share_amt", "sum"),
            consume=("consume_amt", "sum"),
            transfer=("transfer_amt", "sum"),
            balance=("tBalance", "sum"),
            recorded_users=("user_id", "nunique"),
        )
        .sort_index()
    )

    # 某些用户没有记录是正常的；整个日期缺失则不能直接补成零。
    expected_index = pd.date_range(
        daily.index.min(), TRAIN_END, freq="D"
    )
    missing_dates = expected_index.difference(daily.index)
    if len(missing_dates):
        raise ValueError(
            "缺少整天数据，不能直接当成零："
            + ", ".join(d.strftime("%Y%m%d") for d in missing_dates)
        )

    daily = daily.reindex(expected_index)
    daily.index.name = "report_date"

    # 申购包含收益：这里只校验，不再追加收益。
    purchase_diff = (
        daily["purchase"]
        - daily["direct_purchase"]
        - daily["share"]
    ).abs()
    if purchase_diff.max() > 0:
        print("警告：部分日期的总申购不等于直接申购加收益，请核对。")

    daily = join_rate_table(
        daily,
        DATA_DIR / "mfd_day_share_interest.csv",
        prefix="yield",
    )
    daily = join_rate_table(
        daily,
        DATA_DIR / "mfd_bank_shibor.csv",
        prefix="shibor",
    )

    return daily.astype(float)


def join_rate_table(daily, path, prefix):
    if not path.exists():
        print(f"未找到 {path.name}，跳过该组利率特征。")
        return daily

    rates = pd.read_csv(path, dtype={"mfd_date": str})
    rates["mfd_date"] = parse_dates(rates["mfd_date"])
    rates = rates.loc[rates["mfd_date"] <= TRAIN_END].copy()

    if rates["mfd_date"].duplicated().any():
        raise ValueError(f"{path.name} 存在重复日期。")

    rates = rates.set_index("mfd_date").sort_index()
    rates = rates.apply(pd.to_numeric, errors="raise")
    rates.columns = [f"{prefix}_{c}" for c in rates.columns]

    # 在完整日历上向前填充，只用以前发布的值。
    # 不使用 bfill，避免把未来值填到过去。
    full_index = rates.index.union(daily.index).sort_values()
    rates = rates.reindex(full_index).ffill().reindex(daily.index)

    return daily.join(rates)


# ============================================================
# 预测时点特征
# ============================================================

def make_origin_features(daily, origin):
    """
    所有历史特征均截止于 origin。
    未来 30 天共用这些已知历史信息。
    """
    history = daily.loc[:origin]
    if len(history) < LOOKBACK:
        raise ValueError("历史长度不足。")

    features = {
        "origin_weekday": origin.dayofweek,
        "origin_month": origin.month,
        "origin_day": origin.day,
    }

    for target in TARGETS:
        series = history[target]

        # lag=0 表示预测时点当天，已经观测到。
        for lag in [0, 1, 6, 7, 13, 27]:
            features[f"{target}_lag_{lag}"] = np.log1p(
                series.iloc[-1 - lag]
            )

        for window in [7, 14, 28, 56]:
            tail = series.iloc[-window:]
            features[f"{target}_mean_{window}"] = np.log1p(
                tail.mean()
            )
            features[f"{target}_cv_{window}"] = (
                tail.std(ddof=0) / max(tail.mean(), 1.0)
            )

        recent = series.iloc[-14:].mean()
        previous = series.iloc[-28:-14].mean()
        features[f"{target}_trend_14"] = (
            recent / max(previous, 1.0)
        )

    for column in [
        "direct_purchase",
        "share",
        "consume",
        "transfer",
        "balance",
        "recorded_users",
    ]:
        series = history[column]
        features[f"{column}_last"] = np.log1p(series.iloc[-1])
        features[f"{column}_mean_7"] = np.log1p(
            series.iloc[-7:].mean()
        )
        features[f"{column}_mean_28"] = np.log1p(
            series.iloc[-28:].mean()
        )

    features["consume_share_28"] = (
        history["consume"].iloc[-28:].sum()
        / max(history["redeem"].iloc[-28:].sum(), 1.0)
    )
    features["income_share_28"] = (
        history["share"].iloc[-28:].sum()
        / max(history["purchase"].iloc[-28:].sum(), 1.0)
    )

    for column in daily.columns:
        if column.startswith(("yield_", "shibor_")):
            series = history[column]
            features[f"{column}_last"] = series.iloc[-1]
            features[f"{column}_mean_7"] = (
                series.iloc[-7:].mean()
            )
            features[f"{column}_change_7"] = (
                series.iloc[-1] - series.iloc[-8]
            )

    # 对每个星期几构造历史基线；不涉及预测期金额。
    weekday_baselines = {}

    for weekday in range(7):
        weekday_baselines[weekday] = {}

        for target in TARGETS:
            recent_28 = history[target].iloc[-28:]
            recent_56 = history[target].iloc[-56:]

            mean_28 = recent_28.loc[
                recent_28.index.dayofweek == weekday
            ].mean()
            mean_56 = recent_56.loc[
                recent_56.index.dayofweek == weekday
            ].mean()

            baseline = 0.7 * mean_28 + 0.3 * mean_56
            weekday_baselines[weekday][target] = max(
                float(baseline), 0.0
            )

    return features, weekday_baselines


def make_forecast_features(daily, origin):
    origin_features, baselines = make_origin_features(
        daily, origin
    )

    rows = []
    future_dates = pd.date_range(
        origin + pd.Timedelta(days=1),
        periods=HORIZON,
        freq="D",
    )

    for horizon, date in enumerate(future_dates, start=1):
        row = dict(origin_features)

        row.update({
            "horizon": horizon,
            "target_weekday": date.dayofweek,
            "target_month": date.month,
            "target_day": date.day,
            "target_is_weekend": int(date.dayofweek >= 5),
            "target_is_workday": int(is_workday(date.date())),
            "target_is_month_start": int(date.is_month_start),
            "target_is_month_end": int(date.is_month_end),
            "target_days_to_month_end": (
                date.days_in_month - date.day
            ),
            "target_is_first_week": int(date.day <= 7),
            "target_is_last_week": int(
                date.day > date.days_in_month - 7
            ),
        })

        for target in TARGETS:
            row[f"{target}_baseline"] = (
                baselines[date.dayofweek][target]
            )

        rows.append(row)

    return pd.DataFrame(rows, index=future_dates)


# ============================================================
# 训练样本
# ============================================================

def make_training_samples(daily):
    """
    每个历史预测时点生成 30 个样本。
    样本包含 origin 和 label_date，供回测严格截断。

    这些样本共享历史，不能理解为独立的新观测。
    """
    feature_blocks = []
    label_blocks = []
    metadata_blocks = []

    origins = daily.index[
        LOOKBACK - 1: len(daily) - HORIZON
    ]

    for origin in origins:
        features = make_forecast_features(daily, origin)
        truth = daily.loc[features.index, TARGETS]

        labels = pd.DataFrame(index=features.index)
        for target in TARGETS:
            # 学习相对于周周期基线的对数修正。
            labels[target] = (
                np.log1p(truth[target])
                - np.log1p(features[f"{target}_baseline"])
            )

        metadata = pd.DataFrame({
            "origin": origin,
            "label_date": features.index,
        }, index=features.index)

        feature_blocks.append(features)
        label_blocks.append(labels)
        metadata_blocks.append(metadata)

    if not feature_blocks:
        raise ValueError("数据不足，无法构造训练样本。")

    X = pd.concat(feature_blocks, ignore_index=True)
    y = pd.concat(label_blocks, ignore_index=True)
    metadata = pd.concat(metadata_blocks, ignore_index=True)

    return X, y, metadata


def fit_models(X, y):
    models = {}
    for target in TARGETS:
        model = lgb.LGBMRegressor(**MODEL_PARAMS)
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

        prediction[target] = (
            (1.0 - alpha) * baseline
            + alpha * model_values
        )

    return prediction

# ============================================================
# 回测指标
# ============================================================

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

        relative_error = (
            np.abs(predicted[nonzero] - actual[nonzero])
            / actual[nonzero]
        )

        result[f"{target}_mape"] = float(relative_error.mean())
        result[f"{target}_over_30pct_days"] = int(
            (relative_error > 0.30).sum()
        )
        result[f"{target}_zero_actual_days"] = int(
            (~nonzero).sum()
        )

    # 只是参照官方权重构造的误差代理，越小越好。
    # 不是官方积分。
    result["weighted_mape_proxy"] = (
        0.45 * result["purchase_mape"]
        + 0.55 * result["redeem_mape"]
    )
    return result


def run_backtests(daily, X, y, metadata):
    metrics = []
    prediction_blocks = []

    for origin in BACKTEST_ORIGINS:
        forecast_end = origin + pd.Timedelta(days=HORIZON)
        if forecast_end > daily.index.max():
            continue

        # 关键：训练样本的全部标签必须已在 origin 时点可见。
        train_mask = metadata["label_date"] <= origin

        models = fit_models(
            X.loc[train_mask],
            y.loc[train_mask],
        )

        features = make_forecast_features(daily, origin)
        truth = daily.loc[features.index, TARGETS]
        prediction = predict_amounts(models, features)

        baseline = pd.DataFrame({
            target: features[f"{target}_baseline"]
            for target in TARGETS
        }, index=features.index)

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

        detail = truth.add_prefix("actual_").join(
            prediction.add_prefix("predicted_")
        )
        detail = detail.join(baseline.add_prefix("baseline_"))
        detail["origin"] = origin.strftime("%Y-%m-%d")
        prediction_blocks.append(detail)

    if not metrics:
        raise ValueError("没有可用回测窗口。")

    metrics = pd.DataFrame(metrics)
    metrics.to_csv(
        OUTPUT_DIR / "backtest_metrics.csv",
        index=False,
    )

    pd.concat(prediction_blocks).to_csv(
        OUTPUT_DIR / "backtest_predictions.csv",
        index_label="report_date",
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


# ============================================================
# 最终训练与提交
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily = load_daily_data()
    print(
        f"数据范围：{daily.index.min():%Y-%m-%d} "
        f"至 {daily.index.max():%Y-%m-%d}，"
        f"共 {len(daily)} 天"
    )

    X, y, metadata = make_training_samples(daily)
    print(f"训练样本形状：{X.shape}")

    run_backtests(daily, X, y, metadata)

    # 截止 8 月 31 日，使用全部已知标签重新训练。
    final_models = fit_models(X, y)

    future_features = make_forecast_features(daily, TRAIN_END)
    prediction = predict_amounts(final_models, future_features)

    submission = pd.DataFrame({
        "report_date": prediction.index.strftime("%Y%m%d"),
        "purchase": np.rint(prediction["purchase"]).astype("int64"),
        "redeem": np.rint(prediction["redeem"]).astype("int64"),
    }).reset_index(drop=True)

    expected_dates = pd.date_range(
        "2014-09-01", "2014-09-30", freq="D"
    ).strftime("%Y%m%d").tolist()

    assert submission["report_date"].tolist() == expected_dates
    assert len(submission) == 30
    assert not submission.isna().any().any()
    assert (submission[TARGETS] >= 0).all().all()

    submission_path = OUTPUT_DIR / "tc_comp_predict_table.csv"
    submission.to_csv(
        submission_path,
        index=False,
        header=SUBMISSION_HEADER,
    )

    # 另存带表头的版本，方便人工检查。
    submission.to_csv(
        OUTPUT_DIR / "prediction_preview.csv",
        index=False,
    )

    for target, model in final_models.items():
        model.booster_.save_model(
            str(OUTPUT_DIR / f"{target}_lightgbm.txt")
        )

        importance = pd.DataFrame({
            "feature": X.columns,
            "gain": model.booster_.feature_importance(
                importance_type="gain"
            ),
        }).sort_values("gain", ascending=False)

        importance.to_csv(
            OUTPUT_DIR / f"{target}_feature_importance.csv",
            index=False,
        )

    config = {
        "train_end": TRAIN_END.strftime("%Y-%m-%d"),
        "horizon": HORIZON,
        "lookback": LOOKBACK,
        "model_params": MODEL_PARAMS,
        "feature_columns": X.columns.tolist(),
    }
    with open(
        OUTPUT_DIR / "config.json", "w", encoding="utf-8"
    ) as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    print(f"\n提交文件已生成：{submission_path}")
    print("\n预测前 5 行，金额单位为分：")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
