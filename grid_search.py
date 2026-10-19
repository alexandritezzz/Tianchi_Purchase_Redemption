"""按时间顺序回测的网格调参：python grid_search.py。"""

import argparse
from datetime import datetime
from itertools import product
import json
from pathlib import Path
import pandas as pd
from data_processing import DATA_DIR, TRAIN_END, load_daily_data, make_training_samples
from forecast import BACKTEST_ORIGINS, MODEL_PARAMS, run_backtests

# 修改候选列表即可调整默认搜索范围；所有组合使用相同的月度回测窗口。
PARAM_GRID = {
    "n_estimators": [150, 300],
    "learning_rate": [0.03, 0.05],
    "num_leaves": [7, 10],
}


def parameter_combinations(grid):
    if not isinstance(grid, dict) or not grid:
        raise ValueError("参数网格必须是非空 JSON 对象。")
    for name, values in grid.items():
        if name not in MODEL_PARAMS:
            raise ValueError(f"不支持的参数：{name}；可选参数：{list(MODEL_PARAMS)}")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} 的候选值必须是非空列表。")
    return [dict(zip(grid, values)) for values in product(*grid.values())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, help="自定义参数网格 JSON 文件")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, help="输出目录，必须尚不存在")
    args = parser.parse_args()

    grid = PARAM_GRID
    if args.grid is not None:
        grid = json.loads(args.grid.read_text(encoding="utf-8-sig"))
    combinations = parameter_combinations(grid)
    output_dir = args.output_dir or Path(
        f"grid_search_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "param_grid": grid,
        "base_model_params": MODEL_PARAMS,
        "train_end": str(TRAIN_END.date()),
        "backtest_origins": [str(date.date()) for date in BACKTEST_ORIGINS],
        "alpha": 0.25,
        "selection_metric": "weighted_mape_proxy",
    }
    (output_dir / "search_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 数据读取、特征工程只执行一次，所有参数组合复用相同样本。
    print(f"共 {len(combinations)} 组参数；输出目录：{output_dir}")
    daily = load_daily_data(data_dir=args.data_dir, train_end=TRAIN_END)
    X, y, metadata = make_training_samples(daily)
    results = []
    best_score = float("inf")

    for trial, overrides in enumerate(combinations, start=1):
        params = {**MODEL_PARAMS, **overrides}
        trial_dir = output_dir / f"trial_{trial:03d}"
        trial_dir.mkdir()
        (trial_dir / "model_params.json").write_text(
            json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[{trial}/{len(combinations)}] {overrides}")
        metrics = run_backtests(
            daily, X, y, metadata, model_params=params, output_dir=trial_dir
        )
        columns = ["purchase_mape", "redeem_mape", "weighted_mape_proxy"]
        scores = metrics.loc[metrics["model"] == "LightGBM", columns].mean()
        baseline = metrics.loc[
            metrics["model"] == "weekday_baseline", "weighted_mape_proxy"
        ].mean()
        results.append(
            {
                "trial": trial,
                **params,
                **scores.to_dict(),
                "baseline_weighted_mape_proxy": baseline,
            }
        )
        # 每轮保存，长任务中断后仍可查看已完成的结果。
        summary = pd.DataFrame(results).sort_values(
            ["weighted_mape_proxy", "trial"], kind="stable"
        )
        summary.to_csv(output_dir / "grid_search_results.csv", index=False)
        if scores["weighted_mape_proxy"] < best_score:
            best_score = float(scores["weighted_mape_proxy"])
            (output_dir / "best_params.json").write_text(
                json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    print("\n搜索完成，平均加权误差越小越好（不是官方积分）：")
    print(summary[["trial", *grid, *columns]].head(10).to_string(index=False))
    print(f"最佳参数：{output_dir / 'best_params.json'}")


if __name__ == "__main__":
    main()
