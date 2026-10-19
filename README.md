# 项目说明

1. 数据集：天池比赛-Purchase Redemption

2. 拉取仓库后需解压 `Purchase Redemption Data.zip` 并将其中的文件全部放到根目录（需要这一步额外操作的原因是，一个csv超过100MB，无法直接上传到Github）

3. 赛事要求和笔记：`天池.md` 

# 安装依赖

- Python 版本：3.12
- 安装所需的包：
```bash
pip install -r requirements.txt
```

# 脚本说明

- `data_processing.py`
    - 读取和校验数据，按日汇总，合并利率。
    - 构造特征及训练样本，定义共享的数据配置。

- `forecast.py`
    - 调用数据处理函数，完成模型训练、月度回测、预测及结果导出。
    - 在项目根目录运行 `python forecast.py`，结果保存到 `output_时间戳` 目录。无需单独运行 `data_processing.py`。

- `python grid_search.py`。
    - 网格调参工具，默认搜索脚本中 `PARAM_GRID` 定义的 8 组组合，只回测，不生成提交文件，也不会更新 `forecast.py` 中的参数组合
    - 输出目录为 `grid_search_时间戳`，每组的参数及回测详情保存在 `trial_编号` 中；汇总结果为 `grid_search_results.csv`，最佳完整模型参数为 `best_params.json`。
    - 数据处理只执行一次，每组参数均使用相同的时间回测窗口，按平均 `weighted_mape_proxy` 从小到大排序。
    - 运行：在项目根目录直接运行；也可创建 JSON 文件，例如 `{"n_estimators": [150, 300], "learning_rate": [0.03, 0.05]}`，然后运行：`python grid_search.py --grid params.json --output-dir grid_results`。
    - 回测月份属于验证集，建议额外保留未参与选参的时间窗口用于评估最终效果。

