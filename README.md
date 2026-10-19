# 说明

1. 数据集：天池比赛-Purchase Redemption

2. 拉取仓库后需解压 `Purchase Redemption.zip` 并将其中的文件放到根目录（原因是一个csv超过100MB，无法直接上传到Github）

3. 赛事要求和笔记：`天池.md` 

# 依赖

- 依赖文件：requirements_new.txt

- 安装：

```bash
pip install -r requirements_new.txt
```

# 脚本说明

- `data_processing.py`：读取和校验数据、按日汇总、合并利率、构造特征及训练样本；同时定义共享的数据配置。
- `forecast.py`：调用数据处理函数，完成模型训练、月度回测、预测及结果导出。

在项目根目录运行 `python forecast.py`，结果保存到 `output_时间戳` 目录。无需单独运行 `data_processing.py`。

网格调参运行 `python grid_search.py`，默认搜索脚本中 `PARAM_GRID` 定义的 8 组组合。数据处理只执行一次，每组参数均使用相同的时间回测窗口，按平均 `weighted_mape_proxy` 从小到大排序。

也可创建 JSON 文件，例如 `{"n_estimators": [150, 300], "learning_rate": [0.03, 0.05]}`，然后运行：

```bash
python grid_search.py --grid params.json --output-dir grid_results
```

输出目录必须尚不存在。每组的参数及回测详情保存在 `trial_编号` 中，汇总结果为 `grid_search_results.csv`，最佳完整模型参数为 `best_params.json`。将最佳参数用于 `forecast.py` 的 `MODEL_PARAMS` 后，运行 `python forecast.py` 生成正式预测。调参脚本本身只回测，不生成提交文件。

回测月份用于选参后属于验证集；建议额外保留未参与选参的时间窗口评估最终效果。

