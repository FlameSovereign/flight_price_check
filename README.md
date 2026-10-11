# flight_price_check · 回国商务舱价格监控 + 买/等预测

每天自动扫一遍 Google Flights，盯住几条“性价比路线”（亚特兰大直接出发、土航经伊斯坦布尔、经东京转机、先飞休斯顿再坐全日空），
把价格攒成时间序列，再用一个小模型回答一个问题：**这个日期的票，现在买，还是再等等？**

最新报告：[`reports/latest.md`](reports/latest.md)

## 为什么第一天就能训练模型

Google Flights 每次搜索的页面里，除了航班列表，还藏着两样东西：

1. 这条航线 / 日期 / 舱位的 **“常见价格区间”**（就是网页上 “Prices are currently typical” 那个判断）；
2. **过去约 60 天的每日最低价**（网页上的价格历史图）。

所以扫一次 ~350 个（策略 × 城市 × 出发日）组合，就直接拿到 ~2 万个“某天的价格 → 之后两周的价格”的样本，
不用先攒几个月数据。之后每天的定时任务再往每条序列后面续一天。

## 监控什么

全在 [`config/watchlist.toml`](config/watchlist.toml) 里，改完就生效：

| 策略 | 怎么搜 | 有 60 天历史？ |
|---|---|---|
| `atl-any` | ATL 出发，任意航司 | ✅ |
| `atl-tk` | ATL 出发，只看土耳其航空（经 IST） | ✅ |
| `atl-japan` | ATL 出发，只允许在美国/加拿大枢纽 + 东京转机，结果必须经 NRT/HND | ❌（只有当天价，靠每日累积） |
| `iah-ana` | IAH 出发，只看全日空；另查一张前一天的 ATL→IAH 经济舱当定位航段，计入总价 | ✅ |

目的地默认上海（PVG+SHA）和北京（PEK+PKX），一个城市的多个机场在同一次搜索里覆盖。
出发日默认是未来 10–270 天、每 6 天一个；也可以加“重点窗口”（例如春节）按天扫。

## 本地运行

需要 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
uv run flightcheck search ATL PVG,SHA 2026-12-10              # 随手查一次，打印航班和价格历史
uv run flightcheck search ATL PVG,SHA 2026-12-10 --airlines TK
uv run flightcheck collect --limit 20                          # 跑清单里的前 20 个搜索
uv run flightcheck report                                      # 训练模型 + 生成 reports/latest.md
uv run flightcheck run                                         # collect + report
uv run pytest
```

## 每日自动运行

[`.github/workflows/daily.yml`](.github/workflows/daily.yml) 每天亚特兰大时间早上 7 点左右跑一次：
采集 → 训练 → 生成报告 → 把 `data/` 和 `reports/` 提交回仓库。
某个城市的总价新跌破 `[alerts]` 里设的提醒价时，会自动开一个 GitHub issue（GitHub 会发邮件/推送给你）。
也可以在 Actions 页面手动点 “Run workflow”。

一次大约 350–400 个请求、10–15 分钟。

## 数据文件

| 文件 | 内容 |
|---|---|
| `data/observations.csv` | 每次运行 × 每个搜索一行：最低价、航线、航班号、定位航段、Google 常见区间 |
| `data/history.csv` | 每条（策略:城市, 出发日）的每日最低价序列，模型的训练数据 |
| `data/itineraries/<日期>.csv.gz` | 每个搜索最便宜的 5 个行程 |
| `reports/latest.md` | 每日报告 |
| `reports/model.json` | 模型验证指标 |

## 模型

- **标签**：从某天起 14 天内，最低价会不会再便宜 ≥3%（都可在 `[model]` 里调）。
- **特征**：只用当天及以前的价格：离出发天数、相对 28 天最低/中位/最高、3/7/14 天涨跌、波动率、多久没变价、
  这个出发日比前后相邻日期贵多少、出发月份/星期、策略。
- **模型**：所有航线放一起训练的梯度提升树（scikit-learn `HistGradientBoostingClassifier`）。
- **对照规则**：价格比 28 天最低价贵 5% 以上就等。商务舱价格其实是在几个固定档位之间跳（便宜舱位开/关），
  “最近见过更便宜的价”本身就是“便宜舱位可能再开”的信号。
- **验证**：逐周回放（walk-forward）。最近几周每周一折，每折只用那周之前、标签已完全揭晓的数据训练，
  再把模型和规则都当成下单策略回放，和“立刻买”、事后最优比较平均省了多少。
- **建议列**：跟随过去几周回放里省得最多的策略（模型 / 规则 / 直接买）。离出发不足 21 天一律买。
  模型只有在回放里赢过规则，才会接管建议。

## 局限

- 数据来自 Google Flights 网页（非官方接口，参考了 [fast-flights](https://github.com/AWeirdDev/flights) 对查询格式的逆向）。
  Google 改版可能会让解析失效，测试和每日任务会第一时间报错。
- 价格是 Google 当时找到的最低公布价；不含里程票、不看具体舱位代码，个别结果可能是分开出票的组合。
- 同航线相邻日期价格高度相关，有效样本比表面数字少得多；模型会随着每天的数据积累变准，也可能证明“没法预测”——
  报告里的回测数字会如实告诉你是哪种。
