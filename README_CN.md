# CS2 Major Swiss System Prediction

基于历史比赛数据和ELO算法的CS2瑞士轮预测系统。

## 功能特点

- **ELO评分系统**: 考虑对手强度、时间衰减、自适应K因子
- **Monte Carlo模拟**: 10万次瑞士轮模拟，使用Buchholz配对算法
- **Pick'Em优化**: 暴力搜索1000万+组合，找到最优预测
- **断点续传**: 支持长时间运行中途中断后恢复
- **通用设计**: 适用于任何CS2瑞士轮赛事（8/11/16/24队）

## 快速开始

### 前置要求

```bash
# Python 3.11+
pip install pandas
```

### 使用流程

本系统使用纯Python实现的Buchholz配对算法，无需额外依赖。

#### 1. 准备数据

**`data/cs2_cleaned_matches.csv`** - 历史比赛记录，必须严格按照我以下规定的格式。

#### 1. 准备数据

- 列1: date - 比赛日期 (YYYY-MM-DD)
- 列2: team1 - 队伍1名称
- 列3: score1 - 队伍1得分
- 列4: score2 - 队伍2得分
- 列5: team2 - 队伍2名称
- 列6: tournament - 赛事名称
- 列7: format - 比赛格式 (bo1/bo3/bo5)

示例：

```
2025-11-21,Team A,2,1,Team B,Example Tournament,bo3
2025-11-20,Team C,16,14,Team D,Example League,bo1
2025-11-19,Team A,3,0,Team C,Example Cup,bo5
```

**`战队属性.txt`** - HLTV评分（可选）

格式说明：CSV格式，有表头，包含以下列

- team - 队伍名称（需与代码中TEAMS一致）
- Maps - 地图数（用于样本量置信度调整）
- KD_diff - KD差值（可选）
- KD - KD比率（可选）
- Rating - HLTV评分（核心数据，通常在0.9-1.1之间）

示例：

```
team,Maps,KD_diff,KD,Rating
Team A,120,+250,1.08,1.09
Team B,95,+180,1.06,1.07
Team C,110,-50,1.02,1.05
```

提示：可从 HLTV.org 获取最新评分

#### 2. 配置代码

编辑 **`cs2_swiss_predictor.py`** 顶部的配置区域：

```python
# 参赛战队列表（16支队伍）
TEAMS = [
    "FURIA", "Natus Vincere", "Vitality", "FaZe", "Falcons", "B8",
    "The MongolZ", "Imperial", "MOUZ", "PARIVISION", "Spirit", "Liquid",
    "G2", "Passion UA", "paiN", "3DMAX"
]

# 第一轮对局配对（8场BO1）
ROUND1_MATCHUPS = [
    ("FURIA", "Natus Vincere"),
    ("Vitality", "FaZe"),
    ("Falcons", "B8"),
    ("The MongolZ", "Imperial"),
    ("MOUZ", "PARIVISION"),
    ("Spirit", "Liquid"),
    ("G2", "Passion UA"),
    ("paiN", "3DMAX")
]

# 外部数据文件路径
MATCHES_FILE = 'cs2_cleaned_matches.csv'
TEAM_RATINGS_FILE = '战队属性.txt'
```

**重要**: ROUND1_MATCHUPS 的顺序决定Buchholz配对系统的种子位置！

#### 3. 运行预测

```bash
python cs2_swiss_predictor.py
```

程序会自动完成：

1. 加载历史数据并计算ELO评分
2. 运行10万次瑞士轮Monte Carlo模拟
3. 暴力搜索最优Pick'Em组合（约20小时）
4. 生成结果报告

输出文件：

- `prediction_results.json` - 完整预测结果（JSON格式）
- `optimized_report.txt` - 易读的推荐报告
- `checkpoint_progress.json` - 进度断点（用于续传）
- `checkpoint_best.json` - 当前最优解备份

## 项目结构

```
cs2_major_prediction_system/
├── cs2_swiss_predictor.py           # 主程序（唯一需要运行的文件）
├── cs2_cleaned_matches.csv          # 历史比赛数据
├── 战队属性.txt                      # HLTV评分（可选）
├── prediction_results.json          # 输出：完整预测结果
├── optimized_report.txt             # 输出：易读报告
├── checkpoint_progress.json         # 输出：进度断点
└── checkpoint_best.json             # 输出：最优解备份
```

**计算公式:**

```python
expected_win = 1 / (1 + 10^((rating2 - rating1) / 400))
new_rating = old_rating + K * weight * (actual - expected)
```

### 归一化策略

## 核心算法

### 1. 自适应ELO系统

系统根据队伍的历史数据覆盖情况动态调整初始评分权重：

**初始化策略**：

- CSV对局 < 10场：外部Rating占70%权重（数据不足，主要依赖外部评分）
- CSV对局 10-20场：权重线性过渡（70% → 35%）
- CSV对局 20-30场：外部Rating占35%权重
- CSV对局 > 30场：外部Rating占20%权重（数据充足，历史对局为主）

**K因子自适应**：

- 对局 < 15场：K=50（快速调整）
- 对局 15-30场：K=40（平衡）
- 对局 > 30场：K=30（稳定收敛）

**其他因子**：

- 时间衰减：50天半衰期（旧比赛权重降低）
- 格式权重：BO1=1.0, BO3=1.2, BO5=1.5
- 样本量置信度：Maps数据用于贝叶斯调整外部Rating

**计算公式**：

```python
expected = 1 / (1 + 10^((elo2 - elo1) / 400))
new_elo = old_elo + K * format_weight * time_weight * (actual - expected)
```

### 2. Buchholz瑞士轮系统（Valve官方规则）

**实现标准**：严格遵循 [Valve官方规则](https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md)

**配对规则**：

- **Round 1**: 固定配对表（1v9, 2v10, 3v11...）
- **Round 2-3**: 最高种子 vs 最低种子（避免重复对阵）
- **Round 4-5**: 使用15优先级配对表（避免重复对阵）

**种子计算** (Mid-stage Seed)：

1. 当前战绩（W-L）
2. Difficulty Score（对手胜负差之和）
3. 初始种子

**Difficulty Score（Buchholz）**：

```python
difficulty = sum(对手胜场数 - 对手负场数)
```

**15优先级配对表** (Round 4-5)：

```
Priority 1:  1v6, 2v5, 3v4
Priority 2:  1v6, 2v4, 3v5
Priority 3:  1v5, 2v6, 3v4
...
Priority 15: 1v2, 3v4, 5v6
```

系统会选择第一个**不产生重复对阵**的优先级进行配对。

### Pick'Em规则


| 类别    | 数量 | 说明                        |
| ------- | ---- | --------------------------- |
| 3-0     | 2支  | 预测恰好3-0晋级             |
| 3-1/3-2 | 6支  | 预测3-1或3-2晋级（不含3-0） |
| 0-3     | 2支  | 预测0-3淘汰                 |

**约束**: 每支队伍只能选择一次，共10支

**搜索空间**：

- advances组合（6队）：C(16,6) = 8,008
- 每个advances的子组合：
  * 3-0组合（2队）：C(10,2) = 45
  * 0-3组合（2队）：C(8,2) = 28
  * 子组合数：45 × 28 = 1,260
- 总计：8,008 × 1,260 = 10,090,080 组合

## 使用场景

### 场景1: 16支队伍的瑞士轮（标准Major）

```python
TEAMS = [
    "Team A", "Team B", "Team C", "Team D",
    "Team E", "Team F", "Team G", "Team H",
    "Team I", "Team J", "Team K", "Team L",
    "Team M", "Team N", "Team O", "Team P"
]

ROUND1_MATCHUPS = [
    ("Team A", "Team I"),  # Seed 1 vs Seed 9
    ("Team B", "Team J"),  # Seed 2 vs Seed 10
    ("Team C", "Team K"),
    ("Team D", "Team L"),
    ("Team E", "Team M"),
    ("Team F", "Team N"),
    ("Team G", "Team O"),
    ("Team H", "Team P")
]
```

### 场景2: 11支队伍的瑞士轮（Stage 2晋级赛）

```python
TEAMS = [
    "Seed 1", "Seed 2", "Seed 3", "Seed 4", "Seed 5",
    "Seed 6", "Seed 7", "Seed 8", "Seed 9", "Seed 10", "Seed 11"
]

ROUND1_MATCHUPS = [
    ("Seed 1", "Seed 6"),
    ("Seed 2", "Seed 7"),
    ("Seed 3", "Seed 8"),
    ("Seed 4", "Seed 9"),
    ("Seed 5", "Seed 10")
    # Seed 11轮空
]
```

### 场景3: 8支队伍的小型赛事

```python
TEAMS = [
    "Team A", "Team B", "Team C", "Team D",
    "Team E", "Team F", "Team G", "Team H"
]

ROUND1_MATCHUPS = [
    ("Team A", "Team E"),
    ("Team B", "Team F"),
    ("Team C", "Team G"),
    ("Team D", "Team H")
]
```

## 注意事项

### 数据要求

**推荐配置**：

- 历史比赛覆盖3-6个月
- 每支队伍至少10场比赛样本
- 使用最新的HLTV评分（可选）

**注意**：

- 无HLTV评分的队伍会用默认ELO=1000初始化
- CSV中找不到的队伍会给出警告

### 配置要点

**关键注意事项**：

- ROUND1_MATCHUPS的顺序决定Buchholz配对规则
- 确保所有参赛队伍都在TEAMS列表中
- 验证第一轮对阵是否与实际赛程一致

### 性能建议

**最佳实践**：

- 使用16核CPU可获得最佳性能
- 建议预留20小时完成完整搜索
- 可随时中断，程序会自动保存进度

## 多阶段赛事预测

如果赛事有多个阶段（如Stage 1 → Stage 2 → Stage 3）：

1. **更新比赛数据**：
   将前一阶段的实际比赛结果添加到 cs2_cleaned_matches.csv
2. **更新代码配置**：

   ```python
   TEAMS = ["Stage 2晋级的队伍名单"]
   ROUND1_MATCHUPS = [("新阶段的第一轮对阵")]
   ```

```

3. **重新运行**：
   ```bash
   python cs2_swiss_predictor.py
```

**关键**: 每个阶段都使用最新的历史数据，让ELO评分反映队伍最新状态

## 常见问题

**Q: ROUND1_MATCHUPS的顺序为什么重要？**

A: Buchholz配对系统依赖初始种子位置。第一轮对阵的顺序决定了每支队伍的初始种子，影响后续配对逻辑（高difficulty vs 低difficulty，相同difficulty按种子排序）。

**Q: 如果某个队伍没有历史数据怎么办？**

A: 系统会按以下优先级初始化：

1. 尝试从HLTV评分文件读取
2. 如果没有外部评分，使用默认ELO（1000）
3. 给出警告信息

**Q: 为什么优化需要20小时？**

A: 总搜索空间约1009万组合，即使使用布尔数组优化和16核并行，也需要较长时间完成完整搜索。您可以通过checkpoint随时中断并恢复。

**Q: Pick'Em成功率是什么意思？**

A: 表示在10万次模拟中，这个预测组合命中5个或以上的比例。这是Pick'Em奖励的最低标准。

**Q: 断点续传如何工作？**

A: 程序每200个组合保存一次checkpoint，记录当前进度。如果程序中断，重新运行会自动从checkpoint恢复继续计算。

## 后续改进计划

- [ ]  **剪枝优化**：实现L1/L2剪枝逻辑，预计可减少50-70%搜索空间
- [ ]  **自适应并行度**：根据CPU核心数自动调整worker数量
- [ ]  **增量更新**：支持仅重新计算变化部分的评分
- [ ]  **可视化界面**：添加Web界面展示预测结果和模拟过程

欢迎贡献代码或提出建议！

## 依赖项

- Python 3.11+
- pandas

## 致谢

- Buchholz配对算法参考了 [claabs/cs-buchholz-simulator](https://github.com/claabs/cs-buchholz-simulator) 项目
- 队伍评分数据来自 [HLTV.org](https://www.hltv.org/)
- 瑞士轮赛制规则参考Valve CS2 Major官方规则

## 许可证

MIT License
