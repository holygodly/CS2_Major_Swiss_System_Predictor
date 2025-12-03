# 数据格式说明

本文档详细说明所有输入数据文件的格式要求。

## 1. 比赛历史数据 (`data/cs2_cleaned_matches.csv`)

### 格式要求

CSV文件，**必须包含表头**，各列说明：

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `date` | 日期 | 比赛日期，格式：YYYY-MM-DD | `2025-11-21` |
| `team1` | 文本 | 队伍1名称 | `Team A` |
| `score1` | 整数 | 队伍1得分（地图数或回合数） | `2` 或 `16` |
| `score2` | 整数 | 队伍2得分 | `1` 或 `14` |
| `team2` | 文本 | 队伍2名称 | `Team B` |
| `tournament` | 文本 | 赛事名称 | `IEM Katowice 2025` |
| `format` | 文本 | 比赛格式：bo1/bo3/bo5 | `bo3` |

### 示例文件

```csv
date,team1,score1,score2,team2,tournament,format
2025-11-21,Team A,2,1,Team B,Example Tournament,bo3
2025-11-20,Team C,16,14,Team D,Example League,bo1
2025-11-19,Team A,3,0,Team C,Example Championship,bo5
2025-11-18,Team B,13,16,Team D,Example Cup,bo1
2025-11-17,Team E,2,0,Team F,Example Masters,bo3
```

### 注意事项

✅ **正确做法：**
- 最新的比赛放在文件前面（降序排列）
- 每支参赛队伍至少有10场历史记录
- 覆盖3-6个月的比赛数据
- BO3/BO5记录的是地图数（如2-1），BO1记录回合数（如16-14）

❌ **常见错误：**
- 缺少表头行
- 日期格式错误（应为YYYY-MM-DD）
- 队伍名称不一致（如"FaZe"和"FaZe Clan"）
- format拼写错误（必须是bo1/bo3/bo5小写）

### 数据来源建议

- [HLTV.org](https://www.hltv.org/results) - 最全面的CS2比赛数据
- [Liquipedia](https://liquipedia.net/counterstrike/) - 赛事总结
- 赛事官方网站

---

## 2. HLTV评分 (`data/hltv_ratings.txt`)

### 格式要求

CSV文件，**必须包含表头**，各列说明：

| 列名 | 类型 | 必需 | 说明 | 示例 |
|------|------|------|------|------|
| `team` | 文本 | ✅ | 队伍名称（需与config.json一致） | `Team A` |
| `Maps` | 整数 | ❌ | 统计地图数（仅供参考） | `120` |
| `KD_diff` | 文本 | ❌ | K-D差值（仅供参考） | `+250` |
| `KD` | 小数 | ❌ | KD比率（仅供参考） | `1.08` |
| `Rating` | 小数 | ✅ | **HLTV评分**（核心数据） | `1.09` |

### 示例文件

```csv
team,Maps,KD_diff,KD,Rating
Team A,120,+250,1.08,1.09
Team B,95,+180,1.06,1.07
Team C,110,-50,1.02,1.05
Team D,88,+120,1.05,1.06
Team E,75,-80,0.99,1.02
Team F,92,+90,1.03,1.04
```

### 注意事项

✅ **正确做法：**
- Rating是核心数据，必须准确
- 队伍名称必须与`config.json`完全一致
- Rating通常在0.9-1.1之间
- 使用最新的HLTV评分（距离比赛越近越好）

❌ **常见错误：**
- 队伍名称不匹配
- Rating数据过旧（3个月以上）
- Rating超出合理范围（如0.5或1.5）

### 数据来源

访问 [HLTV Team Rankings](https://www.hltv.org/stats/teams)：
1. 点击队伍名称
2. 查看"Rating 2.1"数据
3. 记录最新的Rating值

### 如果没有HLTV数据？

**不用担心！** 系统会自动：
- 使用默认ELO初始值（1000）
- 基于历史比赛数据动态调整
- 给出警告信息

---

## 3. 赛事配置 (`config.json`)

### 格式要求

JSON文件，包含以下字段：

```json
{
  "event_name": "赛事名称",
  "teams": ["队伍列表"],
  "round1_matchups": [["队伍1", "队伍2"]],
  "swiss_rules": {
    "wins_for_qualification": 3,
    "losses_for_elimination": 3
  }
}
```

### 字段说明

#### `event_name` (字符串)
赛事名称，用于标识和输出文件命名。

#### `teams` (数组)
参赛队伍完整列表。
- **顺序无关紧要**（系统会根据round1_matchups重新排序）
- 队伍名称必须与数据文件一致

#### `round1_matchups` (数组)
第一轮对阵安排，**顺序非常重要**！

**Buchholz配对规则：**
```
位置1 vs 位置9
位置2 vs 位置10
位置3 vs 位置11
...
```

**示例：16队瑞士轮**
```json
"round1_matchups": [
  ["Team A", "Team I"],  // 位置1-9
  ["Team B", "Team J"],  // 位置2-10
  ["Team C", "Team K"],  // 位置3-11
  ["Team D", "Team L"],  // 位置4-12
  ["Team E", "Team M"],  // 位置5-13
  ["Team F", "Team N"],  // 位置6-14
  ["Team G", "Team O"],  // 位置7-15
  ["Team H", "Team P"]   // 位置8-16
]
```

#### `swiss_rules` (对象)
瑞士轮规则配置。
- `wins_for_qualification`: 晋级所需胜场（通常为3）
- `losses_for_elimination`: 淘汰所需败场（通常为3）

### 完整示例

#### 16队标准赛制
```json
{
  "event_name": "IEM Katowice 2025",
  "teams": [
    "Vitality", "FaZe", "MOUZ", "Spirit",
    "G2", "Natus Vincere", "Liquid", "Astralis",
    "FURIA", "Heroic", "Cloud9", "BIG",
    "Imperial", "ENCE", "Monte", "GamerLegion"
  ],
  "round1_matchups": [
    ["Vitality", "FURIA"],
    ["FaZe", "Heroic"],
    ["MOUZ", "Cloud9"],
    ["Spirit", "BIG"],
    ["G2", "Imperial"],
    ["Natus Vincere", "ENCE"],
    ["Liquid", "Monte"],
    ["Astralis", "GamerLegion"]
  ],
  "swiss_rules": {
    "wins_for_qualification": 3,
    "losses_for_elimination": 3
  }
}
```

#### 11队晋级赛制
```json
{
  "event_name": "Major Stage 2",
  "teams": [
    "Seed 1", "Seed 2", "Seed 3", "Seed 4", "Seed 5",
    "Seed 6", "Seed 7", "Seed 8", "Seed 9", "Seed 10", "Seed 11"
  ],
  "round1_matchups": [
    ["Seed 1", "Seed 6"],
    ["Seed 2", "Seed 7"],
    ["Seed 3", "Seed 8"],
    ["Seed 4", "Seed 9"],
    ["Seed 5", "Seed 10"]
    // Seed 11轮空
  ],
  "swiss_rules": {
    "wins_for_qualification": 3,
    "losses_for_elimination": 3
  }
}
```

---

## 快速检查清单

在运行预测前，请确认：

### 数据文件
- [ ] `cs2_cleaned_matches.csv` 包含表头
- [ ] 日期格式正确（YYYY-MM-DD）
- [ ] 队伍名称一致（无拼写错误）
- [ ] 比赛格式正确（bo1/bo3/bo5）

### HLTV评分
- [ ] `hltv_ratings.txt` 包含表头
- [ ] Rating列存在且有数据
- [ ] 队伍名称与config.json一致
- [ ] Rating值在合理范围（0.9-1.1）

### 配置文件
- [ ] `config.json` 格式正确（JSON语法）
- [ ] teams列表包含所有参赛队伍
- [ ] round1_matchups与实际对阵一致
- [ ] round1_matchups顺序符合Buchholz规则

---

## 常见问题

### Q: 队伍名称必须用英文吗？
A: 不必须。可以使用任何UTF-8字符（中文、日文等），但要确保所有文件中的名称完全一致。

### Q: 如果某个队伍在CSV中没有数据怎么办？
A: 系统会使用默认ELO（1000）并给出警告。建议至少收集10场历史比赛数据。

### Q: HLTV评分是必需的吗？
A: 不是。没有HLTV评分也能运行，但有了会更准确（提供更好的初始ELO）。

### Q: round1_matchups的顺序真的那么重要吗？
A: **非常重要！** 顺序决定了Buchholz模拟器的队伍配对，错误的顺序会导致第一轮对阵不匹配。

### Q: 可以预测非Major赛事吗？
A: 当然！只要是瑞士轮格式，任何CS2赛事都适用。

---

**需要帮助？** 参考 `README.md` 中的完整使用说明。
