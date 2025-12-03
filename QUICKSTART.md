# CS2 Major Prediction System - 快速开始指南

## 第一次使用

### 1. 安装依赖

```bash
# Python依赖
pip install pandas numpy

# Buchholz模拟器
git clone https://github.com/claabs/cs-buchholz-simulator
cd cs-buchholz-simulator
npm install
cd ..
```

### 2. 准备数据

将你的历史比赛数据放入 `data/cs2_cleaned_matches.csv`

格式：
```
date,team1,score1,score2,team2,tournament,format
2025-11-21,FaZe,2,1,Legacy,BLAST Premier Fall Finals,BO3
...
```

### 3. 运行预测流程

```bash
# 步骤1: 计算ELO评分
cd scripts
python 1_calculate_ratings.py

# 步骤2: 准备Buchholz配置
python 2_prepare_buchholz.py

# 步骤3: 按照提示，手动复制配置文件到Buchholz模拟器
#        然后启动Buchholz模拟器运行10万次模拟
cd ../../cs-buchholz-simulator
npm start
# 在浏览器中运行模拟，下载结果保存为 output/simulation_results.txt

# 步骤4: 优化Pick'Em
cd ../cs2_major_prediction_system/scripts
python 3_optimize_pickem.py
```

## 输出文件

- `output/team_ratings.json` - 队伍ELO评分
- `output/simulation_results.txt` - Buchholz模拟结果
- `output/pickem_recommendation.json` - Pick'Em推荐

## 2025 Major Stage 1 预测结果

**期望得分: 4.363 / 10**

- **3-0**: PARIVISION, Imperial
- **3-1/3-2**: FaZe, Legacy, GamerLegion, Lynn Vision, Ninjas in Pyjamas, B8
- **0-3**: Rare Atom, The Huns

详细概率见 `output/pickem_recommendation.json`

---

📖 完整文档请查看 [README.md](README.md)
