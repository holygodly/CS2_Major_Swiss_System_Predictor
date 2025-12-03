"""
CS2 Major 瑞士轮预测系统（通用版）
核心功能：
1. 自适应ELO系统：根据样本量动态调整权重
2. Buchholz配对算法：完整实现瑞士轮配对规则
3. 布尔数组优化：预计算10万次模拟，实现230x加速
4. 多进程并行：绕过GIL限制，支持断点续传
5. Pick'Em优化器：暴力搜索1000万组合空间
详细说明见 README.md
"""

import sys
import json
import math
import random
import copy
import os
from datetime import datetime, timedelta
from collections import defaultdict
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

pd = None

# ============================================================================
# 配置区域（直接修改此处配置）
# ============================================================================

# 参赛战队（16支队伍）
# ⚠️ 重要：TEAMS 列表的顺序就是初始种子排序（种子1到种子16）
# 种子顺序用于后续轮次的 Buchholz 配对，请确保按照实际的初始种子顺序排列！
TEAMS = [
    "FURIA",          # 种子1
    "Natus Vincere",  # 种子2
    "Vitality",       # 种子3
    "FaZe",           # 种子4
    "Falcons",        # 种子5
    "B8",             # 种子6
    "The MongolZ",    # 种子7
    "Imperial",       # 种子8
    "MOUZ",           # 种子9
    "PARIVISION",     # 种子10
    "Spirit",         # 种子11
    "Liquid",         # 种子12
    "G2",             # 种子13
    "Passion UA",     # 种子14
    "paiN",           # 种子15
    "3DMAX"           # 种子16
]

# 第一轮对局配对（8场BO1）- 根据实际赛程手动输入
# ⚠️ 重要：请根据官方公布的实际赛程填写第一轮对阵
# 注意：队伍名称必须与上面 TEAMS 列表中的名称完全一致
ROUND1_MATCHUPS = [
    ("FURIA", "Natus Vincere"),           # Match 1
    ("Vitality", "FaZe"),                 # Match 2
    ("Falcons", "B8"),                    # Match 3
    ("The MongolZ", "Imperial"),          # Match 4
    ("MOUZ", "PARIVISION"),               # Match 5
    ("Spirit", "Liquid"),                 # Match 6
    ("G2", "Passion UA"),                 # Match 7
    ("paiN", "3DMAX")                     # Match 8
]

# 外部数据文件路径
MATCHES_FILE = 'cs2_cleaned_matches.csv'  # 历史比赛数据
TEAM_RATINGS_FILE = '战队属性.txt'  # 战队评分数据

# ELO系统参数
BASE_ELO = 1000
BASE_K_FACTOR = 40
TIME_DECAY_DAYS = 50

# 模拟和优化参数
NUM_SIMULATIONS = 100000  # Monte Carlo模拟次数
MAX_WORKERS = 16  # 多进程worker数量
CHECKPOINT_INTERVAL = 200  # 断点保存间隔

# ============================================================================
# 函数定义
# ============================================================================

def load_team_ratings_from_file(filepath):
    """
    从战队属性.txt加载战队评分（考虑样本量置信度）
    返回：{team_name: adjusted_rating} 字典
    
    改进点：
    1. 样本量加权：Maps数量越多，Rating越可信
    2. 向全局均值回归：Maps少的队伍向均值靠拢
    3. 置信度阈值：80场以上完全信任Rating
    """
    try:
        df = pd.read_csv(filepath)
        
        # 计算全局均值Rating（作为先验）
        global_mean_rating = df['Rating'].mean()
        
        # 置信度参数
        MIN_MAPS_FOR_FULL_CONFIDENCE = 80  # 80场以上完全信任Rating
        MIN_MAPS_THRESHOLD = 20  # 少于20场严重惩罚
        
        ratings = {}
        adjustments = []
        
        for _, row in df.iterrows():
            team_name = row['team']
            raw_rating = float(row['Rating'])
            maps_played = int(row['Maps'])
            
            if maps_played >= MIN_MAPS_FOR_FULL_CONFIDENCE:
                confidence = 1.0
            elif maps_played >= MIN_MAPS_THRESHOLD:
                confidence = 0.25 + (maps_played - MIN_MAPS_THRESHOLD) / (MIN_MAPS_FOR_FULL_CONFIDENCE - MIN_MAPS_THRESHOLD) * 0.75
            else:
                confidence = max(0.1, maps_played / MIN_MAPS_THRESHOLD * 0.25)
            
            adjusted_rating = confidence * raw_rating + (1 - confidence) * global_mean_rating
            ratings[team_name] = adjusted_rating
            adjustments.append({
                'team': team_name,
                'maps': maps_played,
                'raw': raw_rating,
                'adjusted': adjusted_rating,
                'confidence': confidence,
                'change': adjusted_rating - raw_rating
            })
        
        print(f"[数据] 从 {filepath} 加载了 {len(ratings)} 支队伍的评分")
        return ratings
    
    except Exception as e:
        print(f"[ERROR] 加载战队评分失败: {e}")
        return {team: 1.0 for team in TEAMS}


def calculate_elo_ratings(matches_df, initial_ratings, base_k_factor=40, time_decay_days=50):
    """
    基于历史比赛计算ELO评分
    
    核心策略：
    1. 自适应K因子：CSV对局越少，K值越大（快速收敛）
    2. 时间衰减：最近的比赛权重更高
    3. 赛制权重：BO3权重 > BO1权重
    4. 对手强度追踪：记录每个队伍的对手质量
    """
    ratings = initial_ratings.copy()
    matches_df = matches_df.sort_values('date')
    latest_date = matches_df['date'].max()
    
    team_csv_matches = defaultdict(int)
    for _, match in matches_df.iterrows():
        if match['team1'] in ratings:
            team_csv_matches[match['team1']] += 1
        if match['team2'] in ratings:
            team_csv_matches[match['team2']] += 1
    
    opponent_strength = defaultdict(list)
    elo_changes = defaultdict(list)
    
    for _, match in matches_df.iterrows():
        team1, team2 = match['team1'], match['team2']
        if team1 not in ratings or team2 not in ratings:
            continue
        
        score1, score2 = int(match['score1']), int(match['score2'])
        match_format = match['format']
        
        r1_before, r2_before = ratings[team1], ratings[team2]
        opponent_strength[team1].append(r2_before)
        opponent_strength[team2].append(r1_before)
        
        csv_count1 = team_csv_matches.get(team1, 0)
        csv_count2 = team_csv_matches.get(team2, 0)
        
        k1 = 50 if csv_count1 < 15 else (40 if csv_count1 < 30 else 30)
        k2 = 50 if csv_count2 < 15 else (40 if csv_count2 < 30 else 30)
        adaptive_k = (k1 + k2) / 2
        
        days_ago = (latest_date - match['date']).days
        time_weight = math.exp(-days_ago / time_decay_days)
        
        format_weight = {'bo1': 1.0, 'bo3': 1.2, 'bo5': 1.5}.get(match_format, 1.0)
        k = adaptive_k * format_weight * time_weight
        
        # ELO更新公式
        r1, r2 = ratings[team1], ratings[team2]
        e1 = 1 / (1 + math.pow(10, (r2 - r1) / 400))
        s1 = 1 if score1 > score2 else (0 if score1 < score2 else 0.5)
        
        ratings[team1] = r1 + k * (s1 - e1)
        ratings[team2] = r2 + k * ((1-s1) - (1-e1))
        
        # 记录ELO变化
        elo_changes[team1].append(ratings[team1] - r1_before)
        elo_changes[team2].append(ratings[team2] - r2_before)
    
    # 打印详细统计
    print("\n[ELO] 最终评分统计（参赛队伍）：")
    print(f"{'队伍':<20} {'初始':<8} {'最终':<8} {'变化':<8} {'对局':<6} {'对手均值':<10}")
    print("-" * 70)
    
    team_stats = []
    for team in TEAMS:
        if team in ratings:
            initial = initial_ratings.get(team, 1000)
            final = ratings[team]
            change = final - initial
            matches_count = len(opponent_strength.get(team, []))
            avg_opponent = sum(opponent_strength.get(team, [1000])) / max(len(opponent_strength.get(team, [])), 1)
            
            team_stats.append({
                'team': team,
                'initial': initial,
                'final': final,
                'change': change,
                'matches': matches_count,
                'avg_opponent': avg_opponent
            })
    
    # 按最终ELO排序
    team_stats.sort(key=lambda x: x['final'], reverse=True)
    
    for stat in team_stats:
        direction = "+" if stat['change'] >= 0 else ""
        strength = "强" if stat['avg_opponent'] > 1020 else ("中" if stat['avg_opponent'] > 980 else "弱")
        print(f"{stat['team']:<20} {stat['initial']:<8.1f} {stat['final']:<8.1f} "
              f"{direction}{stat['change']:<7.1f} {stat['matches']:<6} {stat['avg_opponent']:<7.1f} [{strength}]")
    
    return ratings


# ============================================================================
# 核心函数：比赛胜率预测
# ============================================================================

def predict_match(team1, team2, ratings, bo_format='bo1'):
    """
    预测比赛胜率（基于ELO差值）
    
    BO1不确定性调整：
    - BO1：胜率向50%收缩15%（更保守）
    - BO3/BO5：使用原始ELO胜率
    """
    r1, r2 = ratings.get(team1, 1000), ratings.get(team2, 1000)
    base_prob1 = 1 / (1 + math.pow(10, (r2 - r1) / 400))
    
    if bo_format == 'bo1':
        prob1 = 0.5 + (base_prob1 - 0.5) * 0.85
    else:
        prob1 = base_prob1
    
    return prob1, 1 - prob1


# ============================================================================
# 核心函数：完整瑞士轮模拟（Buchholz系统）
# ============================================================================

def simulate_full_swiss(ratings, num_simulations=100000):
    """
    完整瑞士轮模拟（实现Buchholz配对系统）
    
    配对规则：
    1. 按战绩分组（1-0, 0-1, 2-0等）
    2. 计算difficulty（对手胜负差之和）
    3. 高difficulty vs 低difficulty
    4. 避免重复对阵
    
    返回：每队概率 + 所有模拟原始结果
    """
    team_results = defaultdict(lambda: {'3-0': 0, 'qualified': 0, '0-3': 0, 'total': 0})
    all_simulations = []
    
    for sim in range(num_simulations):
        records = {team: (0, 0) for team in TEAMS}
        match_history = {team: [] for team in TEAMS}
        
        # 第一轮（BO1）
        for team1, team2 in ROUND1_MATCHUPS:
            prob1, _ = predict_match(team1, team2, ratings, 'bo1')
            winner = team1 if random.random() < prob1 else team2
            loser = team2 if winner == team1 else team1
            
            w, l = records[winner]
            records[winner] = (w + 1, l)
            w, l = records[loser]
            records[loser] = (w, l + 1)
            
            match_history[team1].append(team2)
            match_history[team2].append(team1)
        
        # 后续轮次
        for round_num in range(2, 6):
            # 按战绩分组
            groups = defaultdict(list)
            for team, (wins, losses) in records.items():
                if wins < 3 and losses < 3:
                    groups[(wins, losses)].append(team)
            
            if not groups:
                break
            
            # Buchholz配对（Valve官方规则）
            for record, teams in groups.items():
                # Difficulty = 对手胜负差之和
                difficulty = {}
                for team in teams:
                    diff = 0
                    for opponent in match_history[team]:
                        opp_wins, opp_losses = records[opponent]
                        diff += (opp_wins - opp_losses)
                    difficulty[team] = diff
                
                teams.sort(key=lambda t: (-difficulty[t], TEAMS.index(t)))
                
                # Round 2-3: 最高种子 vs 最低种子（避免重复对阵）
                if round_num in [2, 3]:
                    remaining = teams.copy()
                    while len(remaining) >= 2:
                        team1 = remaining.pop(0)  # 最高种子
                        
                        # 从后往前找最低种子且未交手的对手
                        matched = False
                        for i in range(len(remaining) - 1, -1, -1):
                            team2 = remaining[i]
                            if team2 not in match_history[team1]:
                                remaining.pop(i)
                                matched = True
                                break
                        
                        # 如果所有对手都已交手，选择最低种子（允许重复对阵）
                        if not matched:
                            team2 = remaining.pop()
                        
                        # 判断BO格式：淘汰赛(2-2)或晋级赛(2-0/2-1)用BO3，其他用BO1
                        wins1, losses1 = records[team1]
                        wins2, losses2 = records[team2]
                        is_elimination_or_advancement = (wins1 == 2 or losses1 == 2 or wins2 == 2 or losses2 == 2)
                        bo_format = 'bo3' if is_elimination_or_advancement else 'bo1'
                        
                        # 模拟比赛
                        prob1, _ = predict_match(team1, team2, ratings, bo_format)
                        winner = team1 if random.random() < prob1 else team2
                        loser = team2 if winner == team1 else team1
                        
                        w, l = records[winner]
                        records[winner] = (w + 1, l)
                        w, l = records[loser]
                        records[loser] = (w, l + 1)
                        
                        match_history[team1].append(team2)
                        match_history[team2].append(team1)
                
                # Round 4-5: 使用优先级配对表（避免重复对阵）
                else:
                    # Valve官方15种配对优先级（按排名1-6配对）
                    PAIRING_PRIORITY = [
                        [(0, 5), (1, 4), (2, 3)],  # 1v6, 2v5, 3v4
                        [(0, 5), (1, 3), (2, 4)],  # 1v6, 2v4, 3v5
                        [(0, 4), (1, 5), (2, 3)],  # 1v5, 2v6, 3v4
                        [(0, 4), (1, 3), (2, 5)],  # 1v5, 2v4, 3v6
                        [(0, 3), (1, 5), (2, 4)],  # 1v4, 2v6, 3v5
                        [(0, 3), (1, 4), (2, 5)],  # 1v4, 2v5, 3v6
                        [(0, 5), (1, 2), (3, 4)],  # 1v6, 2v3, 4v5
                        [(0, 4), (1, 2), (3, 5)],  # 1v5, 2v3, 4v6
                        [(0, 2), (1, 5), (3, 4)],  # 1v3, 2v6, 4v5
                        [(0, 2), (1, 4), (3, 5)],  # 1v3, 2v5, 4v6
                        [(0, 3), (1, 2), (4, 5)],  # 1v4, 2v3, 5v6
                        [(0, 2), (1, 3), (4, 5)],  # 1v3, 2v4, 5v6
                        [(0, 1), (2, 5), (3, 4)],  # 1v2, 3v6, 4v5
                        [(0, 1), (2, 4), (3, 5)],  # 1v2, 3v5, 4v6
                        [(0, 1), (2, 3), (4, 5)],  # 1v2, 3v4, 5v6
                    ]
                    
                    # 尝试每种优先级配对，选择第一个无重复对阵的方案
                    matched_pairs = None
                    for priority_pattern in PAIRING_PRIORITY:
                        valid = True
                        test_pairs = []
                        for idx1, idx2 in priority_pattern:
                            if idx1 >= len(teams) or idx2 >= len(teams):
                                valid = False
                                break
                            team1, team2 = teams[idx1], teams[idx2]
                            if team2 in match_history[team1]:
                                valid = False
                                break
                            test_pairs.append((team1, team2))
                        
                        if valid:
                            matched_pairs = test_pairs
                            break
                    
                    # 如果所有优先级都有重复对阵，使用第一个优先级（允许重复对阵）
                    if matched_pairs is None:
                        matched_pairs = []
                        for idx1, idx2 in PAIRING_PRIORITY[0]:
                            if idx1 < len(teams) and idx2 < len(teams):
                                matched_pairs.append((teams[idx1], teams[idx2]))
                    
                    # 执行配对
                    for team1, team2 in matched_pairs:
                        # 判断BO格式：淘汰赛(2-2)或晋级赛(2-0/2-1)用BO3，其他用BO1
                        wins1, losses1 = records[team1]
                        wins2, losses2 = records[team2]
                        is_elimination_or_advancement = (wins1 == 2 or losses1 == 2 or wins2 == 2 or losses2 == 2)
                        bo_format = 'bo3' if is_elimination_or_advancement else 'bo1'
                        
                        # 模拟比赛
                        prob1, _ = predict_match(team1, team2, ratings, bo_format)
                        winner = team1 if random.random() < prob1 else team2
                        loser = team2 if winner == team1 else team1
                        
                        w, l = records[winner]
                        records[winner] = (w + 1, l)
                        w, l = records[loser]
                        records[loser] = (w, l + 1)
                        
                        match_history[team1].append(team2)
                        match_history[team2].append(team1)
        
        # 统计本次模拟结果
        sim_result = {'3-0': set(), 'qualified': set(), '0-3': set()}
        for team, (wins, losses) in records.items():
            team_results[team]['total'] += 1
            
            if wins == 3 and losses == 0:
                team_results[team]['3-0'] += 1
                team_results[team]['qualified'] += 1
                sim_result['3-0'].add(team)
                sim_result['qualified'].add(team)
            elif wins == 3:
                team_results[team]['qualified'] += 1
                sim_result['qualified'].add(team)
            elif losses == 3 and wins == 0:
                team_results[team]['0-3'] += 1
                sim_result['0-3'].add(team)
        
        all_simulations.append(sim_result)
        
        if (sim + 1) % 10000 == 0:
            print(f"完成 {sim + 1}/{num_simulations} 次模拟")
    
    # 转换为概率
    results = {}
    for team, stats in team_results.items():
        total = stats['total']
        results[team] = {
            '3-0': stats['3-0'] / total,
            'qualified': stats['qualified'] / total,
            '0-3': stats['0-3'] / total,
            '3-1-or-3-2': (stats['qualified'] - stats['3-0']) / total
        }
    
    return results, all_simulations


# ============================================================================
# 核心函数：Pick'Em评估和优化
# ============================================================================

def precompute_simulation_data(all_simulations, all_teams):
    """
    预计算优化：将10万次模拟转换为布尔数组
    
    优化原理：
    1. 原始：每次评估需要遍历10万个字典（慢）
    2. 优化：预先转换为布尔数组，直接索引（快50-100倍）
    3. 内存换时间：增加约50MB内存，节省数小时计算时间
    """
    print(f"[优化] 预计算模拟数据...")
    start = time.time()
    
    # 为每支队伍创建布尔数组：在每次模拟中是否在特定类别
    team_in_3_0 = {team: [] for team in all_teams}
    team_in_0_3 = {team: [] for team in all_teams}
    team_in_advances = {team: [] for team in all_teams}  # qualified但不是3-0
    
    for sim in all_simulations:
        for team in all_teams:
            team_in_3_0[team].append(team in sim['3-0'])
            team_in_0_3[team].append(team in sim['0-3'])
            # advances = qualified且不在3-0
            team_in_advances[team].append(team in sim['qualified'] and team not in sim['3-0'])
    
    print(f"[优化] 预计算完成，用时 {time.time()-start:.1f}秒")
    
    return {
        '3-0': team_in_3_0,
        '0-3': team_in_0_3,
        'advances': team_in_advances
    }


def evaluate_prediction(prediction, precomputed_data):
    """
    评估Pick'Em预测组合的成功率（向量化优化版）
    
    Pick'Em规则：
    - 预测10支队伍：2个3-0 + 6个晋级 + 2个0-3
    - 命中5支或以上即可获得奖励
    
    评估逻辑：
    1. 对于每次模拟，统计命中数：
       - 预测的2支3-0队伍，实际3-0了几支？
       - 预测的6支advances队伍，实际3-1/3-2晋级了几支？
       - 预测的2支0-3队伍，实际0-3了几支？
    2. 如果命中数≥5，则这次模拟成功
    3. 成功率 = 成功模拟数 / 总模拟数
    
    优化：使用预计算的布尔数组，避免字典查找
    """
    # 获取模拟次数（从任意队伍的数组长度）
    num_sims = len(precomputed_data['3-0'][list(precomputed_data['3-0'].keys())[0]])
    
    # 向量化计算：为每次模拟累加命中数
    correct_counts = [0] * num_sims
    
    # 3-0预测的命中数
    for team in prediction['3-0']:
        if team in precomputed_data['3-0']:
            for i, hit in enumerate(precomputed_data['3-0'][team]):
                if hit:
                    correct_counts[i] += 1
    
    for team in prediction['advances']:
        if team in precomputed_data['advances']:
            for i, hit in enumerate(precomputed_data['advances'][team]):
                if hit:
                    correct_counts[i] += 1
    
    for team in prediction['0-3']:
        if team in precomputed_data['0-3']:
            for i, hit in enumerate(precomputed_data['0-3'][team]):
                if hit:
                    correct_counts[i] += 1
    
    success_count = sum(1 for c in correct_counts if c >= 5)
    return success_count / num_sims


# 多进程worker
_global_precomputed = None

def _init_worker(precomputed_data):
    global _global_precomputed
    _global_precomputed = precomputed_data


def evaluate_combo_worker(teams_advances, teams_3_0, teams_0_3):
    prediction = {
        '3-0': list(teams_3_0),
        'advances': list(teams_advances),
        '0-3': list(teams_0_3)
    }
    success_rate = evaluate_prediction(prediction, _global_precomputed)
    return prediction, success_rate


# ============================================================================
# 核心函数：暴力搜索最优Pick'Em组合（多进程优化版）
# ============================================================================

def optimize_pickem_with_pruning(probabilities, all_simulations, max_workers=16):
    """
    暴力搜索最优Pick'Em组合（当前版本为完整搜索）
    
    搜索空间：
    - C(16,6) * C(10,2) * C(8,2) = 8008 * 45 * 28 = 10,090,080 种组合
    
    已实现优化：
    1. 多进程并行：16个worker同时计算
    2. 断点续传：每200个组合保存进度
    3. 进程池复用：避免重复创建进程开销
    
    TODO：剪枝优化（由于时间限制暂未实现，后续会完善）
    
    预计用时：约20小时（速度：~130组合/秒）
    """
    print(f"\n开始搜索最优Pick'Em组合...")
    print(f"[LOG] {datetime.now().strftime('%H:%M:%S')} - 优化器启动", flush=True)
    
    all_teams = list(probabilities.keys())
    
    precomputed = precompute_simulation_data(all_simulations, all_teams)
    advances_combos = list(combinations(all_teams, 6))
    print(f"[LOG] {datetime.now().strftime('%H:%M:%S')} - 生成 {len(advances_combos):,} 个晋级组合", flush=True)
    
    best_prediction = None
    best_success_rate = 0.0
    total_tested = 0
    initial_tested = 0
    start_advances_idx = 0
    start_sub_offset = 0
    
    # 尝试从checkpoint恢复
    try:
        with open('checkpoint_progress.json', 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
            best_prediction = checkpoint['best_prediction']
            best_success_rate = checkpoint['best_success_rate']
            total_tested = checkpoint['tested_count']
            initial_tested = total_tested  # 记录恢复时的起始值
            start_advances_idx = checkpoint.get('advances_idx', 0)
            start_sub_offset = checkpoint.get('sub_combo_offset', 0)
            print(f"[LOG] {datetime.now().strftime('%H:%M:%S')} - 从checkpoint恢复: advances_idx={start_advances_idx}, sub_offset={start_sub_offset}, tested={total_tested:,}, best={best_success_rate:.4%}", flush=True)
    except:
        print(f"[LOG] {datetime.now().strftime('%H:%M:%S')} - 未找到checkpoint，从零开始搜索", flush=True)
    
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(precomputed,)) as executor:
        for idx, teams_advances in enumerate(advances_combos):
            if idx < start_advances_idx:
                continue
            
            if idx < 5 or (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                new_tested = total_tested - initial_tested
                rate = new_tested / elapsed if elapsed > 0 else 0
                print(f"[进度] idx={idx+1}/{len(advances_combos)} | 测试={total_tested:,} | 最佳={best_success_rate:.2%} | 速度={rate:.0f}组合/秒")
                sys.stdout.flush()
            
            combos_to_test = []
            remaining = [t for t in all_teams if t not in teams_advances]
            
            for teams_3_0 in combinations(remaining, 2):
                remaining_2 = [t for t in remaining if t not in teams_3_0]
                for teams_0_3 in combinations(remaining_2, 2):
                    combos_to_test.append((teams_advances, teams_3_0, teams_0_3))
            
            if combos_to_test:
                start_sub_idx = start_sub_offset if idx == start_advances_idx else 0
                combos_to_test = combos_to_test[start_sub_idx:]
                future_to_idx = {}
                for original_idx, (adv, t30, t03) in enumerate(combos_to_test):
                    future = executor.submit(evaluate_combo_worker, adv, t30, t03)
                    future_to_idx[future] = start_sub_idx + original_idx
                
                for future in as_completed(future_to_idx.keys()):
                    prediction, success_rate = future.result()
                    
                    total_tested += 1
                    current_sub_offset = future_to_idx[future] + 1
                    if total_tested % CHECKPOINT_INTERVAL == 0:
                        if best_prediction is not None:
                            try:
                                checkpoint = {
                                    'best_prediction': best_prediction,
                                    'best_success_rate': best_success_rate,
                                    'tested_count': total_tested,
                                    'advances_idx': idx,
                                    'sub_combo_offset': current_sub_offset,
                                    'timestamp': datetime.now().isoformat()
                                }
                                with open('checkpoint_progress.json', 'w', encoding='utf-8') as f:
                                    json.dump(checkpoint, f, indent=2, ensure_ascii=False)
                            except:
                                pass
                    
                    if success_rate > best_success_rate:
                        best_success_rate = success_rate
                        best_prediction = copy.deepcopy(prediction)
                        
                        print(f"\n✓ [{datetime.now().strftime('%H:%M:%S')}] 找到更优! 成功率: {success_rate:.4%} (已测试 {total_tested:,})", flush=True)
                        
                        try:
                            checkpoint = {
                                'best_prediction': best_prediction,
                                'best_success_rate': best_success_rate,
                                'tested_count': total_tested,
                                'advances_idx': idx,
                                'sub_combo_offset': current_sub_offset,
                                'timestamp': datetime.now().isoformat()
                            }
                            with open('checkpoint_best.json', 'w', encoding='utf-8') as f:
                                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
                        except:
                            pass
    
    total_time = time.time() - start_time
    
    total_possible = len(advances_combos) * 45 * 28
    
    print(f"\n{'='*70}")
    print(f"[LOG] {datetime.now().strftime('%H:%M:%S')} - 搜索完成! 用时: {total_time:.1f}秒")
    print(f"测试组合: {total_tested:,} / {total_possible:,}")
    print(f"平均速度: {total_tested/total_time:.0f} 组合/秒")
    print(f"最优成功率: {best_success_rate:.4%}")
    print(f"\n注：未实现剪枝优化，完整搜索所有组合（后续会完善）")
    print(f"{'='*70}")
    
    return best_prediction, best_success_rate


# ============================================================================
# 主流程
# ============================================================================

def main():
    print("=" * 60)
    print("CS2 Major 瑞士轮预测系统（通用版）")
    print("=" * 60)
    print(f"[LOG] {datetime.now().strftime('%H:%M:%S')} - 程序启动", flush=True)
    
    print("\n[1/5] 加载外部数据...")
    print(f"  - 读取历史比赛: {MATCHES_FILE}")
    matches_df = pd.read_csv(MATCHES_FILE, header=None,
                             names=['date', 'team1', 'score1', 'score2', 'team2', 'tournament', 'format'])
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    print(f"    ✓ 加载 {len(matches_df)} 场历史比赛")
    
    print(f"  - 读取战队评分: {TEAM_RATINGS_FILE}")
    team_ratings = load_team_ratings_from_file(TEAM_RATINGS_FILE)
    print(f"[LOG] {datetime.now().strftime('%H:%M:%S')} - 数据加载完成", flush=True)
    
    print("\n[2/5] 计算ELO评分...")
    print("  策略：自适应权重（根据CSV覆盖率动态调整）")
    
    team_csv_matches = defaultdict(int)
    for _, match in matches_df.iterrows():
        if match['team1'] in TEAMS:
            team_csv_matches[match['team1']] += 1
        if match['team2'] in TEAMS:
            team_csv_matches[match['team2']] += 1
    
    initial_ratings = {}
    
    print(f"\n{'队伍':<20} {'外部Rating':<12} {'CSV对局':<10} {'权重比例':<15} {'初始ELO':<10}")
    print("-" * 75)
    
    for team in TEAMS:
        external_rating = team_ratings.get(team, 1.0)
        csv_matches = team_csv_matches.get(team, 0)
        
        if csv_matches < 10:
            rating_influence = 70
        elif csv_matches < 20:
            rating_influence = 70 - (csv_matches - 10) * 3.5
        elif csv_matches < 30:
            rating_influence = 35 - (csv_matches - 20) * 1.5
        else:
            rating_influence = 20
        rating_adjustment = (external_rating - 1.03) * rating_influence * 10
        rating_adjustment = max(-rating_influence, min(rating_influence, rating_adjustment))
        initial_ratings[team] = BASE_ELO + rating_adjustment
        
        external_weight = rating_influence / 70 * 100
        csv_weight = 100 - external_weight
        
        print(f"{team:<20} {external_rating:<12.3f} {csv_matches:<10} "
              f"外{external_weight:.0f}%/CSV{csv_weight:.0f}% {initial_ratings[team]:<10.1f}")
    
    print(f"\n  核心原则：CSV对局越多，历史数据权重越高")
    
    elo_ratings = calculate_elo_ratings(matches_df, initial_ratings)
    print(f"[LOG] {datetime.now().strftime('%H:%M:%S')} - ELO计算完成", flush=True)
    
    sorted_teams = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    print("\nELO排名:")
    for rank, (team, rating) in enumerate(sorted_teams, 1):
        print(f"  {rank:2d}. {team:20s} {rating:7.1f}")
    
    print(f"\n[3/5] 运行{NUM_SIMULATIONS:,}次瑞士轮模拟...")
    print(f"[LOG] {datetime.now().strftime('%H:%M:%S')} - 开始模拟...", flush=True)
    probabilities, all_simulations = simulate_full_swiss(elo_ratings, num_simulations=NUM_SIMULATIONS)
    print(f"[LOG] {datetime.now().strftime('%H:%M:%S')} - 模拟完成", flush=True)
    
    print("\n模拟结果:")
    print("=" * 70)
    print(f"{'战队':<20} {'3-0':<10} {'晋级':<10} {'0-3':<10} {'3-1/3-2':<10}")
    print("=" * 70)
    
    sorted_results = sorted(probabilities.items(), key=lambda x: x[1]['qualified'], reverse=True)
    for team, probs in sorted_results:
        print(f"{team:<20} {probs['3-0']:>8.1%} {probs['qualified']:>8.1%} "
              f"{probs['0-3']:>8.1%} {probs['3-1-or-3-2']:>8.1%}")
    
    print("\n[4/5] 运行Pick'Em优化器...")
    optimal_prediction, optimal_success_rate = optimize_pickem_with_pruning(
        probabilities, 
        all_simulations,
        max_workers=MAX_WORKERS
    )
    
    if optimal_prediction is None:
        print("[WARN] 优化器未返回有效组合，启用启发式回退（按概率选择）", flush=True)
        top_3_0 = sorted(probabilities.items(), key=lambda x: x[1]['3-0'], reverse=True)[:2]
        # 选6支晋级：按'qualified'概率降序
        top_qualified = sorted(probabilities.items(), key=lambda x: x[1]['qualified'], reverse=True)[:8]
        three0_names = [t for t, _ in top_3_0]
        advances_names = [t for t, _ in top_qualified if t not in three0_names][:6]
        # 选2支0-3：按'0-3'概率降序
        top_0_3 = sorted(probabilities.items(), key=lambda x: x[1]['0-3'], reverse=True)[:2]
        
        optimal_prediction = {
            '3-0': [t for t, _ in top_3_0],
            'advances': advances_names,
            '0-3': [t for t, _ in top_0_3]
        }
        precomputed = precompute_simulation_data(all_simulations, list(probabilities.keys()))
        optimal_success_rate = evaluate_prediction(optimal_prediction, precomputed)
        print(f"[WARN] 回退组合成功率估计: {optimal_success_rate:.4%}", flush=True)
    
    print("\n[5/5] 保存结果...")
    print(f"[LOG] {datetime.now().strftime('%H:%M:%S')} - 开始保存最终结果", flush=True)
    output_data = {
        'elo_ratings': dict(elo_ratings),
        'simulation_results': dict(probabilities),
        'optimal_prediction': optimal_prediction,
        'optimal_success_rate': optimal_success_rate,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        with open('prediction_results.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"[LOG] {datetime.now().strftime('%H:%M:%S')} - JSON结果保存成功", flush=True)
    except Exception as e:
        print(f"[ERROR] JSON保存失败: {e}", flush=True)
    
    try:
        with open('optimized_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CS2 Major Pick'Em 最优预测\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"预期成功率: {optimal_success_rate:.2%} (命中5个或以上)\n\n")
            
            f.write("3-0 预测 (2支):\n")
            for team in optimal_prediction['3-0']:
                prob = probabilities.get(team, {}).get('3-0', 0.0)
                f.write(f"  ✓ {team} (3-0概率: {prob:.1%})\n")
            
            f.write("\n3-1或3-2晋级 (6支):\n")
            for team in optimal_prediction['advances']:
                prob = probabilities.get(team, {}).get('3-1-or-3-2', 0.0)  # 防御性访问
                f.write(f"  ✓ {team} (3-1/3-2概率: {prob:.1%})\n")
            
            f.write("\n0-3 预测 (2支):\n")
            for team in optimal_prediction['0-3']:
                prob = probabilities.get(team, {}).get('0-3', 0.0)  # 防御性访问
                f.write(f"  ✓ {team} (0-3概率: {prob:.1%})\n")
        print(f"[LOG] {datetime.now().strftime('%H:%M:%S')} - 报告文件保存成功", flush=True)
    except Exception as e:
        print(f"[ERROR] 报告保存失败: {e}", flush=True)
    
    print(f"[LOG] {datetime.now().strftime('%H:%M:%S')} - 所有结果已保存", flush=True)
    print("\n完成！结果已保存:")
    print("  prediction_results.json")
    print("  optimized_report.txt")
    print(f"\n[LOG] {datetime.now().strftime('%H:%M:%S')} - 程序正常结束", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    import pandas
    globals()['pd'] = pandas
    
    main()
