"""
CS2 Major 瑞士轮预测系统（Part 2: GPU/Tensor 加速优化）
核心功能：
1. 张量化计算：将模拟数据转换为布尔矩阵
2. 矩阵乘法加速：利用 GPU (CUDA) 并行计算数百万种组合的通过率
3. 批量处理：通过 batchsize.yaml 控制显存占用
"""

import json
import time
import os
import sys
import yaml
import itertools
import torch
from datetime import datetime
from tqdm import tqdm
# ============================================================================
# 配置加载
# ============================================================================

def load_config():
    """
    加载 batchsize.yaml 配置文件
    """
    config_path = 'batchsize.yaml'
    defaults = {
        'device': {
            'use_gpu': True,
            'gpu_id': 0
        },
        'performance': {
            'eval_batch_size': 10000,  # 默认batch size
            'save_every': 1000000       # 默认每100万组保存一次
        }
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    if 'device' in user_config:
                        defaults['device'].update(user_config['device'])
                    if 'performance' in user_config:
                        defaults['performance'].update(user_config['performance'])
                print(f"[配置] 已加载 {config_path}")
        except Exception as e:
            print(f"[警告] 加载配置文件失败，使用默认设置: {e}")
    else:
        print(f"[提示] 未找到 {config_path}，使用默认设置")
        
    return defaults

# ============================================================================
# 核心类：GPU 优化器
# ============================================================================

class PickemOptimizer:
    def __init__(self, data_file, config):
        self.config = config
        self.load_data(data_file)
        self.setup_device()
        self.prepare_tensors()

    def load_data(self, filepath):
        print(f"[1/4] 加载模拟数据: {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self.teams = self.data['teams']
            self.team_to_idx = {team: i for i, team in enumerate(self.teams)}
            self.raw_sims = self.data['raw_simulations']
            self.num_sims = len(self.raw_sims)
            print(f"      - 队伍数量: {len(self.teams)}")
            print(f"      - 模拟样本: {self.num_sims}")
        except FileNotFoundError:
            print(f"[错误] 找不到文件 {filepath}。请先运行 CPU 生成脚本。")
            sys.exit(1)

    def setup_device(self):
        use_gpu = self.config['device']['use_gpu']
        gpu_id = self.config['device']['gpu_id']
        
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_id}')
            props = torch.cuda.get_device_properties(self.device)
            print(f"[设备] 使用 GPU: {props.name} (VRAM: {props.total_memory / 1024**3:.1f} GB)")
        else:
            self.device = torch.device('cpu')
            print(f"[设备] 使用 CPU (PyTorch)")

    def prepare_tensors(self):
        """
        将模拟结果预计算为三个布尔矩阵 (Num_Sims, Num_Teams)
        矩阵1: 是否3-0
        矩阵2: 是否晋级 (且非3-0) -> 对应原来的 'advances' 逻辑
        矩阵3: 是否0-3
        """
        print(f"[2/4] 预计算张量数据...")
        t0 = time.time()
        
        num_teams = len(self.teams)
        
        # 初始化CPU张量
        tensor_3_0 = torch.zeros((self.num_sims, num_teams), dtype=torch.float32)
        tensor_adv = torch.zeros((self.num_sims, num_teams), dtype=torch.float32)
        tensor_0_3 = torch.zeros((self.num_sims, num_teams), dtype=torch.float32)
        
        for sim_idx, sim in enumerate(self.raw_sims):
            # 填充 3-0
            for team in sim['3-0']:
                if team in self.team_to_idx:
                    tensor_3_0[sim_idx, self.team_to_idx[team]] = 1.0
            
            # 填充 0-3
            for team in sim['0-3']:
                if team in self.team_to_idx:
                    tensor_0_3[sim_idx, self.team_to_idx[team]] = 1.0
            
            # 填充 Advances (Original logic: Qualified AND NOT 3-0)
            for team in sim['qualified']:
                if team in self.team_to_idx and team not in sim['3-0']:
                    tensor_adv[sim_idx, self.team_to_idx[team]] = 1.0
        
        # 转移到计算设备并转置为 (Num_Teams, Num_Sims) 以便矩阵乘法
        # Matrix Shape: [16, 100000]
        self.matrix_3_0 = tensor_3_0.t().to(self.device)
        self.matrix_adv = tensor_adv.t().to(self.device)
        self.matrix_0_3 = tensor_0_3.t().to(self.device)
        
        print(f"      - 张量构建完成，耗时 {time.time()-t0:.2f}s")

    def evaluate_batch(self, batch_adv, batch_30, batch_03):
        """
        核心计算核心：利用矩阵乘法评估一批组合
        输入 batch 形状: [Batch_Size, N_Picks] (索引)
        """
        batch_size = len(batch_adv)
        num_teams = len(self.teams)
        
        # 1. 将队伍索引转换为 One-Hot 编码 [Batch_Size, 16]
        # PyTorch 的 scatter 操作非常快
        oh_adv = torch.zeros((batch_size, num_teams), device=self.device)
        oh_adv.scatter_(1, batch_adv, 1.0)
        
        oh_30 = torch.zeros((batch_size, num_teams), device=self.device)
        oh_30.scatter_(1, batch_30, 1.0)
        
        oh_03 = torch.zeros((batch_size, num_teams), device=self.device)
        oh_03.scatter_(1, batch_03, 1.0)
        
        # 2. 矩阵乘法计算得分 [Batch_Size, 16] @ [16, Num_Sims] -> [Batch_Size, Num_Sims]
        # 结果矩阵中的每个元素 (i, j) 代表第 i 个预测组合在第 j 次模拟中的得分
        scores_adv = torch.mm(oh_adv, self.matrix_adv)
        scores_30  = torch.mm(oh_30, self.matrix_3_0)
        scores_03  = torch.mm(oh_03, self.matrix_0_3)
        
        # 3. 总分
        total_scores = scores_adv + scores_30 + scores_03
        
        # 4. 计算通过率 (得分 >= 5)
        # mean(dim=1) 计算每行的平均值，即每个组合的通过率
        pass_rates = (total_scores >= 5.0).float().mean(dim=1)
        
        # 5. 找到这批中最好的
        max_rate, max_idx = torch.max(pass_rates, dim=0)
        
        return max_rate.item(), max_idx.item()

    def run_optimization(self):
        print(f"\n[3/4] 开始搜索...")
        batch_size = self.config['performance']['eval_batch_size']
        save_interval = self.config['performance'].get('save_every', 1000000)
        checkpoint_file = 'gpu_checkpoint.json'
        
        print(f"      - Batch Size: {batch_size:,}")
        print(f"      - 自动保存: 每 {save_interval:,} 组")
        
        all_indices = list(range(len(self.teams)))
        
        # === 断点续传逻辑 ===
        start_count = 0
        best_prediction = None
        best_rate = -1.0
        
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    ckpt = json.load(f)
                    start_count = ckpt.get('processed_count', 0)
                    best_rate = ckpt.get('best_rate', -1.0)
                    best_prediction = ckpt.get('best_prediction', None)
                    print(f"      [恢复进度] 从第 {start_count:,} 组继续，当前最佳: {best_rate:.4%}")
            except Exception as e:
                print(f"      [警告] 无法读取存档: {e}")

        # 缓冲区
        buffer_adv = []
        buffer_30 = []
        buffer_03 = []
        
        # 计数器
        global_counter = 0     
        processed_counter = 0  
        
        start_time = time.time()
        
        # 计算总任务量
        adv_combinations = list(itertools.combinations(all_indices, 6))
        total_ops = len(adv_combinations) * 45 * 28
        
        print(f"      - 生成组合空间中...")
        
        # initial=start_count 让进度条从断点处开始显示
        with tqdm(total=total_ops, initial=start_count, unit="组", desc="      - 进度", ncols=100) as pbar:
            
            for adv_combo in adv_combinations:
                
                adv_set = set(adv_combo)
                remaining = [i for i in all_indices if i not in adv_set]
                
                for t30_combo in itertools.combinations(remaining, 2):
                    t30_set = set(t30_combo)
                    remaining_2 = [i for i in remaining if i not in t30_set]
                    
                    for t03_combo in itertools.combinations(remaining_2, 2):
                        
                        # === 核心跳转逻辑 ===
                        global_counter += 1
                        if global_counter <= start_count:
                            continue 
                        
                        # 添加到缓冲区
                        buffer_adv.append(adv_combo)
                        buffer_30.append(t30_combo)
                        buffer_03.append(t03_combo)
                        
                        if len(buffer_adv) >= batch_size:
                            current_batch_len = len(buffer_adv)
                            
                            t_adv = torch.tensor(buffer_adv, dtype=torch.long, device=self.device)
                            t_30 = torch.tensor(buffer_30, dtype=torch.long, device=self.device)
                            t_03 = torch.tensor(buffer_03, dtype=torch.long, device=self.device)
                            
                            batch_rate, batch_best_idx = self.evaluate_batch(t_adv, t_30, t_03)
                            
                            if batch_rate > best_rate:
                                best_rate = batch_rate
                                best_prediction = {
                                    '3-0': [self.teams[i] for i in buffer_30[batch_best_idx]],
                                    'advances': [self.teams[i] for i in buffer_adv[batch_best_idx]],
                                    '0-3': [self.teams[i] for i in buffer_03[batch_best_idx]]
                                }
                                pbar.write(f"   ✓ 发现新高: {best_rate:.4%} (已处理 {global_counter:,})")
                            
                            # 更新进度条
                            pbar.update(current_batch_len)
                            
                            # 保存Checkpoint逻辑
                            processed_counter += current_batch_len
                            if processed_counter >= save_interval:
                                processed_counter = 0 
                                ckpt_data = {
                                    'processed_count': global_counter,
                                    'best_rate': best_rate,
                                    'best_prediction': best_prediction,
                                    'timestamp': datetime.now().isoformat()
                                }
                                try:
                                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                                        json.dump(ckpt_data, f)
                                except:
                                    pass 


                            buffer_adv, buffer_30, buffer_03 = [], [], []

            if buffer_adv:
                current_batch_len = len(buffer_adv)
                t_adv = torch.tensor(buffer_adv, dtype=torch.long, device=self.device)
                t_30 = torch.tensor(buffer_30, dtype=torch.long, device=self.device)
                t_03 = torch.tensor(buffer_03, dtype=torch.long, device=self.device)
                
                batch_rate, batch_best_idx = self.evaluate_batch(t_adv, t_30, t_03)
                
                if batch_rate > best_rate:
                    best_rate = batch_rate
                    best_prediction = {
                        '3-0': [self.teams[i] for i in buffer_30[batch_best_idx]],
                        'advances': [self.teams[i] for i in buffer_adv[batch_best_idx]],
                        '0-3': [self.teams[i] for i in buffer_03[batch_best_idx]]
                    }
                    pbar.write(f"   ✓ 发现新高: {best_rate:.4%} (已处理 {global_counter:,})")
                
                pbar.update(current_batch_len)

        # 跑完删除 checkpoint
        if os.path.exists(checkpoint_file):
            try:
                os.remove(checkpoint_file)
            except:
                pass

        total_time = time.time() - start_time
        print(f"\n[4/4] 优化完成!")
        print(f"      - 总耗时: {total_time:.1f} 秒")
        print(f"      - 处理组合: {global_counter:,}")
        
        if best_prediction is None:
            print("\n[警告] 暴力搜索未找到有效结果，正在启用启发式回退策略...")
            

            sim_results = self.data.get('simulation_results', {})
            
            if sim_results:
                try:
                    # 3-0
                    top_3_0 = sorted(sim_results.items(), key=lambda x: x[1]['3-0'], reverse=True)
                    t30_names = [t for t, _ in top_3_0][:2]
                    
                    # 晋级
                    top_qual = sorted(sim_results.items(), key=lambda x: x[1]['qualified'], reverse=True)
                    adv_names = []
                    for t, _ in top_qual:
                        if t not in t30_names:
                            adv_names.append(t)
                        if len(adv_names) == 6:
                            break
                    
                    # 0-3
                    top_03 = sorted(sim_results.items(), key=lambda x: x[1]['0-3'], reverse=True)
                    t03_names = []
                    picked_so_far = set(t30_names + adv_names)
                    for t, _ in top_03:
                        if t not in picked_so_far:
                            t03_names.append(t)
                        if len(t03_names) == 2:
                            break
                    

                    best_prediction = {
                        '3-0': t30_names,
                        'advances': adv_names,
                        '0-3': t03_names
                    }
                    
                    print(f"      - [回退] 已基于概率生成保底组合")


                    idx_30 = [self.team_to_idx[t] for t in t30_names]
                    idx_adv = [self.team_to_idx[t] for t in adv_names]
                    idx_03 = [self.team_to_idx[t] for t in t03_names]
                    

                    t_adv = torch.tensor([idx_adv], dtype=torch.long, device=self.device)
                    t_30 = torch.tensor([idx_30], dtype=torch.long, device=self.device)
                    t_03 = torch.tensor([idx_03], dtype=torch.long, device=self.device)
                    
                    best_rate, _ = self.evaluate_batch(t_adv, t_30, t_03)
                    print(f"      - [回退] 该组合经 GPU 验证成功率: {best_rate:.4%}")

                except Exception as e:
                    print(f"      [错误] 启发式回退计算失败: {e}")
            else:
                print("      [错误] 无法读取 simulation_results 数据")


        return best_prediction, best_rate


# ============================================================================
# 主入口
# ============================================================================

def main():
    print("=" * 60)
    print("CS2 Major Pick'Em 优化器 (GPU Accelerated)")
    print("=" * 60)
    
    # 1. 加载配置
    config = load_config()
    
    # 2. 初始化优化器

    data_file = os.path.join('output', 'intermediate_sim_data.json')
    
    # 这里加个检查，防止用户忘了先运行 preresult
    if not os.path.exists(data_file):
        print(f"[错误] 找不到数据文件: {data_file}")
        print(f"       请先运行 python cs2_gen_preresult.py 生成模拟数据！")
        return

    optimizer = PickemOptimizer(data_file, config)
    
    # 3. 运行
    best_pred, best_rate = optimizer.run_optimization()
    
    # 4. 输出报告
    print("\n" + "=" * 60)
    print(f"最优 Pick'Em 预测 (成功率: {best_rate:.2%})")
    print("=" * 60)
    
    if best_pred:
        print(f"3-0 投币 ({len(best_pred['3-0'])}):")
        for t in best_pred['3-0']:
            print(f"  [★] {t}")
            
        print(f"\n晋级 投币 ({len(best_pred['advances'])}):")
        for t in best_pred['advances']:
            print(f"  [+] {t}")
            
        print(f"\n0-3 投币 ({len(best_pred['0-3'])}):")
        for t in best_pred['0-3']:
            print(f"  [x] {t}")
    
    # 保存最终结果
    output = {
        'best_prediction': best_pred,
        'success_rate': best_rate,
        'timestamp': datetime.now().isoformat(),
        'config_used': config
    }
    
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, 'final_prediction.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[保存] 结果已保存至 {output_path}")

if __name__ == "__main__":
    main()