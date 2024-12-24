import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from datetime import datetime
import os
import time

class BlockchainVisualizer:
    def __init__(self):
        plt.style.use('seaborn')
        # Tạo tên thư mục với timestamp
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'plots/run_{self.timestamp}'
        # Tạo thư mục nếu chưa tồn tại
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def _save_plot(self, filename):
        """Lưu biểu đồ hiện tại thành file ảnh"""
        # Không cần thêm timestamp vào tên file nữa vì đã có trong tên thư mục
        filepath = os.path.join(self.output_dir, f'{filename}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\nĐã lưu biểu đồ: {filepath}")
        plt.close()
        
    def plot_tps_history(self, tps_history: List[float]):
        plt.figure(figsize=(10, 6))
        plt.plot(tps_history, label='TPS')
        plt.title('Lịch sử TPS')
        plt.xlabel('Số block')
        plt.ylabel('Transactions/giây')
        plt.grid(True)
        plt.legend()
        self._save_plot('tps_history')
        
    def plot_shard_distribution(self, num_shards: int, transactions_per_shard: List[int]):
        plt.figure(figsize=(10, 6))
        plt.bar(range(num_shards), transactions_per_shard)
        plt.title('Transaction Distribution Across Shards')
        plt.xlabel('Shard ID')
        plt.ylabel('Number of Transactions')
        plt.grid(True)
        plt.show()
        
    def plot_security_metrics(self, security_scores: List[float], malicious_ratios: List[float]):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(security_scores)
        plt.title('Security Score Over Time')
        plt.xlabel('Time')
        plt.ylabel('Security Score')
        
        plt.subplot(1, 2, 2)
        plt.plot(malicious_ratios)
        plt.title('Malicious Nodes Ratio')
        plt.xlabel('Time')
        plt.ylabel('Ratio')
        
        plt.tight_layout()
        plt.show()
        
    def plot_drl_metrics(self, rewards: List[float], epsilons: List[float]):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.title('DRL Rewards Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(1, 2, 2)
        plt.plot(epsilons)
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        plt.tight_layout()
        plt.show()
        
    def plot_scenario_comparison(self, scenario_metrics: Dict[str, List[dict]]):
        plt.figure(figsize=(15, 10))
        
        # TPS Comparison
        plt.subplot(2, 2, 1)
        for scenario, metrics in scenario_metrics.items():
            try:
                tps = [m.get('tps', 0) for m in metrics]
                plt.plot(tps, label=scenario)
            except (KeyError, TypeError):
                print(f"Warning: Missing TPS data for scenario {scenario}")
        plt.title('TPS Comparison')
        plt.legend()
        
        # Security Comparison
        plt.subplot(2, 2, 2)
        for scenario, metrics in scenario_metrics.items():
            try:
                security = [m.get('security', 0) for m in metrics]
                plt.plot(security, label=scenario)
            except (KeyError, TypeError):
                print(f"Warning: Missing security data for scenario {scenario}")
        plt.title('Security Score')
        plt.legend()
        
        # Shard Distribution
        plt.subplot(2, 2, 3)
        for scenario, metrics in scenario_metrics.items():
            try:
                shards = [m.get('num_shards', 0) for m in metrics]
                plt.plot(shards, label=scenario)
            except (KeyError, TypeError):
                print(f"Warning: Missing shard data for scenario {scenario}")
        plt.title('Number of Shards')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_comparison(self, malicious_ratios, drl_metrics, static_metrics):
        """Vẽ biểu đồ so sánh hiệu suất"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Vẽ biểu đồ TPS
        tps_data = [float(drl_metrics['tps']), float(static_metrics['tps'])]
        ax1.bar(['DRL Agent', 'Static Blockchain'], tps_data, color=['blue', 'red'])
        ax1.set_title('So sánh TPS')
        ax1.set_ylabel('TPS')
        ax1.grid(True)
        
        # Vẽ biểu đồ Security
        security_data = [float(drl_metrics['security']), float(static_metrics['security'])]
        ax2.bar(['DRL Agent', 'Static Blockchain'], security_data, color=['blue', 'red'])
        ax2.set_title('So sánh Security')
        ax2.set_ylabel('Security Score')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        self._save_plot('performance_comparison')
        
    def plot_training_progress(self, training_stats):
        """Vẽ biểu đồ tiến trình training"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        episodes = list(range(len(training_stats['reward'])))
        
        # Vẽ reward
        ax1.plot(episodes, training_stats['reward'])
        ax1.set_title('Reward theo Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        # Vẽ TPS
        ax2.plot(episodes, training_stats['tps'])
        ax2.set_title('TPS theo Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('TPS')
        ax2.grid(True)
        
        # Vẽ Security
        ax3.plot(episodes, training_stats['security'])
        ax3.set_title('Security theo Episode')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Security Score')
        ax3.grid(True)
        
        # Vẽ Epsilon
        ax4.plot(episodes, training_stats['epsilon'])
        ax4.set_title('Epsilon theo Episode')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.grid(True)
        
        plt.tight_layout()
        self._save_plot('training_progress')
        
    def plot_config_impact(self, config_impacts):
        """Vẽ biểu đồ ảnh hưởng của các thay đổi cấu hình"""
        params = list(config_impacts.keys())
        tps_impacts = []
        security_impacts = []
        
        for param in params:
            if config_impacts[param]:
                tps_impacts.append(np.mean([i['tps_impact'] for i in config_impacts[param]]))
                security_impacts.append(np.mean([i['security_impact'] for i in config_impacts[param]]))
            else:
                tps_impacts.append(0)
                security_impacts.append(0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Vẽ ảnh hưởng đến TPS
        ax1.bar(params, tps_impacts)
        ax1.set_title('Ảnh hưởng đến TPS')
        ax1.set_xlabel('Tham số')
        ax1.set_ylabel('Thay đổi TPS')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        ax1.grid(True)
        
        # Vẽ ảnh hưởng đến Security
        ax2.bar(params, security_impacts)
        ax2.set_title('Ảnh hưởng đến Security')
        ax2.set_xlabel('Tham số')
        ax2.set_ylabel('Thay đổi Security')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        ax2.grid(True)
        
        plt.tight_layout()
        self._save_plot('config_impact')
        
    def plot_training_metrics(self, metrics):
        """Vẽ biểu đồ metrics trong quá trình training"""
        episodes = list(range(len(metrics['rewards'])))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot rewards
        ax1.plot(episodes, metrics['rewards'], 'b-')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        
        # Plot TPS và Security trên cùng trục
        ax2.plot(episodes, metrics['tps'], 'r-', label='TPS')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('TPS/Security')
        
        ax2_sec = ax2.twinx()
        ax2_sec.plot(episodes, metrics['security'], 'g-', label='Security')
        
        if 'epsilon' in metrics:
            ax2_sec.plot(episodes, metrics['epsilon'], 'y--', alpha=0.5, label='Epsilon')
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_sec.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        self._save_plot('training_metrics')
        
    def plot_action_distribution(self, action_history):
        plt.figure(figsize=(12, 6))
        
        # Định nghĩa tên các action
        action_names = ['Tăng shard', 'Giảm shard', 'Tăng nodes', 'Giảm nodes']
        
        # Đếm số lần xuất hiện của mỗi action
        action_counts = np.zeros(len(action_names))
        for action in action_history:
            if 0 <= action < len(action_names):
                action_counts[action] += 1
            
        # Tính phần trăm
        total_actions = sum(action_counts)
        action_percentages = (action_counts / total_actions) * 100
        
        # Vẽ biểu đồ cột với 2 y-axis
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Trục y bên trái cho số lượng
        bars = ax1.bar(action_names, action_counts, color=['blue', 'red', 'green', 'orange'])
        ax1.set_ylabel('Số lần thực hiện')
        
        # Trục y bên phải cho phần trăm
        ax2 = ax1.twinx()
        ax2.set_ylabel('Phần trăm (%)')
        
        # Thêm số liệu lên đầu cột
        for i, (count, percentage) in enumerate(zip(action_counts, action_percentages)):
            ax1.text(i, count, f'{int(count)}\n({percentage:.1f}%)', 
                    ha='center', va='bottom')
        
        plt.title('Phân phối Hành động của Agent')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot('action_distribution')
        
    def plot_performance_comparison(self, drl_metrics, static_metrics):
        """Vẽ biểu đồ so sánh chi tiết giữa DRL và Static"""
        plt.figure(figsize=(15, 5))
        
        # Subplot 1: TPS Comparison 
        plt.subplot(1, 3, 1)
        labels = ['DRL', 'Static']
        
        # Lấy giá trị TPS trung bình cho DRL
        drl_tps = np.mean(drl_metrics['tps'])
        tps_values = [drl_tps, static_metrics['tps']]
        
        plt.bar(labels, tps_values, color=['blue', 'gray'])
        plt.title('So sánh TPS')
        plt.ylabel('TPS')
        plt.grid(True)
        
        # Thêm giá trị lên đầu cột
        for i, v in enumerate(tps_values):
            plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
        # Subplot 2: Security Comparison
        plt.subplot(1, 3, 2)
        
        # Lấy giá trị security trung bình cho DRL
        drl_security = np.mean(drl_metrics['security'])
        security_values = [drl_security, static_metrics['security']]
        
        plt.bar(labels, security_values, color=['red', 'gray'])
        plt.title('So sánh Security')
        plt.ylabel('Security Score')
        plt.grid(True)
        
        # Thêm giá trị lên đầu cột
        for i, v in enumerate(security_values):
            plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
        # Subplot 3: Efficiency Score (TPS * Security)
        plt.subplot(1, 3, 3)
        efficiency_drl = drl_tps * drl_security
        efficiency_static = static_metrics['tps'] * static_metrics['security']
        efficiency_values = [efficiency_drl, efficiency_static]
        
        plt.bar(labels, efficiency_values, color=['green', 'gray'])
        plt.title('So sánh Hiệu suất Tổng thể')
        plt.ylabel('Efficiency Score')
        plt.grid(True)
        
        # Thêm giá trị lên đầu cột
        for i, v in enumerate(efficiency_values):
            plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        self._save_plot('performance_comparison')
        
    def plot_tps_comparison(self, tx_sizes, drl_tps, static_tps):
        """Vẽ biểu đồ so sánh TPS với các kích thước transaction khác nhau"""
        plt.figure(figsize=(10, 6))
        
        # Vẽ đường TPS
        plt.plot(tx_sizes, drl_tps, 'b-o', label='DRL Agent', linewidth=2)
        plt.plot(tx_sizes, static_tps, 'r-o', label='Static Blockchain', linewidth=2)
        
        plt.xlabel('Transaction Size (Bytes)')
        plt.ylabel('Transactions Per Second (TPS)')
        plt.title('TPS Comparison with Different Transaction Sizes')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Thêm giá trị lên các điểm
        for i, (drl, static) in enumerate(zip(drl_tps, static_tps)):
            plt.annotate(f'{drl:.1f}', (tx_sizes[i], drl), 
                        textcoords="offset points", xytext=(0,10), ha='center')
            plt.annotate(f'{static:.1f}', (tx_sizes[i], static), 
                        textcoords="offset points", xytext=(0,-15), ha='center')
        
        self._save_plot('tps_size_comparison')
  