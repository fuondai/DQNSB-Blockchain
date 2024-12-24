from typing import Tuple, Dict
import numpy as np
from ..blockchain.directory_committee import DirectoryCommittee
from ..blockchain.dynamic_blockchain import DynamicShardedBlockchain
from ..config import Config
import random
from gym import spaces

class BlockchainEnvironment:
    def __init__(self, min_shards: int = 2, max_shards: int = 6,
                 min_nodes_per_shard: int = 4, max_nodes_per_shard: int = 12,
                 security_weight: float = 0.7, malicious_ratio: float = 0.3):
        """Khởi tạo môi trường blockchain"""
        
        # Các tham số cấu hình
        self.min_shards = min_shards
        self.max_shards = max_shards
        self.min_nodes_per_shard = min_nodes_per_shard
        self.max_nodes_per_shard = max_nodes_per_shard
        self.security_weight = security_weight
        self.malicious_ratio = malicious_ratio
        self.transaction_size = 100  # Default size
        
        # Khởi tạo blockchain
        self.current_shards = random.randint(min_shards, max_shards)
        self.nodes_per_shard = random.randint(min_nodes_per_shard, max_nodes_per_shard)
        
        # Lưu trạng thái trước
        self.prev_shards = self.current_shards
        self.prev_nodes_per_shard = self.nodes_per_shard
        
        # Khởi tạo blockchain
        self.blockchain = DynamicShardedBlockchain(
            num_shards=self.current_shards,
            nodes_per_shard=self.nodes_per_shard,
            malicious_ratio=malicious_ratio
        )
        
        # Các tham số reward
        self.target_tps = Config.TARGET_TPS
        self.target_security = Config.TARGET_SECURITY
        self.min_tps_threshold = Config.MIN_TPS_THRESHOLD
        self.min_security_threshold = Config.MIN_SECURITY_THRESHOLD
        self.shard_penalty_factor = Config.SHARD_PENALTY_FACTOR

        # Thêm các thuộc tính để lưu trữ metrics
        self.current_tps = 0
        self.current_security = 0
        self.old_tps = 0
        self.old_security = 0
        
        # Thêm các list để lưu lịch sử
        self.config_history = []
        self.performance_history = []
        self.action_history = []  # Lưu lịch sử action dưới dạng list số nguyên
        
        # Không gian trạng thái và hành động
        self.observation_space = spaces.Box(
            low=np.array([min_shards, min_nodes_per_shard, 0, 0], dtype=np.float32),
            high=np.array([max_shards, max_nodes_per_shard, np.inf, 1], dtype=np.float32)
        )
        
        self.action_space = spaces.Discrete(4)  # Tăng/giảm số shard hoặc nodes per shard

    def reset(self):
        """Reset môi trường về trạng thái ban đầu"""
        # Reset các thuộc tính
        self.current_shards = random.randint(self.min_shards, self.max_shards)
        self.nodes_per_shard = random.randint(self.min_nodes_per_shard, self.max_nodes_per_shard)
        self.prev_shards = self.current_shards
        self.prev_nodes_per_shard = self.nodes_per_shard
        
        # Reset blockchain
        self.blockchain = DynamicShardedBlockchain(
            num_shards=self.current_shards,
            nodes_per_shard=self.nodes_per_shard,
            malicious_ratio=self.malicious_ratio
        )
        
        # Reset histories
        self.config_history = []
        self.performance_history = []
        
        # Get initial state
        self.current_tps = self.blockchain.calculate_tps()
        self.current_security = self.blockchain.calculate_security_score()
        
        state = np.array([
            self.current_shards,
            self.nodes_per_shard,
            self.current_tps,
            self.current_security
        ], dtype=np.float32)
        
        return state

    def step(self, action: int):
        """Thực hiện một bước trong môi trường"""
        
        # Log chi tiết về action và kết quả
        #print(f"Action: {action}, Current shards: {self.current_shards}, " 
        #      f"Nodes per shard: {self.nodes_per_shard}")
        #print(f"TPS change: {self.current_tps - self.old_tps:.2f}, "
        #      f"Security change: {self.current_security - self.old_security:.2f}")
        
        # Lưu trạng thái trước
        self.old_tps = self.blockchain.calculate_tps()
        self.old_security = self.blockchain.calculate_security_score()
        
        # Lưu action vào history (chỉ lưu số action)
        self.action_history.append(action)  # Lưu trực tiếp action number
        
        # Thực hiện action
        if action == 0:  # Tăng số shard
            if self.current_shards >= self.min_shards + 3:
                action = 2  # Chuyển sang tăng nodes thay vì tăng shard
            else:
                self.current_shards = min(self.current_shards + 1, self.max_shards)
        elif action == 1:  # Giảm số shard
            self.current_shards = max(self.current_shards - 1, self.min_shards)
        elif action == 2:  # Tăng nodes per shard
            self.nodes_per_shard = min(self.nodes_per_shard + 1, self.max_nodes_per_shard)
        else:  # Giảm nodes per shard
            self.nodes_per_shard = max(self.nodes_per_shard - 1, self.min_nodes_per_shard)
            
        # Cập nhật blockchain
        self.blockchain = DynamicShardedBlockchain(
            num_shards=self.current_shards,
            nodes_per_shard=self.nodes_per_shard,
            malicious_ratio=self.malicious_ratio
        )
        
        # Tính toán metrics mới
        self.current_tps = self.blockchain.calculate_tps()
        self.current_security = self.blockchain.calculate_security_score()
        
        # Lưu cấu hình và metrics vào history
        config = {
            'shards': self.current_shards,
            'nodes_per_shard': self.nodes_per_shard,
            'tps': self.current_tps,
            'security': self.current_security,
            'processed_tx': 10  # Giả định mỗi step xử lý 10 transactions
        }
        self.config_history.append(config)
        
        # Tính reward
        reward = self.calculate_reward()
        
        # Lấy state mới
        state = np.array([
            self.current_shards,
            self.nodes_per_shard,
            self.current_tps,
            self.current_security
        ], dtype=np.float32)
        
        # Kiểm tra điều kiện kết thúc
        done = False
        info = {
            'tps': self.current_tps,
            'security': self.current_security
        }
        
        return state, reward, done, info

    def calculate_reward(self):
        """Tính reward với nhiều mức thưởng/phạt"""
        # Chuẩn hóa metrics
        normalized_tps = self.current_tps / self.target_tps
        tps_delta = (self.current_tps - self.old_tps) / max(self.old_tps, 1)
        security_delta = (self.current_security - self.old_security) / max(self.old_security, 0.1)
        
        # Base reward - khuyến khích cả TPS và security cao
        base_reward = (
            0.4 * normalized_tps + 
            0.6 * self.current_security
        ) * 500
        
        # Thưởng cải thiện
        improvement_reward = 0
        if tps_delta > 0:
            improvement_reward += tps_delta * 1000
            if tps_delta > 0.1:  # Thưởng đột phá
                improvement_reward += 500
                
        if security_delta > 0:
            improvement_reward += security_delta * 2000
            if security_delta > 0.05:  # Thưởng đột phá
                improvement_reward += 1000
        
        # Phạt suy giảm
        degradation_penalty = 0
        if tps_delta < 0:
            degradation_penalty += abs(tps_delta) * 1500
            if tps_delta < -0.1:  # Phạt nặng suy giảm lớn
                degradation_penalty += 800
                
        if security_delta < 0:
            degradation_penalty += abs(security_delta) * 2500
            if security_delta < -0.05:  # Phạt nặng suy giảm lớn
                degradation_penalty += 1500

        # Phạt mất cân bằng
        imbalance_penalty = 0
        if normalized_tps < 0.4:
            imbalance_penalty += (0.4 - normalized_tps) * 2000
        if self.current_security < 0.5:
            imbalance_penalty += (0.5 - self.current_security) * 3000
            
        # Tăng penalty cho việc thay đổi cấu hình quá thường xuyên
        configuration_change_penalty = 0
        if len(self.action_history) > 10:
            recent_actions = self.action_history[-10:]
            # Tăng penalty lên để hạn chế thay đổi liên tục
            if len(set(recent_actions)) > 3:  # Giảm ngưỡng từ 5 xuống 3
                configuration_change_penalty = 1000  # Tăng từ 500 lên 1000

        # Tăng reward cho việc duy trì cấu hình ổn định
        stability_bonus = 0 
        if len(self.action_history) > 20:
            if len(set(self.action_history[-20:])) <= 2:  # Giảm từ 3 xuống 2
                stability_bonus = 2000  # Tăng từ 1000 lên 2000

        # Thêm penalty cho việc tăng shard quá nhiều
        shard_penalty = 0
        if self.current_shards > self.min_shards + 3:
            shard_penalty = (self.current_shards - (self.min_shards + 3)) * 500

        # Thêm penalty cho việc thay đổi liên tục cùng một hướng
        direction_penalty = 0
        if len(self.action_history) > 5:
            recent_actions = self.action_history[-5:]
            if len(set(recent_actions)) == 1:  # Nếu 5 action gần nhất giống nhau
                direction_penalty = 1000

        # Thêm reward cho cross-shard efficiency
        cross_shard_success_rate = self._calculate_cross_shard_success_rate()
        cross_shard_reward = cross_shard_success_rate * 1000

        # Tính tổng reward
        total_reward = (
            base_reward + 
            improvement_reward + 
            stability_bonus - 
            degradation_penalty - 
            imbalance_penalty - 
            configuration_change_penalty -
            shard_penalty -
            direction_penalty +
            cross_shard_reward
        )
        
        return total_reward

    def _calculate_cross_shard_success_rate(self) -> float:
        """Tính tỷ lệ thành công của cross-shard transactions"""
        total_cross_shard = len(self.blockchain.cross_shard.pending_txs)
        if total_cross_shard == 0:
            return 1.0
            
        successful = sum(1 for tx in self.blockchain.cross_shard.pending_txs.values() 
                        if tx.status == 'completed')
        return successful / total_cross_shard

    def _get_state(self) -> np.ndarray:
        """Get state với noise"""
        state = np.array([
            self.current_shards / self.max_shards,
            self.nodes_per_shard / self.max_nodes_per_shard,
            self.current_tps / 100.0,  # Normalize TPS
            self.current_security
        ])
        
        # Thêm noise khi training
        if not self.evaluate:
            noise = np.random.normal(0, 0.02, state.shape)
            state = np.clip(state + noise, 0, 1)
            
        return state
        
    def _calculate_tps(self) -> float:
        # TPS cơ bản cho 1 shard với cấu hình chuẩn
        base_tps = 20  # Tăng TPS cơ bản lên 20
        
        # Hiệu suất tăng theo số shard
        if self.current_shards <= 4:
            shard_factor = self.current_shards * 0.9  # Giảm hệ số xuống 0.9
        else:
            shard_factor = 3.6 + (self.current_shards - 4) * 0.25  # Giảm tốc độ tăng
            
        # Hiệu suất phụ thuộc vào số node/shard
        if self.nodes_per_shard < 10:
            node_factor = 0.7  # Phạt mạnh hơn cho quá ít node
        elif self.nodes_per_shard <= 20:  # Giảm ngưỡng tối ưu xuống 20
            node_factor = 1.0
        else:
            node_factor = 1.0 - (self.nodes_per_shard - 20) * 0.02  # Tăng penalty
            
        # Hiệu suất phụ thuộc vào block size
        block_factor = 0.7 + 0.3 * (min(self.current_block_size, 1500) / 1500)  # Giảm ảnh hưởng
            
        # Hiệu suất phụ thuộc vào block interval
        interval_factor = 0.8 + 0.2 * (min(5, self.current_block_interval) / 5)  # Giảm ảnh hưởng
            
        # Ảnh hưởng của node độc hại
        security_impact = max(0.5, 1.0 - self.malicious_ratio * 1.2)  # Giảm ảnh hưởng
            
        # Tính TPS cuối cùng
        tps = base_tps * shard_factor * node_factor * block_factor * interval_factor * security_impact
        
        # Giới hạn TPS trong khoảng hợp lý
        return min(max(15, tps), 80)  # Giảm giới hạn trên xuống 80
        
    def _calculate_security(self) -> float:
        # Security cơ bản phụ thuộc vào số node/shard
        if self.nodes_per_shard < 10:
            base_security = 0.65  # Giảm security cơ bản
        elif self.nodes_per_shard <= 20:  # Giảm ngưỡng tối ưu
            base_security = 0.8  # Giảm security tối đa
        else:
            base_security = 0.8 - (self.nodes_per_shard - 20) * 0.015  # Tăng penalty
            
        # Ảnh hưởng của số shard đến security
        if self.current_shards <= 4:
            shard_penalty = 0
        else:
            shard_penalty = (self.current_shards - 4) * 0.05  # Tăng penalty
            
        # Ảnh hưởng của node độc hại
        malicious_impact = self.malicious_ratio * 1.2  # Tăng ảnh hưởng
            
        # Tính security cuối cùng
        security = base_security * (1 - shard_penalty) * (1 - malicious_impact)
        
        # Giới hạn security trong khoảng hợp lý
        return max(0.45, min(security, 0.85))  # Thu hẹp khoảng security
        
    def _adjust_shards(self, action: int):
        old_shards = self.current_shards
        
        # Cho phép thay đổi tự do hơn
        if action == 0:  # Giảm shard
            new_shards = max(self.min_shards, self.current_shards - 1)
        elif action == 2:  # Tăng shard
            new_shards = min(self.max_shards, self.current_shards + 1)
        else:
            return
            
        # Tự động điều chỉnh số node/shard
        self.current_shards = new_shards
        self.nodes_per_shard = self.total_nodes // new_shards
        self.dc = DirectoryCommittee(self.current_shards, self.nodes_per_shard)
        
    def _adjust_block_size(self, action: int):
        current_idx = self.block_sizes.index(self.current_block_size)
        if action == 0 and current_idx > 0:
            self.current_block_size = self.block_sizes[current_idx - 1]
        elif action == 1 and current_idx < len(self.block_sizes) - 1:
            self.current_block_size = self.block_sizes[current_idx + 1]
            
    def _adjust_block_interval(self, action: int):
        current_idx = self.block_intervals.index(self.current_block_interval)
        if action == 0 and current_idx > 0:
            self.current_block_interval = self.block_intervals[current_idx - 1]
        elif action == 1 and current_idx < len(self.block_intervals) - 1:
            self.current_block_interval = self.block_intervals[current_idx + 1]
        
    def set_malicious_ratio(self, ratio: float):
        """Set target malicious ratio"""
        self.target_malicious_ratio = max(0.0, min(0.3, ratio))
        
    def adjust_nodes_per_shard(self, new_nodes: int):
        """Điều chỉnh số lượng node trong mỗi shard"""
        self.nodes_per_shard = max(self.min_nodes_per_shard, 
                                 min(new_nodes, self.max_nodes_per_shard))
        # Cập nhật lại Directory Committee
        self.dc = DirectoryCommittee(self.current_shards, self.nodes_per_shard)

    def get_performance_stats(self):
        """Trở về thống kê hiệu suất"""
        if not self.performance_history:
            return {
                'best_tps': 0,
                'best_security': 0,
                'avg_tps': 0,
                'avg_security': 0
            }
            
        tps_values = [m['tps'] for m in self.performance_history if m['tps'] > 0]
        security_values = [m['security'] for m in self.performance_history if m['security'] > 0]
        
        return {
            'best_tps': max(tps_values) if tps_values else 0,
            'best_security': max(security_values) if security_values else 0,
            'avg_tps': np.mean(tps_values) if tps_values else 0,
            'avg_security': np.mean(security_values) if security_values else 0
        }
        
    def suggest_improvements(self):
        """Đề xuất cải thiện dựa trên lịch sử"""
        suggestions = []
        
        # Phân tích hiệu suất
        avg_tps = np.mean([m['tps'] for m in self.performance_history[-10:]])
        avg_security = np.mean([m['security'] for m in self.performance_history[-10:]])
        
        if avg_tps < 50:
            suggestions.append("TPS thấp: Xem xét tăng số shard hoặc giảm block interval")
        if avg_security < 0.6:
            suggestions.append("Security thấp: Xem xét tăng số node/shard hoặc giảm số shard")
            
        return suggestions
        
    def analyze_config_impact(self):
        """Phân tích ảnh hưởng của các thay đổi cấu hình"""
        impacts = {
            'shard_changes': [],
            'node_changes': [],
            'block_size_changes': [],
            'interval_changes': []
        }
        
        for i in range(1, len(self.config_history)):
            prev = self.config_history[i-1]
            curr = self.config_history[i]
            
            # Phân tích thay đổi
            if prev['shards'] != curr['shards']:
                impacts['shard_changes'].append({
                    'change': curr['shards'] - prev['shards'],
                    'tps_impact': curr['tps'] - prev['tps'],
                    'security_impact': curr['security'] - prev['security']
                })
                
            # Tương tự cho các tham số khác...
            
        return impacts

    def set_transaction_size(self, size: int):
        """Thiết lập kích thước transaction"""
        self.transaction_size = size
        self.blockchain.set_transaction_size(size)