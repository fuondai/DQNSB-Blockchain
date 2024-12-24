import numpy as np
from typing import List, Dict
from .directory_committee import DirectoryCommittee
from .cross_shard import CrossShardCoordinator

class DynamicShardedBlockchain:
    def __init__(self, num_shards: int, nodes_per_shard: int, malicious_ratio: float = 0.3):
        """Khởi tạo blockchain với số lượng shard và node có thể thay đổi"""
        self.num_shards = num_shards
        self.nodes_per_shard = nodes_per_shard
        self.malicious_ratio = malicious_ratio
        self.transaction_size = 100  # Default size
        
        # Khởi tạo directory committee
        self.dc = DirectoryCommittee(num_shards, nodes_per_shard)
        
        # Khởi tạo cross-shard coordinator
        self.cross_shard = CrossShardCoordinator(num_shards)
        
        # Các tham số hiệu suất
        self.base_tps = 20  # TPS cơ bản cho 1 shard
        self.base_security = 0.8  # Security score cơ bản
        
    def set_transaction_size(self, size: int):
        """Thiết lập kích thước transaction"""
        self.transaction_size = size
        
    def calculate_tps(self) -> float:
        """Tính TPS với nhiều yếu tố ảnh hưởng"""
        # Base TPS theo transaction size
        base_tps = 100 * np.exp(-0.0007 * self.transaction_size)
        
        # Shard factor - tăng hiệu quả theo số shard
        if self.num_shards <= 2:
            shard_factor = 0.5 + self.num_shards * 0.2  # Bắt đầu thấp
        elif self.num_shards <= 4:
            shard_factor = 0.9 + (self.num_shards - 2) * 0.15  # Tăng vừa phải
        else:
            shard_factor = 1.2 + (self.num_shards - 4) * 0.08  # Tăng chậm
            
        # Node factor - tăng theo số node
        if self.nodes_per_shard < 6:
            node_factor = 0.4 + self.nodes_per_shard * 0.08  # Bắt đầu thấp
        elif self.nodes_per_shard < 12:
            node_factor = 0.8 + (self.nodes_per_shard - 6) * 0.03  # Tăng vừa phải
        else:
            node_factor = 0.98 - (self.nodes_per_shard - 12) * 0.01  # Giảm nhẹ
            
        # Security impact - ảnh hưởng mạnh của node độc hại
        security_impact = 1.0 - (self.malicious_ratio ** 0.6)
        
        # Network factor - hiệu ứng mạng
        total_nodes = self.num_shards * self.nodes_per_shard
        network_factor = 1.0 - (total_nodes / 100) * 0.1  # Giảm 10% mỗi 100 node
        
        # Cross-shard impact - hiệu ứng của cross-shard transactions
        cross_shard_txs = len(self.cross_shard.pending_txs)
        if cross_shard_txs > 0:
            cross_shard_factor = 0.9  # Giảm 10% TPS khi có cross-shard tx
        else:
            cross_shard_factor = 1.0
            
        # Tính TPS cuối cùng
        tps = (base_tps * shard_factor * node_factor * 
               security_impact * network_factor * cross_shard_factor)
        
        # Giới hạn động theo transaction size
        min_tps = 20 * np.exp(-0.0005 * self.transaction_size)
        max_tps = 85 * np.exp(-0.0003 * self.transaction_size)
        
        return min(max(min_tps, tps), max_tps)
        
    def calculate_security_score(self) -> float:
        """Tính security score với nhiều yếu tố"""
        # Base security từ số node
        if self.nodes_per_shard < 6:
            base_security = 0.3 + self.nodes_per_shard * 0.05
        elif self.nodes_per_shard < 12:
            base_security = 0.6 + (self.nodes_per_shard - 6) * 0.03
        else:
            base_security = 0.78 - (self.nodes_per_shard - 12) * 0.01
            
        # Shard penalty
        shard_penalty = max(0, (self.num_shards - 3) * 0.07)
        
        # Malicious impact
        malicious_impact = 1.0 - (self.malicious_ratio ** 0.5)
        
        # Network size impact
        total_nodes = self.num_shards * self.nodes_per_shard
        network_security = min(1.0, total_nodes / 50)  # Tăng theo số node
        
        # Cross-shard security impact
        cross_shard_txs = len(self.cross_shard.pending_txs)
        if cross_shard_txs > 0:
            cross_shard_security = 0.85  # Giảm security khi có cross-shard tx
        else:
            cross_shard_security = 1.0
            
        security = (
            0.4 * base_security +
            0.3 * (1 - shard_penalty) +
            0.2 * malicious_impact +
            0.1 * network_security
        ) * cross_shard_security
        
        return min(0.85, max(0.2, security))
        
    def process_transactions(self, num_transactions: int) -> Dict:
        """Xử lý một số lượng transaction và trả về metrics"""
        tps = self.calculate_tps()
        security = self.calculate_security_score()
        
        return {
            'tps': tps,
            'security': security,
            'processed_tx': num_transactions,
            'cross_shard_txs': len(self.cross_shard.pending_txs)
        } 