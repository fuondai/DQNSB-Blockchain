from typing import List, Dict
import numpy as np
from .directory_committee import DirectoryCommittee
from .block import Transaction
import time

class StaticShardedBlockchain:
    def __init__(self, num_shards: int, nodes_per_shard: int, malicious_ratio: float = 0.3):
        self.num_shards = num_shards
        self.nodes_per_shard = nodes_per_shard
        self.total_nodes = num_shards * nodes_per_shard
        self.malicious_ratio = malicious_ratio
        self.transaction_size = 100  # Default size
        
        self.dc = DirectoryCommittee(num_shards, nodes_per_shard)
        
    def set_transaction_size(self, size: int):
        """Thiết lập kích thước transaction"""
        self.transaction_size = size
        
    def process_transactions(self, num_transactions: int) -> Dict:
        """Xử lý các giao dịch và trả về metrics"""
        start_time = time.time()
        
        # Phân phối giao dịch vào các shard
        tx_per_shard = num_transactions // self.num_shards
        remaining = num_transactions % self.num_shards
        
        processed_tx = 0
        for i in range(self.num_shards):
            shard_tx = tx_per_shard + (1 if i < remaining else 0)
            processed_tx += shard_tx
            
        # Tính thời gian xử lý
        processing_time = time.time() - start_time
        
        # Tính TPS dựa trên cấu hình và kích thước transaction
        base_tps = 25 * (1000 / self.transaction_size)  # TPS cơ bản điều chỉnh theo kích thước
        
        # Hiệu suất phụ thuộc vào số shard
        if self.num_shards <= 4:
            shard_factor = self.num_shards * 0.85
        else:
            shard_factor = 3.4 + (self.num_shards - 4) * 0.15  # Giảm tốc độ tăng
            
        # Hiệu suất phụ thuộc vào số node/shard
        if self.nodes_per_shard < 8:
            node_factor = 0.6  # Phạt mạnh hơn cho quá ít node
        elif self.nodes_per_shard <= 20:
            node_factor = 0.9  # Giảm hiệu suất tối đa
        else:
            node_factor = 0.9 - (self.nodes_per_shard - 20) * 0.02
            
        # Ảnh hưởng của node độc hại
        security_impact = max(0.5, 1.0 - self.malicious_ratio * 1.5)  # Tăng ảnh hưởng của node độc hại
        
        # Tính TPS cuối cùng
        tps = base_tps * shard_factor * node_factor * security_impact
        
        # Giới hạn TPS trong khoảng hợp lý
        tps = min(max(15, tps), 60)  # Giảm giới hạn trên
        
        # Tính security score
        security = self._calculate_security()
        
        return {
            'tps': tps,
            'security': security,
            'processing_time': processing_time,
            'processed_tx': processed_tx
        }

    def _calculate_security(self) -> float:
        # Security cơ bản phụ thuộc vào số node/shard
        if self.nodes_per_shard < 4:
            base_security = 0.4  # Giảm security cơ bản
        elif self.nodes_per_shard <= 20:
            base_security = 0.7  # Giảm security tối đa
        else:
            base_security = 0.7 - (self.nodes_per_shard - 20) * 0.015
            
        # Ảnh hưởng của số shard đến security
        if self.num_shards <= 4:
            shard_penalty = 0
        else:
            shard_penalty = (self.num_shards - 4) * 0.06  # Tăng penalty
            
        # Ảnh hưởng của node độc hại
        if self.malicious_ratio <= 0.2:
            malicious_impact = self.malicious_ratio * 1.5  # Tăng ảnh hưởng
        else:
            malicious_impact = 0.3 + (self.malicious_ratio - 0.2) * 2.5
            
        # Tính security cuối cùng
        security = base_security * (1 - shard_penalty) * (1 - malicious_impact)
        
        # Giới hạn security trong khoảng hợp lý
        return max(0.3, min(security, 0.7))  # Thu hẹp khoảng security
        
    def increase_malicious_nodes(self, increment: float = 0.001):
        self.malicious_ratio = min(0.3, self.malicious_ratio + increment)

    def calculate_tps(self) -> float:
        """Tính TPS cho static blockchain"""
        # Base TPS giảm theo transaction size
        base_tps = 70 * np.exp(-0.0012 * self.transaction_size)  # Giảm nhanh hơn
        
        # Shard factor cố định theo số shard
        if self.num_shards <= 4:
            shard_factor = self.num_shards * 0.6
        else:
            shard_factor = 2.4 + (self.num_shards - 4) * 0.1
            
        # Node factor cố định
        if self.nodes_per_shard < 8:
            node_factor = 0.4
        elif self.nodes_per_shard <= 20:
            node_factor = 0.7
        else:
            node_factor = 0.7 - (self.nodes_per_shard - 20) * 0.02
            
        # Security impact
        security_impact = max(0.3, 1.0 - self.malicious_ratio * 2.0)
        
        # Tính TPS cuối cùng
        tps = base_tps * shard_factor * node_factor * security_impact
        
        return min(max(10, tps), 60)

    def calculate_security_score(self) -> float:
        """Tính security score cho static blockchain"""
        # Base security thấp hơn do cấu trúc cố định
        if self.nodes_per_shard < 4:
            base_security = 0.2
        elif self.nodes_per_shard <= 20:
            base_security = 0.4
        else:
            base_security = 0.4 - (self.nodes_per_shard - 20) * 0.01
            
        # Shard penalty cao hơn do không thể điều chỉnh
        if self.num_shards <= 4:
            shard_penalty = self.num_shards * 0.08
        else:
            shard_penalty = 0.32 + (self.num_shards - 4) * 0.1
            
        # Malicious impact mạnh hơn
        if self.malicious_ratio <= 0.1:
            malicious_impact = 1.0 - self.malicious_ratio * 3
        else:
            malicious_impact = 0.7 - (self.malicious_ratio - 0.1) * 2
            
        security = base_security * (1 - shard_penalty) * malicious_impact
        return min(0.5, max(0.2, security))  # Giới hạn security thấp hơn