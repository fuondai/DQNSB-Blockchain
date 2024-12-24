class StaticShardedBlockchain:
    def __init__(self, num_shards: int, nodes_per_shard: int, malicious_ratio: float = 0.3):
        self.num_shards = num_shards
        self.nodes_per_shard = nodes_per_shard
        self.malicious_ratio = malicious_ratio
        self.transaction_size = 100  # Default size
        
    def set_transaction_size(self, size: int):
        """Thiết lập kích thước transaction"""
        self.transaction_size = size
        
    def calculate_tps(self) -> float:
        """Tính toán TPS dựa trên cấu hình và kích thước transaction"""
        base_tps = self.num_shards * (1000 / self.transaction_size)  # TPS cơ bản
        network_factor = 1 - (self.num_shards * self.nodes_per_shard) / 1000  # Hệ số mạng
        return base_tps * network_factor 