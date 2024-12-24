from typing import List, Dict
import random
from ..blockchain.directory_committee import DirectoryCommittee
from ..blockchain.block import Transaction
import time
from ..drl.environment import BlockchainEnvironment

class NetworkSimulator:
    def __init__(self, num_shards: int, nodes_per_shard: int):
        self.dc = DirectoryCommittee(num_shards, nodes_per_shard)
        self.transactions_pool = []
        self.malicious_nodes_ratio = 0.0
        self.scenarios = {
            'normal': {'malicious_ratio': 0.1, 'tx_size': 500},
            'high_malicious': {'malicious_ratio': 0.2, 'tx_size': 500},
            'large_tx': {'malicious_ratio': 0.1, 'tx_size': 1000},
            'worst_case': {'malicious_ratio': 0.2, 'tx_size': 1000}
        }
        
    def generate_random_transaction(self):
        sender = f"wallet_{random.randint(1, 1000)}"
        receiver = f"wallet_{random.randint(1, 1000)}"
        amount = random.uniform(0.1, 100.0)
        return Transaction(sender, receiver, amount)
        
    def simulate_network_load(self, num_transactions: int, tx_size: int):
        for _ in range(num_transactions):
            tx = self.generate_random_transaction()
            self.transactions_pool.append(tx)
            
    def process_transactions(self, batch_size: int = 5):
        processed = 0
        start_time = time.time()
        
        while self.transactions_pool and processed < batch_size:
            tx = self.transactions_pool.pop(0)
            self.dc.process_transaction(tx)
            processed += 1
            
        final_block = self.dc.create_final_block()
        
        end_time = time.time()
        duration = end_time - start_time
        
        security_score = 1.0 / (1.0 + 0.1 * len(self.dc.shards))
        
        base_tps = 15
        shard_factor = len(self.dc.shards) * 0.8
        security_factor = 1.0 - self.malicious_nodes_ratio
        
        tps = base_tps * shard_factor * security_factor
        tps *= (0.9 + 0.2 * random.random())
        
        return {
            'processed_txs': processed,
            'duration': duration,
            'tps': min(tps, 80),
            'security': security_score,
            'num_shards': len(self.dc.shards)
        }
        
    def set_malicious_nodes_ratio(self, ratio: float):
        self.malicious_nodes_ratio = max(0.0, min(1.0, ratio))
        # Cập nhật số lượng node độc hại trong mỗi shard
        for shard in self.dc.shards:
            shard.consensus.fault_tolerance = self.malicious_nodes_ratio 
        
    def run_scenario(self, scenario_name: str, num_transactions: int):
        config = self.scenarios[scenario_name]
        self.set_malicious_nodes_ratio(config['malicious_ratio'])
        
        # Tạo transactions với kích thước được chỉ định
        self.simulate_network_load(num_transactions, tx_size=config['tx_size'])
        
        metrics = []
        while self.transactions_pool:
            result = self.process_transactions(batch_size=10)
            metrics.append(result)
            
        return metrics
        
    def run_baseline_simulation(self):
        metrics = {}
        for scenario_name in self.scenarios:
            scenario_metrics = self.run_scenario(scenario_name, 1000)
            # Đảm bảo mỗi metric có đủ các key cần thiết
            for m in scenario_metrics:
                if 'security' not in m:
                    m['security'] = 1.0 / (1.0 + 0.1 * len(self.dc.shards))
                if 'num_shards' not in m:
                    m['num_shards'] = len(self.dc.shards)
            metrics[scenario_name] = scenario_metrics
        return metrics
        
    def run_drl_simulation(self, agent):
        # Chạy mô phỏng với DRL agent
        metrics = {}
        env = BlockchainEnvironment()
        
        for scenario in self.scenarios:
            state = env.reset()
            scenario_metrics = []
            
            while True:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                scenario_metrics.append({
                    'tps': env._calculate_tps(),
                    'security': env.security_score,
                    'num_shards': env.current_shards
                })
                
                if done:
                    break
                state = next_state
                
            metrics[scenario] = scenario_metrics
        return metrics