from typing import List, Dict, Optional
from .block import Block, Transaction
from .shard import Shard
from .cross_shard import CrossShardCoordinator
import time

class DirectoryCommittee:
    def __init__(self, num_shards: int, nodes_per_shard: int):
        self.num_shards = num_shards
        self.shards: List[Shard] = []
        self.final_chain: List[Block] = []
        
        # Khởi tạo cross-shard coordinator
        self.cross_shard = CrossShardCoordinator(num_shards)
        
        # Khởi tạo các shard
        for i in range(num_shards):
            self.shards.append(Shard(i, nodes_per_shard))
            
    def process_transaction(self, transaction: Transaction) -> bool:
        """Xử lý transaction với hỗ trợ cross-shard"""
        # Xác định source và target shard
        source_shard = self.cross_shard.get_shard_for_account(transaction.sender)
        target_shard = self.cross_shard.get_shard_for_account(transaction.receiver)
        
        # Nếu là cross-shard transaction
        if source_shard != target_shard:
            return self._handle_cross_shard_tx(transaction, source_shard, target_shard)
            
        # Nếu là single-shard transaction
        return self.shards[source_shard].add_transaction(transaction)
        
    def _handle_cross_shard_tx(self, tx: Transaction, source: int, target: int) -> bool:
        """Xử lý cross-shard transaction"""
        # 1. Bắt đầu cross-shard transaction
        self.cross_shard.start_cross_shard_tx(tx)
        
        # 2. Gửi đến source shard để verify
        if not self.shards[source].verify_cross_shard_tx(tx):
            return False
            
        # 3. Gửi đến target shard để process
        if not self.shards[target].process_cross_shard_tx(tx):
            return False
            
        # 4. Thu thập chữ ký từ cả hai shard
        signatures = self._collect_signatures(tx, [source, target])
        
        # 5. Verify và commit
        return self.cross_shard.verify_and_commit(
            tx.calculate_hash(),
            target,
            self.shards[target].consensus.primary,
            signatures[target]
        )
        
    def _collect_signatures(self, tx: Transaction, shard_ids: List[int]) -> Dict[int, str]:
        """Thu thập chữ ký từ các shard"""
        signatures = {}
        for shard_id in shard_ids:
            sig = self.shards[shard_id].sign_transaction(tx)
            if sig:
                signatures[shard_id] = sig
        return signatures
        
    def create_final_block(self) -> Block:
        """Tạo block tổng hợp từ các shard"""
        # Thu thập block từ tất cả các shard
        shard_blocks = []
        for shard in self.shards:
            block = shard.create_new_block()
            if block:
                shard_blocks.append(block)
                
        if not shard_blocks:
            return None
            
        # Tạo block tổng hợp
        previous_hash = self.final_chain[-1].hash if self.final_chain else "0"
        final_block = Block(shard_id=-1, previous_hash=previous_hash)
        
        # Gộp tất cả giao dịch từ các shard
        for block in shard_blocks:
            for tx in block.transactions:
                final_block.add_transaction(tx)
                
        self.final_chain.append(final_block)
        return final_block
        
    def get_metrics(self) -> Dict:
        """Thu thập metrics từ tất cả các shard"""
        total_transactions = 0
        total_blocks = 0
        
        for shard in self.shards:
            total_transactions += shard.processed_tx_count
            total_blocks += len(shard.chain)
            
        return {
            'total_transactions': total_transactions,
            'total_blocks': total_blocks,
            'num_shards': self.num_shards,
            'avg_block_time': sum(s.avg_block_time for s in self.shards) / self.num_shards
        } 