from typing import List, Dict, Optional
from .block import Block, Transaction
from ..consensus.pbft import PBFT
import time

class Shard:
    def __init__(self, shard_id: int, num_nodes: int):
        self.shard_id = shard_id
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.num_nodes = num_nodes
        self.consensus = PBFT(num_nodes)
        self.max_block_size = 1000  # Giới hạn kích thước block
        self.min_transactions = 10   # Số lượng transaction tối thiểu để tạo block
        
        # Thêm tracking metrics
        self.processed_tx_count = 0
        self.avg_block_time = 0
        self.last_block_time = time.time()
        
        self.create_genesis_block()
        
        # Dynamic resharding
        # Adaptive load balancing
        
        self.cross_shard_pool = {}  # Pool cho cross-shard txs
        
    def create_genesis_block(self):
        genesis = Block(self.shard_id, "0")
        self.chain.append(genesis)
        
    def add_transaction(self, transaction: Transaction) -> bool:
        """Thêm transaction với kiểm tra điều kiện"""
        if len(self.pending_transactions) >= self.max_block_size:
            return False
            
        self.pending_transactions.append(transaction)
        
        # Tự động tạo block khi đủ điều kiện
        if len(self.pending_transactions) >= self.max_block_size:
            self.create_new_block()
            
        return True
        
    def create_new_block(self) -> Optional[Block]:
        """Tạo block mới với các cải tiến"""
        if len(self.pending_transactions) < self.min_transactions:
            return None
            
        current_time = time.time()
        block_time = current_time - self.last_block_time
        
        # Cập nhật metrics
        if self.avg_block_time == 0:
            self.avg_block_time = block_time
        else:
            self.avg_block_time = 0.9 * self.avg_block_time + 0.1 * block_time
            
        previous_hash = self.chain[-1].hash
        new_block = Block(self.shard_id, previous_hash)
        
        # Thêm transactions vào block
        for tx in self.pending_transactions:
            new_block.add_transaction(tx)
            
        self.pending_transactions = []
        
        # Thực hiện đồng thuận PBFT
        if self.consensus.reach_consensus(new_block):
            self.chain.append(new_block)
            self.processed_tx_count += len(new_block.transactions)
            self.last_block_time = current_time
            return new_block
            
        return None 
        
    def verify_cross_shard_tx(self, tx: Transaction) -> bool:
        """Xác thực cross-shard transaction"""
        # Verify balance và nonce
        if not self._verify_account_state(tx.sender):
            return False
            
        # Lock số dư
        self._lock_balance(tx.sender, tx.amount)
        
        # Thêm vào cross-shard pool
        self.cross_shard_pool[tx.calculate_hash()] = tx
        return True
        
    def process_cross_shard_tx(self, tx: Transaction) -> bool:
        """Xử lý cross-shard transaction"""
        # Verify transaction
        if not self._verify_transaction(tx):
            return False
            
        # Update receiver balance
        self._update_balance(tx.receiver, tx.amount)
        
        # Add to pending transactions
        return self.add_transaction(tx)
        
    def sign_transaction(self, tx: Transaction) -> Optional[str]:
        """Ký transaction bởi shard"""
        if self.consensus.is_primary():
            return self.consensus.sign_message(tx.calculate_hash())
        return None 