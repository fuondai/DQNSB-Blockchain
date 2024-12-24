from typing import Dict, List, Optional, Tuple
from .block import Block, Transaction
import hashlib
import time

class CrossShardProof:
    def __init__(self, tx_hash: str, source_shard: int, target_shard: int):
        self.tx_hash = tx_hash
        self.source_shard = source_shard
        self.target_shard = target_shard
        self.timestamp = time.time()
        self.signatures = {}
        
    def add_signature(self, node_id: int, signature: str):
        self.signatures[node_id] = signature
        
    def verify(self, min_signatures: int) -> bool:
        return len(self.signatures) >= min_signatures

class CrossShardCoordinator:
    def __init__(self, num_shards: int):
        self.num_shards = num_shards
        self.pending_txs: Dict[str, Transaction] = {}
        self.proofs: Dict[str, CrossShardProof] = {}
        self.locked_accounts: Dict[str, bool] = {}
        
    def start_cross_shard_tx(self, tx: Transaction) -> Tuple[int, int]:
        """Bắt đầu cross-shard transaction"""
        # Xác định source và target shard
        source_shard = self.get_shard_for_account(tx.sender)
        target_shard = self.get_shard_for_account(tx.receiver)
        
        # Lock tài khoản nguồn
        self.lock_account(tx.sender)
        
        # Tạo proof
        tx_hash = tx.calculate_hash()
        proof = CrossShardProof(tx_hash, source_shard, target_shard)
        self.proofs[tx_hash] = proof
        self.pending_txs[tx_hash] = tx
        
        return source_shard, target_shard
        
    def get_shard_for_account(self, account: str) -> int:
        """Xác định shard cho một tài khoản"""
        return int(hashlib.sha256(account.encode()).hexdigest(), 16) % self.num_shards
        
    def lock_account(self, account: str):
        """Lock tài khoản để tránh double-spending"""
        self.locked_accounts[account] = True
        
    def unlock_account(self, account: str):
        """Unlock tài khoản sau khi hoàn thành"""
        if account in self.locked_accounts:
            del self.locked_accounts[account]
            
    def verify_and_commit(self, tx_hash: str, shard_id: int, node_id: int, signature: str) -> bool:
        """Xác thực và commit cross-shard transaction"""
        if tx_hash not in self.proofs:
            return False
            
        proof = self.proofs[tx_hash]
        proof.add_signature(node_id, signature)
        
        # Kiểm tra đủ chữ ký
        if proof.verify(self.min_signatures):
            tx = self.pending_txs[tx_hash]
            self.unlock_account(tx.sender)
            del self.proofs[tx_hash]
            del self.pending_txs[tx_hash]
            return True
            
        return False 