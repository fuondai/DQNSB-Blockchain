from typing import List, Dict
from ..blockchain.block import Block
import random

class PBFT:
    def __init__(self, num_nodes: int, fault_tolerance: float = 0.33):
        self.phases = ['pre-prepare', 'prepare', 'commit']
        self.view_number = 0
        self.primary = 0
        self.num_nodes = num_nodes
        self.fault_tolerance = fault_tolerance
        self.min_votes = int((2 * fault_tolerance + 1) * num_nodes)
        self.prepared_blocks = {}
        self.committed_blocks = {}
        self.cross_shard_votes = {}
        
    def change_view(self):
        """Xử lý khi primary node fail"""
        self.view_number += 1
        self.primary = self.view_number % self.num_nodes
        
    def reach_consensus(self, block: Block) -> bool:
        """Thực hiện quy trình đồng thuận PBFT"""
        # Phase 1: Pre-prepare
        if not self._pre_prepare_phase(block):
            return False
            
        # Phase 2: Prepare
        prepare_votes = self._prepare_phase()
        if prepare_votes < self.min_votes:
            return False
            
        # Phase 3: Commit
        return self._commit_phase()
        
    def _simulate_voting(self) -> int:
        """Mô phỏng số phiếu bầu từ các node"""
        honest_nodes = int(self.num_nodes * (1 - self.fault_tolerance))
        return random.randint(honest_nodes, self.num_nodes)
        
    def _pre_prepare_phase(self, block: Block) -> bool:
        """Primary node kiểm tra và broadcast block"""
        if self.primary == self._get_node_id():
            block_hash = block.calculate_hash()
            message = {
                'type': 'pre-prepare',
                'view': self.view_number,
                'block_hash': block_hash,
                'block': block
            }
            return self._broadcast_message(message)
        return True
        
    def _prepare_phase(self) -> int:
        """Các node xác nhận block hợp lệ"""
        prepare_votes = self._simulate_voting()
        if prepare_votes >= self.min_votes:
            self.prepared_blocks[block.hash] = block
        return prepare_votes
        
    def _commit_phase(self) -> int:
        """Các node commit block"""
        commit_votes = self._simulate_voting()
        if commit_votes >= self.min_votes:
            self.committed_blocks[block.hash] = block
        return commit_votes
        
    def _get_node_id(self) -> int:
        """Mô phỏng node ID"""
        return random.randint(0, self.num_nodes - 1)
        
    def _broadcast_message(self, message: dict) -> bool:
        """Mô phỏng broadcast message"""
        return random.random() > self.fault_tolerance
        
    def sign_message(self, message: str) -> str:
        """Ký một message"""
        # Implement actual signing logic here
        return f"sig_{self.primary}_{message}"
        
    def verify_cross_shard_tx(self, tx_hash: str, signatures: Dict[int, str]) -> bool:
        """Xác thực cross-shard transaction"""
        if tx_hash not in self.cross_shard_votes:
            self.cross_shard_votes[tx_hash] = {}
            
        # Thu thập votes
        for node_id, sig in signatures.items():
            self.cross_shard_votes[tx_hash][node_id] = sig
            
        # Kiểm tra đủ số lượng votes
        return len(self.cross_shard_votes[tx_hash]) >= self.min_votes