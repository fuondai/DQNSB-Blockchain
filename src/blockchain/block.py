from dataclasses import dataclass
from typing import List
import time
import hashlib
import json

@dataclass
class Transaction:
    sender: str
    receiver: str
    amount: float
    data: str = ""
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self):
        return {
            'sender': self.sender,
            'receiver': self.receiver,
            'amount': self.amount,
            'timestamp': self.timestamp
        }
    
    def calculate_hash(self) -> str:
        tx_string = f"{self.sender}{self.receiver}{self.amount}{self.data}{self.timestamp}"
        return hashlib.sha256(tx_string.encode()).hexdigest()

class Block:
    def __init__(self, shard_id: int, previous_hash: str):
        self.shard_id = shard_id
        self.timestamp = time.time()
        self.transactions: List[Transaction] = []
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()
        self.block_size = 0
        self.block_interval = 0
        self.max_transactions = 100
        
    def add_transaction(self, transaction: Transaction):
        self.transactions.append(transaction)
        
    def calculate_hash(self) -> str:
        block_string = json.dumps({
            'shard_id': self.shard_id,
            'timestamp': self.timestamp,
            'transactions': [t.to_dict() for t in self.transactions],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest() 
    
    def is_full(self) -> bool:
        return len(self.transactions) >= self.max_transactions