class Config:
    # Network Configuration
    INITIAL_SHARDS = 4
    MAX_SHARDS = 8
    NODES_PER_SHARD = 12
    
    # Block Configuration
    BLOCK_SIZES = [800, 1200, 1600]
    BLOCK_INTERVALS = [2, 3, 4]
    MAX_TRANSACTIONS_PER_BLOCK = 50
    
    # DRL Configuration
    BATCH_SIZE = 256
    MEMORY_SIZE = 100000
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_MIN = 0.1
    EPSILON_DECAY = 0.997
    LEARNING_RATE = 0.0001
    
    # PER Configuration
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_INCREMENT = 0.001
    
    # Network Architecture
    HIDDEN_SIZES = [1024, 512, 256]
    DROPOUT_RATE = 0.15
    
    # Training Configuration
    TRAINING_EPISODES = 5000
    STEPS_PER_EPISODE = 400
    SECURITY_WEIGHT = 0.6
    TPS_WEIGHT = 0.4
    
    # Optimizer Configuration
    WEIGHT_DECAY = 0.01
    GRAD_CLIP = 0.5
    MIN_LR = 1e-6
    LR_PATIENCE = 10
    LR_FACTOR = 0.5
    
    # Simulation Configuration
    SCENARIOS = {
        'normal': {'malicious_ratio': 0.1, 'tx_size': 800},
        'high_malicious': {'malicious_ratio': 0.2, 'tx_size': 800},
        'large_tx': {'malicious_ratio': 0.1, 'tx_size': 1200},
        'worst_case': {'malicious_ratio': 0.2, 'tx_size': 1200}
    }
    
    # Checkpoint Configuration
    SAVE_FREQUENCY = 50
    CHECKPOINT_DIR = 'checkpoints'
    
    # Evaluation Configuration
    EVAL_FREQUENCY = 25
    EVAL_EPISODES = 15
    
    # DRL parameters
    HIDDEN_SIZES = [512, 256, 128]
    DROPOUT_RATE = 0.2
    MEMORY_SIZE = 50000
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.995
    LEARNING_RATE = 0.0003
    WEIGHT_DECAY = 0.01
    LR_FACTOR = 0.5
    LR_PATIENCE = 10
    MIN_LR = 1e-6
    GRAD_CLIP = 0.5
    
    # Environment parameters
    TARGET_TPS = 80
    TARGET_SECURITY = 0.85
    MIN_TPS_THRESHOLD = 40
    MIN_SECURITY_THRESHOLD = 0.65
    SHARD_PENALTY_FACTOR = 0.12
    
    @staticmethod
    def get_scenario_config(scenario_name: str):
        return Config.SCENARIOS.get(scenario_name, Config.SCENARIOS['normal'])