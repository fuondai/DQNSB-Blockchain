import sys
print("Python path:", sys.path)

try:
    import argparse
    print("Imported argparse")
    import numpy as np
    print("Imported numpy")
    from src.blockchain.directory_committee import DirectoryCommittee
    print("Imported DirectoryCommittee")
    from src.drl.agent import DRLAgent
    print("Imported DRLAgent")
    from src.drl.environment import BlockchainEnvironment
    print("Imported BlockchainEnvironment")
    from src.simulation.network import NetworkSimulator
    print("Imported NetworkSimulator")
    from src.visualization.plots import BlockchainVisualizer
    print("Imported BlockchainVisualizer")
    from src.blockchain.static_blockchain import StaticShardedBlockchain
    print("Imported StaticShardedBlockchain")
    from src.blockchain.block import Transaction
    print("Imported Transaction")
    from src.config import Config
    print("Imported Config")
    import os
    print("Imported os")
except Exception as e:
    print("Import error:", str(e))
    sys.exit(1)

# Tạo thư mục checkpoints nếu chưa tồn tại
if not os.path.exists(Config.CHECKPOINT_DIR):
    os.makedirs(Config.CHECKPOINT_DIR)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--transactions', type=int, default=2000)
    parser.add_argument('--malicious', type=float, default=0.3)
    parser.add_argument('--security-weight', type=float, default=0.6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--memory-size', type=int, default=100000)
    return parser.parse_args()

def evaluate_agent(env, agent, num_episodes: int = 3, max_steps: int = 1000):
    """Evaluate agent performance với giới hạn số bước"""
    print("\nĐánh giá agent...")
    eval_rewards = []
    eval_tps = []
    eval_security = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        episode_tps = []
        episode_security = []
        done = False
        
        while not done and steps < max_steps:
            action = agent.act(state, noise_scale=0)  # Tắt noise khi evaluate
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            episode_tps.append(info['tps'])
            episode_security.append(info['security'])
            
            state = next_state
            steps += 1
            
            if steps % 200 == 0:
                print(f"Episode {episode}: {steps}/{max_steps} steps")
                print(f"Current TPS: {info['tps']:.2f}, Security: {info['security']:.2f}")
        
        eval_rewards.append(total_reward)
        eval_tps.append(np.mean(episode_tps))
        eval_security.append(np.mean(episode_security))
    
    avg_reward = np.mean(eval_rewards)
    avg_tps = np.mean(eval_tps)
    avg_security = np.mean(eval_security)
    
    print(f"Kết quả đánh giá ({num_episodes} episodes):")
    print(f"Reward trung bình: {avg_reward:.2f}")
    print(f"TPS trung bình: {avg_tps:.2f}")
    print(f"Security trung bình: {avg_security:.2f}")
    print("----------------------------------------")
    
    return avg_reward

def train_agent(env, agent, num_episodes, num_transactions):
    # Khởi tạo train_metrics dictionary đúng cách
    train_metrics = {
        'rewards': [],  # Đổi từ 'reward' thành 'rewards'
        'tps': [],
        'security': [],
        'epsilon': []
    }
    
    best_reward = -np.inf
    patience = 20
    no_improve = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for t in range(num_transactions):
            # Thực hiện action với noise giảm dần
            action = agent.act(state, noise_scale=1.0 - episode/num_episodes)
            next_state, reward, done, info = env.step(action)
            
            # Lưu experience với priority
            agent.remember(state, action, reward, next_state, done, 
                         priority=abs(reward))
            
            state = next_state
            episode_reward += reward
            
            # Train sau mỗi N steps
            if t % 4 == 0:
                agent.replay()
                
        # Cập nhật train_metrics
        train_metrics['rewards'].append(episode_reward)
        train_metrics['tps'].append(info['tps'])
        train_metrics['security'].append(info['security'])
        train_metrics['epsilon'].append(agent.epsilon)
                
        # Early stopping
        if episode_reward > best_reward:
            best_reward = episode_reward
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print("Early stopping triggered")
            break
            
        # Đánh giá và log metrics
        if episode % 10 == 0:
            evaluate_agent(env, agent)
            
    return train_metrics

def compare_performance(env: BlockchainEnvironment, num_episodes: int, num_transactions: int, malicious_ratio: float):
    """So sánh hiệu suất giữa DQN và static blockchain"""
    print("Bắt đầu so sánh hiệu suất...")
    
    # Train DQN agent
    print("Training DRL agent...")
    agent = DRLAgent(env.observation_space.shape[0], env.action_space.n)
    drl_metrics = train_agent(env, agent, num_episodes, num_transactions)
    
    # Evaluate static blockchain
    print("Khởi tạo static blockchain...")
    static_blockchain = StaticShardedBlockchain(
        num_shards=4,
        nodes_per_shard=8,
        malicious_ratio=malicious_ratio
    )
    
    print("Thu thập metrics từ static blockchain...")
    static_metrics = static_blockchain.process_transactions(num_transactions)
    
    # Plot comparison
    print("Vẽ biểu đồ so sánh...")
    visualizer = BlockchainVisualizer()
    visualizer.plot_comparison(drl_metrics, static_metrics)
    
    print("So sánh hoàn tất!")

def main():
    """Main function to run the program"""
    print("Starting program...")
    
    # Parse arguments
    args = parse_args()
    print("Arguments:", args)
    
    # Initialize environment
    print("Initializing environment...")
    env = BlockchainEnvironment(
        min_shards=2,
        max_shards=8,
        min_nodes_per_shard=4,
        max_nodes_per_shard=16,
        security_weight=args.security_weight,
        malicious_ratio=args.malicious
    )
    
    # Khởi tạo agent
    agent = DRLAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        memory_size=args.memory_size
    )
    
    # Training mode
    if args.train:
        print("Bắt đầu training...")
        train_metrics = train_agent(env, agent, args.episodes, args.transactions)
                
        # Vẽ biểu đồ
        visualizer = BlockchainVisualizer()
        visualizer.plot_training_metrics(train_metrics)
        
        # Kiểm tra xem có action history không trước khi vẽ
        if hasattr(env, 'action_history') and env.action_history:
            visualizer.plot_action_distribution(env.action_history)
            
        # So sánh với static blockchain
        static_blockchain = StaticShardedBlockchain(4, 10, args.malicious)
        static_result = static_blockchain.process_transactions(args.transactions)
        visualizer.plot_performance_comparison(train_metrics, static_result)
        
    print("\nProgram completed successfully!")

if __name__ == "__main__":
    main()
 