# DQNSB - Deep Q-Network Sharding Blockchain

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

DQNSB là một dự án nghiên cứu ứng dụng Deep Reinforcement Learning để tối ưu hóa hiệu suất blockchain sharding tự động.

## 🌟 Tính năng chính

- **Dynamic Sharding**: Tự động điều chỉnh số lượng shard và node dựa trên tải mạng
- **DRL Optimization**: Sử dụng Deep Q-Network để tối ưu hóa cấu hình blockchain
- **PBFT Consensus**: Cơ chế đồng thuận Byzantine Fault Tolerance
- **Performance Monitoring**: Theo dõi và trực quan hóa các metrics quan trọng
- **Security Analysis**: Đánh giá và tối ưu độ an toàn của hệ thống

## 🚀 Bắt đầu

### Yêu cầu

- Python 3.8+
- PyTorch 2.0+
- Các thư viện khác trong requirements.txt

### Cài đặt

```bash
# Clone repository
git clone https://github.com/fuondai/DQNSB-Blockchain
cd DQNSB-Blockchain

# Cài đặt dependencies
pip install -r requirements.txt
```

### Chạy thử

```bash
# Training DRL agent
python main.py --train \
    --episodes 100 \
    --transactions 1000 \
    --malicious 0.1 \
    --security-weight 0.5 \
    --batch-size 128 \
    --learning-rate 0.0005 \
    --memory-size 50000

# Train và giả lập nhanh môi trường
python main.py --train --episodes 100 --transactions 1000 --malicious 0.1 --security-weight 0.5 --batch-size 128 --learning-rate 0.0005 --memory-size 50000

# Đánh giá mô hình [IN PROGRESS ⚠️]
python main.py --evaluate \
    --model checkpoints/best_model.pth
```

## 📊 Kết quả Thực nghiệm

### Performance Metrics

| Metric      | Static | DRL   | Improvement |
| ----------- | ------ | ----- | ----------- |
| TPS         | 35     | 39.28 | +12.2%      |
| Security    | 0.65   | 0.75  | +15.4%      |
| Stability   | 0.70   | 0.95  | +35.7%      |
| Cross-shard | N/A    | 95%   | N/A         |

### Cross-shard Performance

- Success rate: 95%
- Latency overhead: +20%
- Communication cost: 15% total bandwidth

### Training Evolution

- Early TPS: 53-62 (unstable)
- Final TPS: 39-40 (stable)
- Security Score: 0.75 (consistent)
- Cross-shard success rate: 95%

## 🔧 Cấu trúc dự án [IN PROGRESS ⚠️]

```
DQNSB/
├── docs/
│   ├── overview.md
│   └── technical_details.md
│
├── src/
│   ├── blockchain/
│   │   ├── __init__.py
│   │   ├── block.py
│   │   ├── blockchain.py
│   │   ├── directory_committee.py
│   │   ├── dynamic_blockchain.py
│   │   ├── shard.py
│   │   └── static_blockchain.py
│   │
│   ├── consensus/
│   │   ├── __init__.py
│   │   └── pbft.py
│   │
│   ├── drl/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── environment.py
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   └── network.py
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py
│   │
│   └── config.py
│
├── tests/
│   ├── __init__.py
│   ├── test_blockchain.py
│   ├── test_consensus.py
│   └── test_drl.py
│
├── plots/
│   └── run_20241224_020600/
│       ├── action_distribution.png
│       ├── config_impact.png
│       ├── performance_comparison.png
│       ├── training_metrics.png
│       └── tps_size_comparison.png
│
├── README.md
├── requirements.txt
└── setup.py
```

## 📖 Documentation

Chi tiết documentation có thể tìm thấy trong thư mục [docs](docs/).

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Xem [CONTRIBUTING.md](CONTRIBUTING.md) để biết thêm chi tiết.

## 📝 License

MIT License

## 📧 Liên hệ

- **Author**: Fuon Dai
- **Email**: fuondai1314@gmail.com

## 🙏 Ghi nhận

Dự án này được phát triển dựa trên các nghiên cứu và mã nguồn mở sau:

- [DQN Paper](https://ieeexplore.ieee.org/document/9133069)
- [Ethereum Sharding](https://github.com/ethereum/sharding/tree/master)
