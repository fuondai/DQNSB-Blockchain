# DQNSB Performance Analysis

## 1. Baseline Comparison

### TPS Performance

- Static: 35 TPS
- DRL: 39.28 TPS
- Improvement: +12.2%

### Security Metrics

- Static: 0.65 score
- DRL: 0.75 score
- Improvement: +15.4%

### Stability Analysis

- Static: 0.70 stability index
- DRL: 0.95 stability index
- Improvement: +35.7%

## 2. Cross-shard Analysis

### Success Rate

- Overall: 95%
- Inter-shard: 92%
- Intra-shard: 98%

### Latency Impact

- Base latency: 2.1s
- Cross-shard overhead: +20%
- Average total: 2.52s

### Resource Usage

- CPU: 45% utilization
- Memory: 60% utilization
- Network: 15% bandwidth

## 3. Detailed Metrics

Chi tiết documentation có thể tìm thấy trong thư mục [plots](plots/).
