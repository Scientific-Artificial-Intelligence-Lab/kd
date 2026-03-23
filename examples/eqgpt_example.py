"""KD_EqGPT example: wave-breaking PDE discovery with pre-trained GPT.

Demonstrates the EqGPT workflow using pre-trained surrogate models and
GPT-guided RL search on wave-breaking experimental data.

Two modes:
  1. fit_pretrained (default): Load pre-trained surrogates → RL search
  2. retrain_surrogate=True: Train surrogates from scratch → RL search

Requirements:
  - ref_lib/EqGPT_wave_breaking/wave_breaking_data.pkl (199MB observation data)
  - kd/model/eqgpt/gpt_model/PDEGPT_wave_breaking.pt (GPT weights)
  - kd/model/eqgpt/model_save/ (pre-trained surrogate weights, for mode 1)

Usage:
    python examples/eqgpt_example.py
"""

from kd.model import KD_EqGPT

# ============================================================
# 1. Pre-trained mode (default) / 预训练模式（默认）
# ============================================================

# Uses pre-trained surrogate NN weights + pre-trained GPT.
# RL search only — fastest path. ~12 min on MPS, ~3 min on CUDA.
# 使用预训练 surrogate NN 权重 + 预训练 GPT，仅做 RL 搜索。

model = KD_EqGPT(
    optimize_epochs=5,        # RL search epochs / RL 搜索轮数
    samples_per_epoch=400,    # Candidates per epoch / 每轮采样数
    case_filter="N",          # "N" = 12 N-type cases, "all" = 23 cases
    seed=0,                   # Random seed / 随机种子
)

result = model.fit_pretrained()

print("=" * 60)
print("EqGPT Wave-Breaking PDE Discovery Results")
print("=" * 60)
print(f"Best equation:  {result['best_equation']}")
print(f"Best reward:    {result['best_reward']:.4f}")
print()
print("Top-10 equations:")
for i, (eq, rw) in enumerate(zip(result["equations"], result["rewards"])):
    print(f"  {i+1:2d}. [reward={rw:.4f}]  {eq}")

# ============================================================
# 2. Retrain mode (optional) / 重训练模式（可选）
# ============================================================

# Uncomment to train surrogates from scratch before RL search.
# This verifies the full pipeline but takes longer (~18 min on MPS).
# 取消注释可从头训练 surrogate，验证完整 pipeline，但更慢。

# model_retrain = KD_EqGPT(
#     optimize_epochs=5,
#     samples_per_epoch=400,
#     case_filter="N",
#     seed=0,
#     retrain_surrogate=True,       # Train surrogates from scratch
#     surrogate_epochs=50000,       # Full training (default)
#     # surrogate_epochs=5000,      # Quick test (~6x faster)
# )
# result_retrain = model_retrain.fit_pretrained()
# print(f"\nRetrained best: {result_retrain['best_equation']}")
# print(f"Retrained reward: {result_retrain['best_reward']:.4f}")
