# Experiment Card — vinh-grpo-v2-rewardmode-001

- **Người phụ trách:** Bùi Quang Vinh
- **Trạng thái:** Completed (kèm run 240 bước ở card `vinh-grpo-240-001`)
- **Ngày:** 2026-11-22
- **Config:** `configs/grpo_v2_binary.yaml`, `configs/grpo_v2_shaped.yaml` — **chỉ differ mỗi
  `reward_mode`**, mọi thứ khác y hệt (adapter khởi tạo, data, seed, 80 bước, G=4, beta 0.04)

## Câu hỏi và giả thuyết

- **RQ:** Reward `shaped` (0.1 cho biểu thức hợp lệ nhưng sai đích) có tạo ra tiến bộ thật,
  hay chỉ tạo ra **chỉ số đẹp**?
- **Giả thuyết cần kiểm, không phải mục tiêu phải đạt:** nếu shaped chỉ mua định dạng thì
  `group_variance_rate` và `reward_mean` tăng mạnh trong khi `solve_rate` không đổi.

## Model và init

- Base `Qwen2.5-0.5B-Instruct` rev `7ae5576…`; **adapter đầu vào `logs/sft-v2-001/checkpoint-80`**
  (checkpoint đã chọn trên validation, không phải final-adapter)
- KL reference theo TRL 0.15.2 = **base model khi tắt adapter**, *không phải* SFT snapshot
  (`grpo/train.py:50-52`). Đây là lựa chọn phương pháp, phải ghi rõ khi so sánh với tài liệu khác.

## Data / Verifier

- `data-v2` train 320 / validation 40; `verifier-v1`; reward = `src/reward.py` mode binary hoặc shaped
- `pytest -q` 45 passed

## Training

- TRL `GRPOTrainer` (subclass `AuditedGRPOTrainer` ghi step event), G=4, batch 4/1 GPU,
  lr 1e-6 constant, beta 0.04, temp 0.8, `max_completion_length` 96, 80 bước
- binary: `wall 218 s`, peak 1.18 GB · shaped: `wall 227 s`, peak 1.23 GB

## Kết quả — tín hiệu huấn luyện (80 nhóm train)

| Chỉ số | binary | shaped | Δ |
|---|---:|---:|---:|
| `reward_mean` | 0.0500 | **0.1363** | **+0.086** |
| `group_variance_rate` | 0.175 | **0.463** | **+0.288** |
| `valid_rate` | 0.925 | 0.9125 | **−0.013** |
| `solve_rate` | 0.050 | 0.050 | **0.000** |
| KL min/median/max | 0.93/1.03/2.32 | 0.94/1.04/**2.95** | drift mạnh hơn |

Phân tách `reward_mean` của shaped: `0.05·1.0 + 0.875·0.1 = 0.1375` ≈ 0.136 quan sát
→ **64 % tổng phần thưởng là tiền thưởng định dạng**, không phải tiền giải đúng.

## Kết quả — độ chính xác thật (validation 40 đề)

| Cấu hình | greedy pass@1 | sampling pass@1 | pass@4 | B3 sel4 |
|---|---:|---:|---:|---:|
| SFT ckpt-80 (init) | 0.175 | 0.031 | 0.100 | 0.100 |
| GRPO binary 80 | 0.175 | 0.031 | 0.100 | 0.100 |
| GRPO shaped 80 | 0.175 | 0.031 | 0.100 | 0.100 |

**Ba con số giống hệt nhau, và đây KHÔNG phải lỗi nạp adapter.** Đã xác minh:

- `adapter_model.safetensors` của 3 run có sha256 **khác nhau**
  (`d96a159b…`, `6671f94e…`, `edbb5265…`).
- Output greedy chỉ trùng nhau 27–35/40 đề, sampling trùng 124–136/160 → trọng số thật sự khác.
- Nhưng **tập đề giải được** thì khác nhau:
  ```
  SFT    : {1, 7,10,12,20,24,35}
  BINARY : {1, 7,10,12,20,24,31}   (mat 35, duoc 31)
  SHAPED : {4, 7,10,12,20,24,35}   (mat 1,  duoc 4)
  ```
  Mỗi RL chỉ **trao đổi đúng 1 đề**. McNemar với cặp bất đối xứng 1–1 → **p = 1.0**.

## Artifacts

`logs/grpo-v2-{binary,shaped}-001/{manifest,runtime,trainer_state,reward-diagnostics,reward-outputs}.json*`,
`logs/v2-grpo-{binary,shaped}-{greedy,pool4}.jsonl` + `logs/metrics-v2-grpo-*.json`

## Lỗi và bất thường

- KL nhảy lên 2.32 (binary) và 2.95 (shaped) ở vài bước cuối trong khi median ~1.0:
  `beta = 0.04` có thể quá yếu để ghì policy. Chưa chạy ablation `beta`.
- `reward-outputs.jsonl` ghi lại từng completion thô, nên kết luận ở trên kiểm chứng lại được
  mà không cần train lại.

## Kết luận giới hạn

1. **`shaped` là reward hacking đo được, không phải nghi ngờ:** tăng 2.7 lần lượng tín hiệu
   học được và tăng drift, đổi lại **0.000** cải thiện độ chính xác. Protocol cuối nên giữ
   **binary**.
2. **GRPO 80 bước ở lr 1e-6 không cử động được `pass@1`** vì `solve_rate` khi sinh mẫu chỉ
   0.05 → 82 % nhóm có advantage = 0. Đây là bài toán **tín hiệu thưa**, nằm ở dữ liệu/init,
   không phải ở thuật toán.
3. Kết luận "RLVR không giúp gì" là **quá xa** với bằng chứng này: 80 bước, 1 seed, 40 đề,
   1 learning rate. Run 240 bước (`vinh-grpo-240-001`) mới trả lời được câu "thiếu bước
   hay thiếu tín hiệu".
