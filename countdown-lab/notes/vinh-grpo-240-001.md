# Experiment Card — vinh-grpo-240-001

- **Người phụ trách:** Bùi Quang Vinh
- **Trạng thái:** Completed
- **Ngày:** 2026-11-22
- **Config:** `configs/vinh-grpo_binary-240step.yaml` — copy của `configs/grpo_v2_binary.yaml`,
  **chỉ đổi `max_steps` 80→240** và tần suất eval/save 40→120

## Câu hỏi

`vinh-grpo-v2-rewardmode-001` cho thấy GRPO 80 bước không cử động được `pass@1`
(7/40, trao đổi đúng 1 đề, McNemar p = 1.0). Hai giả thuyết cạnh tranh:

- **H1 — thiếu bước:** 80 bước quá ngắn, RLVR cần chạy dài hơn mới thấy hiệu ứng.
- **H2 — thiếu tín hiệu:** `solve_rate` khi sinh mẫu chỉ ~5 % → 78–82 % nhóm có
  `group_variance = 0` → phần lớn step không có gradient, chạy dài hơn cũng vô ích.

- **Kết quả kỳ vọng để kiểm tra (không phải mục tiêu phải đạt):** nếu H1 đúng thì `pass@1`
  và `solve_rate` phải **tăng theo khúc train**; nếu H2 đúng thì các đường này phẳng.

## Model / init

`Qwen2.5-0.5B-Instruct` rev `7ae5576…` + adapter `logs/sft-v2-001/checkpoint-80`
(chính là model có `pass@1 = 0.175` — mốc phải đánh bại). KL reference = base khi tắt
adapter theo TRL 0.15.2.

## Data / Verifier

`data-v2` train 320 / validation 40; `verifier-v1`; `reward_mode: binary`; 45 test passed.

## Training

G=4, batch 4/1 GPU, lr 1e-6 constant, beta 0.04, temp 0.8, `max_completion_length` 96.
`wall 524 s` (8.7 phút), `peak 1.15 GB`, `reserved` không đổi so với run 80 bước.

## Kết quả — hành vi huấn luyện theo 3 khúc

| Khúc step | `reward_mean` | `solve_rate` | `valid_rate` | `group_variance_rate` |
|---|---:|---:|---:|---:|
| 1–80 | 0.0656 | 0.0656 | 0.894 | 0.237 |
| 81–160 | 0.0563 | 0.0563 | 0.909 | 0.200 |
| 161–240 | 0.0656 | 0.0656 | 0.909 | 0.225 |

Đường **phẳng hoàn toàn**. Self-check: với `binary`, `reward_mean == solve_rate` tới từng
chữ số thập phân ở cả ba khúc → xác nhận toàn bộ phần thưởng đến từ verifier, không có
đường tắt.

## Kết quả — độ chính xác thật (validation 40 đề, greedy 1 mẫu)

| Mô hình | pass@1 | CI 95 % | Tập đề giải được | So với init |
|---|---:|---|---|---|
| SFT ckpt-80 (init) | **0.175** | [0.075, 0.300] | `{1,7,10,12,20,24,35}` (7) | — |
| GRPO 80 bước | 0.175 | [0.075, 0.300] | `{1,7,10,12,20,24,31}` (7) | +1 / −1 |
| **GRPO 240 bước** | **0.100** | [0.025, 0.200] | `{1,10,12,24}` (4) | **+0 / −3** |

KL run 240 bước được kiểm soát *tốt hơn* run 80: min/median/max `0.92/1.01/1.24` so với
`0.93/1.03/2.32`.

## Kết luận

**H2 thắng, H1 bị bác.** Gấp 3 lần số bước không tạo tiến bộ; ngược lại `pass@1` **giảm**
0.175 → 0.100 và model **mất 3 đề, thêm 0 đề** (McNemar 0-vs-3, p = 0.25 — chưa có ý nghĩa
thống kê nhưng sai lệch theo hướng âm). Đây **không** phải sụp đổ do KL (KL ổn định hơn
run ngắn), mà là **đi lang thang**: khi 3/4 mẫu trong nhóm cùng sai nhưng hợp lệ, advantage
nhóm chủ yếu khuếch đại hành vi đa số — tức *nghiêng phân phối* chứ không *tìm lời giải*.

Hệ quả cho đề tài: **nút thắt của RLVR ở quy mô này là mật độ tín hiệu**, không phải ngân
sách tính toán. Muốn GRPO có cơ hội, theo thứ tự chi phí tăng dần:

1. Init mạnh hơn (SFT dài/đa dạng hơn) để `solve_rate` khi sinh mẫu vượt ~15 %.
2. Curriculum: bộ đề sao cho model *thỉnh thoảng* giải đúng — hiện `group_variance = 0` ở
   ~78 % nhóm.
3. `num_generations` lớn hơn (G=4 → 8) nhưng phải trả giá batch/VRAM.
4. Ablation `beta` và `learning_rate` — chưa chạy; `lr = 1e-6` có thể quá nhỏ để học nhưng
   đủ lớn để làm hỏng đề mong manh.

## Giới hạn

1 seed; 40 đề; greedy duy nhất (chưa đo sampling N=4 cho run này); 240 bước vẫn chỉ là
smoke so với các run RLVR trong tài liệu (hàng nghìn bước); model 0.5B. Kết luận chỉ áp dụng
cho *chế độ tín hiệu thưa* như ở đây, không phải phủ định GRPO nói chung.
