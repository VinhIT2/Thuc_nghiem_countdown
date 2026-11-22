# Experiment Card — vinh-baseline-v2-001

- **Người phụ trách:** Bùi Quang Vinh
- **Trạng thái:** Completed
- **Ngày:** 2026-11-22
- **Code:** commit `d83de53` (src/ nguyên bản từ `countdown-code-bundle.json`, không sửa)

## Câu hỏi và giả thuyết

- **RQ:** Model chưa train, dùng bao nhiêu tiền suy luận (số mẫu N) thì mua được bao nhiêu
  độ chính xác, khi có verifier chọn giúp?
- **Thay đổi so với baseline pilot:** validation 12 đề → **40 đề** (`data-v2`), và chạy
  **replicate bằng seed thứ hai** vì nghi ngờ kết quả 12 đề là nhiễu.
- **Kết quả kỳ vọng để kiểm tra:** `B3 == pass@4` của cùng pool. Nếu hai số lệch nhau thì
  parser hoặc `select()` có bug, không phải model kém.

## Code và môi trường

- Python 3.13.12; `logs/environment.lock-cuda.txt`; GPU RTX 3050 Laptop 4 GB, driver 610.60 (`logs/hardware.txt`)
- torch 2.6.0+cu124, transformers 4.49.0 (baseline không dùng trl/peft train)

## Model

- `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`
- Adapter: không (base)

## Data

- `data-v2/` generator `generator-v1-reference-profiles`, seed 100
- validation: 40 đề, sha256 `0366a98522a5b1b705f8cfddb6cc0cb609102b0dcc3dd4a2bf25ba35b1e55b6a`

## Verifier

- `verifier-v1` / `parser-v1`; `pytest -q` = **45 passed** trước khi chạy

## Prompt và decoding

- B0 `--style direct`; B1 và pool `--style fewshot`; `max_new_tokens 128`, temp 0.8 / top-p 0.95
- n = 1 (B0, B1) và n = 4 (pool → B2 selected, B3 selected); seed 42 và **1234**

## Kết quả

| Cấu hình | valid | parse fail | pass@1 | pass@4 | sel (B2/B3) | token/đề | s/đề |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 direct, 1 mẫu | 0.000 | 1.000 | 0.000 | — | — | 10.2 | 0.52 |
| B1 few-shot, 1 mẫu | 0.425 | 0.100 | 0.025 | — | — | 14.3 | 3.17 |
| pool N=4 few-shot seed42 | 0.419 | 0.075 | 0.000 | 0.000 | — | 63.1 | **868.88 (hỏng, xem mục Lỗi)** |
| pool N=4 few-shot seed1234 | 0.350 | 0.094 | 0.000 | **0.000** | — | 62.0 | 3.56 |
| B2 vote, N=4 | 0.900 | 0.025 | — | — | 0.000 | 63.1 | (nhiễm) |
| B3 verifier, N=4 | 0.900 | 0.025 | — | — | 0.000 | 63.1 | (nhiễm) |

## Artifacts

- `logs/v2-b0.jsonl`, `logs/v2-b1.jsonl`, `logs/v2-pool4.jsonl`, `logs/v2-pool4-s1234.jsonl`
  (+ `.B2.selected.jsonl`, `.B3.selected.jsonl`, `.manifest.json`)
- `logs/metrics-v2-*.json`

## Lỗi và bất thường

1. **`mean_seconds_per_problem = 868.88` của `v2-pool4.jsonl` là số rác.** Một record có
   `seconds = 34359.66` trong khi median toàn pool 1.25 s. Nguyên nhân: máy tính **ngủ qua
   đêm** giữa chừng run, và `time.perf_counter()` trên Windows vẫn đếm khi suspend.
   Chỉ tiêu chí chi phí bị nhiễm; `valid/pass@k/error_counts` vẫn dùng được vì verifier
   chấm lại từng bản ghi (`eval.py:36`). Bản replicate seed 1234 đo lại đúng: **3.56 s/đề**;
   con số được phép trích cho log nhiễm là **median 5.03 s/đề** (trimmed mean 7.05).
   Toàn bộ 59 log đã quét: **5 log nhiễm**, mỗi log chỉ 1–3 bản ghi. Quy tắc đọc chỉ số chi
   phí ghi ở `README.md` mục "Cách đọc chỉ số chi phí" — nghiêm cấm sửa log vì `eval.py`
   không kiểm tra trường `seconds`, sửa xong không ai phát hiện được.
2. Trần B3 trên 40 đề = **0.000**, tức con số **0.083 của pilot `data/` (12 đề) là một câu
   ăn may**, không tái lập ở cả hai seed. Đây là lý do không được kết luận cross-method
   trên validation 12 đề.

## Kết luận giới hạn

- Sistem tự kiểm `B3 == pass@4` khớp (0.000 == 0.000): verifier và selection hoạt động đúng.
- Trên 40 đề, **mọi kỹ thuật inference-time (few-shot, vote, best-of-4 verifier) đều không
  giải được đề nào** với model 0.5B gốc. Muốn có tín hiệu phải đổi model-weight (SFT/RL),
  không phải đổi cách chọn mẫu.
- Chưa đo trên test-ID/OOD (xem giai đoạn E).
