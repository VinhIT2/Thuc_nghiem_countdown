# Experiment Card — vinh-sft-v2-001

- **Người phụ trách:** Bùi Quang Vinh
- **Trạng thái:** Completed (checkpoint 80 đã chọn; control đang chạy ở card `vinh-sft-control96-001`)
- **Ngày:** 2026-11-22
- **Code:** commit `d83de53`; config `configs/sft_lora_v2.yaml`

## Câu hỏi và giả thuyết

- **RQ:** Sau khi baseline hết đường (B3 = 0.000), SFT LoRA có nâng được `pass@1` không,
  và **checkpoint nào** nên chọn?
- **Thay đổi so với pilot SFT (96 đề, 24 bước):** data 96 → **320 đề**, 24 → **80 bước**
  (= đúng 1 epoch), eval/save mỗi 20 bước.
- **Kết quả kỳ vọng để kiểm tra:** nếu SFT chỉ dạy định dạng thì `valid_output_rate` tăng
  còn `pass@1` vẫn ~0. Nếu `pass@1` tăng thì SFT học được nhiều hơn bắt chước khuôn.

## Model

- Base `Qwen/Qwen2.5-0.5B-Instruct` rev `7ae5576…`; adapter đầu vào: không (SFT từ base)
- LoRA r=8, alpha=16, dropout 0.05, target `q_proj,v_proj` → **540.672/494.573.440 tham số
  huấn luyện = 0.109 %**, file adapter 2.17 MB

## Data

- `data-v2/train.jsonl` 320 đề, sha256 `5051d7ab15f2461e287645ef505be190f962063379ef77421ab19299ca82ce69`
- `data-v2/validation.jsonl` 40 đề, sha256 `0366a985…`; **test-id không được chạm tới**
  (`training_common.py:29-30` chặn)

## Verifier

- `verifier-v1`, 45 passed. SFT không dùng reward.

## Training

- TRL `SFTTrainer`, seed 42, bf16, lr 2e-4 cosine warmup 0.1, batch 1 × accum 4,
  `max_seq_length` 256, gradient checkpointing, 80 bước
- `CompletionCollator` gán nhãn `-100` cho prompt+padding → loss chỉ tính trên `FINAL: …`
- `wall 176.5 s`, `peak_allocated 1.23 GB`, `peak_reserved 1.51 GB`

## Kết quả (validation 40 đề, greedy 1 mẫu)

| Checkpoint | valid | parse fail | pass@1 | CI 95 % | eval_loss |
|---|---:|---:|---:|---|---:|
| 20 | 0.800 | 0.000 | 0.025 | [0.000, 0.075] | 0.5186 |
| 40 | 0.925 | 0.000 | 0.025 | [0.000, 0.075] | 0.4060 |
| 60 | 0.950 | 0.000 | 0.100 | [0.025, 0.200] | 0.3968 |
| **80 (chọn)** | **0.975** | 0.000 | **0.175** | [0.075, 0.300] | 0.3964 |

Cùng ngân sách N=4 sampling temp 0.8: `pass@4 = 0.100`, `B3 sel4 = 0.100`, `B2 sel4 = 0.050`.
B3 của **model gốc** là **0.000**.

## Artifacts

`logs/sft-v2-001/{manifest.json,runtime.json,trainer_state.json,checkpoint-*/}`,
`logs/v2-sft-c{20,40,60,80}.jsonl`, `logs/metrics-v2-sft-*.json`, `logs/v2-sft80-pool4*`

## Lỗi và bất thường

1. **`eval_loss` không phân biệt được checkpoint:** nó giảm đơn điệu
   (0.519→0.396) nên mọi ckpt đều "tốt hơn" theo loss; tie-break thực tế phải dùng
   `pass@1` khi generate. Ở run này hai tiêu chí tình huống cùng chiều.
2. **Đổi hai biến một lúc** (96→320 đề **và** 24→80 bước). Kết luận "nhiều dữ liệu hơn thì
   SFT giỏi hơn" **chưa được cô lập**; control `vinh-sft-control96-001` (80 bước trên
   96 đề, validation y hệt) mới trả lời được.
3. **Sampling làm giảm độ chính xác:** greedy `pass@1 = 0.175` nhưng temp 0.8 chỉ cho
   `pass@1 = 0.031` và `pass@4 = 0.100`. Giáo huấn: protocol nào cũng phải nói rõ chế độ
   giải mã, và điều này ảnh hưởng trực tiếp tới `temperature` của GRPO.
4. Pilot `data/` (96 đề, 24 bước) cho kết luận **ngược** — "SFT chỉ mua định dạng". Số liệu
   đó đúng nhưng phạm vi hẹp; giữ lại trong repo làm bằng chứng quá trình.

## Kết luận giới hạn

SFT 1 epoch trên 320 đề nâng `pass@1` từ 0.000 (mọi baseline inference-time) lên **0.175**
ở **cùng chi phí một mẫu**, CI loại 0. Nhưng: model 0.5B, 1 epoch, 40 đề, chưa control
lượng dữ liệu, chưa đo test-ID/OOD. Không được phát biểu thành "SFT giải quyết bài toán".
