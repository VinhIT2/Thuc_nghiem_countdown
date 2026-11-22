# Experiment Card — vinh-sft-control96-001

- **Người phụ trách:** Bùi Quang Vinh
- **Trạng thái:** Completed
- **Ngày:** 2026-11-22
- **Config:** `configs/vinh-sft_control96-001.yaml`

## Câu hỏi

`vinh-sft-v2-001` đổi **hai biến cùng lúc** (96→320 đề và 24→80 bước), nên không kết luận
được 0.175 đến từ dữ liệu hay từ số bước. Run này **giữ nguyên 80 bước**, chỉ giảm dữ liệu.

- **RQ:** Lượng dữ liệu và số bước train, cái nào quyết định `pass@1` của SFT?
- **Kết quả kỳ vọng để kiểm tra:** nếu chỉ cần *bước* thì control đạt ~0.175; nếu cần *dữ
  liệu* thì control rơi về quanh 0.025–0.10.

## Data

- `controls/slice96/train96.jsonl` = **96 dòng đầu** của `data-v2/train.jsonl` (đã được
  `generator.build` xáo bằng `seed+1`), sha256 `ee61be6c74a056724b17b210abb96d415ad6ab92cb839d74431c2068eed039f7`
- Nguồn: `data-v2/train.jsonl` sha256 `5051d7ab…`; ghi chú nguyên nhân ở
  `controls/slice96/README.json`
- Validation **giữ y hệt** `data-v2/validation.jsonl` (40 đề, `0366a985…`) để so sánh trực
  tiếp; không giao với train vì slice lấy từ train
- 96 đề / batch hiệu dụng 4 × 80 bước = **3.3 epoch** (khác 1.0 epoch của treatment)

## Training

Giống `sft-v2-001` mọi thứ khác dữ liệu: LoRA r8 q/v, lr 2e-4 cosine, seed 42, bf16,
batch 1 × accum 4, seq 256. `wall 92 s`, `peak 1.14 GB`.

## Kết quả (validation 40 đề, greedy 1 mẫu)

| | valid | parse fail | pass@1 | CI 95 % | eval_loss | đề giải được |
|---|---:|---:|---:|---|---:|---|
| **control** 96 đề × 80 bước | 0.975 | 0.000 | **0.075** | [0.000, 0.175] | 0.4007 | `{1, 13, 24}` |
| treatment 320 đề × 80 bước | 0.975 | 0.000 | **0.175** | [0.075, 0.300] | 0.3964 | `{1,7,10,12,20,24,35}` |

## Kết luận

1. **Định dạng đến từ số bước, không từ lượng dữ liệu.** `valid_output_rate` và
   `parse_failure_rate` **bằng nhau chính xác** (0.975 / 0.000) dù data chênh 3.3 lần.
2. **Độ chính xác theo lượng dữ liệu, nhưng chưa có ý nghĩa thống kê.** 3/40 so với 7/40,
   Fisher exact **p ≈ 0.36**, CI chồng lấn. Đây là bằng chứng *hướng*.
3. **Phát hiện quan trọng nhất về mặt phương pháp: `eval_loss` hai bên gần như bằng nhau
   (0.4007 vs 0.3964) nhưng khả năng giải chênh 2.3 lần.** Không được chọn model bằng loss.
   Control còn train tới 3.3 epoch nên loss thấp *một cách vô hại* — nó nhớ tập nhỏ.
4. **Tính ổn định kém:** tập đề giải được của hai run chỉ giao nhau 2/40 đề
   (`{1, 24}`). Cùng pipeline, cùng seed, chỉ khác lượng dữ liệu.

## Giới hạn

Một seed duy nhất cho mỗi nhánh; 40 đề; slice 96 là 96 dòng đầu (deterministic) chứ không
phải 96 đề rút ngẫu nhiên — điều này làm slice có thể lệch phân phối nhẹ, dù file nguồn đã
được xáo. chưa chạy lại control với seed khác.
