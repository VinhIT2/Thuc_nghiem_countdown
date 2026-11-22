# Experiment card — vinh-eval-ood-v2-001 (giai đoạn E: test-ID + 4 trục OOD)

- **Người phụ trách:** Bùi Quang Vinh
- **Trạng thái:** Completed
- **Ngày:** 2026-11-22
- **Protocol đã khóa:** greedy 1 mẫu, `max_new_tokens 128`, revision `7ae5576…`, seed 42,
  `--expected-n 1`; chi phí lấy theo **median** (xem README mục "Cách đọc chỉ số chi phí")

## Câu hỏi

- **RQ:** Model sau SFT/GRPO tổng hoá thế nào trên 4 trục ngoài phân phối, trục nào gãy?
- **Giả thuyết đến từ chẩn đoán ở pilot:** `generator.py:92` ép mọi reference của ID có
  `depth == 2` và 4 lá, nên model học **khuôn hình thức** chứ không học luật số học.
  Dự đoán: gãy mạnh nhất ở trục **arity** (5 số), vì đó là trục phá vỡ khuôn 4 lá.

## Data (data-v2, đã đóng băng, 0 trùng nội bộ)

| Tập | n | số/đề | depth ref | có phân số |
|---|---:|---|---|---|
| `test-id` | 40 | 4 (1–10) | 2 | 0/40 |
| `ood-range` | 60 | 4 (11–20) | 2 | 0/60 |
| `ood-arity` | 60 | **5** | 3 | 0/60 |
| `ood-fraction` | 60 | 4 | 2 | 60/60 |
| `ood-depth` | 60 | 4 | 3 | 0/60 |

## Kết quả — `pass@1`

| Model | test-id | ood-range | ood-arity | ood-fraction | ood-depth |
|---|---:|---:|---:|---:|---:|
| base | 0.000 (0/40) | 0.000 (0/60) | 0.000 (0/60) | 0.000 (0/60) | 0.000 (0/60) |
| SFT ckpt-80 | 0.050 (2/40) | 0.133 (8/60) | **0.000 (0/60)** | 0.033 (2/60) | 0.050 (3/60) |
| GRPO 80 bước | 0.075 (3/40) | 0.117 (7/60) | 0.000 (0/60) | 0.033 (2/60) | 0.067 (4/60) |
| GRPO 240 bước | 0.050 (2/40) | 0.133 (8/60) | 0.000 (0/60) | 0.017 (1/60) | 0.050 (3/60) |

## Kết quả — `valid_output_rate` (viết được biểu thức hợp lệ) và kiểm định Fisher

So với `test-id` (SFT ckpt-80 đạt **40/40**):

| Tập OOD | valid | Fisher p | Kết luận |
|---|---:|---:|---|
| `ood-range` | 60/60 | 1.00 | không khác |
| **`ood-arity`** | **0/60** | **7.3e-29** | **gãy hoàn toàn** |
| `ood-fraction` | 58/60 | 0.52 | không khác |
| `ood-depth` | 58/60 | 0.52 | không khác |

## Cơ chế, chứng minh bằng bản ghi thô

Đếm số lượng hằng số trong biểu thức model sinh ra trên **toàn bộ 60 đề 5 số**:

```
{4: 60}      <- 60/60 dung DUNG 4 so, khong le
```

Ví dụ thật: đề `[10, 8, 1, 9, 4] -> 50`, model trả `FINAL: ((10 + 8) + (1 + 9))` —
dùng 4 trong 5 số, **bỏ số 4**, và vẫn giữ nguyên khuôn `((a op b) op (c op d))` chiều sâu 2.
Verifier bắt đúng 100 % với `error_code = NUMBERS`.

## Kết luận

1. **OOD gap là có thật nhưng chỉ tồn tại trên MỘT trục: số lượng số.** Range, phân số trung
   gian, và độ sâu cây đều **không** tạo khác biệt có ý nghĩa (p ≥ 0.52). Đây là kết quả
   *tích cực* cho phương pháp luận: bộ 4 trục OOD của thầy đã tách được đúng một biến gây
   gãy, thay vì trả về "mọi thứ đều kém hơn".
2. **Nguyên nhân không phải số học khó hơn, mà là khuôn hình thức bị phá.** Với arity, model
   không "giải kém đi" — nó **mất khả năng viết đầu ra hợp lệ** (0/60), vì luật "dùng mỗi số
   đúng một lần" phải nhân bản theo số lượng đầu vào, còn khuôn ngoặc thì không.
3. **Giới hạn của chính kết luận này:** `ood-arity` đồng thời đổi **hai** thuộc tính — 5 số
   *và* depth 3 (`generator.py:89`) — nên "arity" và "độ sâu lời giải" chưa tách được ở
   đây. Muốn tách cần một profile 5-số-depth-2, tức sửa `matches()`/`PROFILES` trong
   `generator.py` → phải tăng `generator_version` và đăng ký experiment mới.
4. `pass@1` trên các tập khác **không** khác biệt có ý nghĩa (mọi Fisher p ≥ 0.06 với
   test-id). Con số 0.133 của `ood-range` từng làm tôi đoán "OOD dễ hơn ID" ở báo cáo nhanh
   — với p = 0.31 thì **không được phép phát biểu như vậy**, và đó là lý do mục này viết
   sau khi chạy kiểm định chứ không viết từ bảng số.
5. GRPO (80 và 240 bước) không cải thiện ở bất kỳ trục OOD nào so với SFT init — nhất quán
   với kết luận của `vinh-grpo-240-001`.

## Artifacts

20 log `logs/v2-E-<tap>-<model>.jsonl` (+ `.manifest.json`), 20 metrics
`logs/metrics-E-<tap>-<model>.json`. `config_id` dạng `E-<tap>-<model>`.
