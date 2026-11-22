# Nhật ký tuần 01

- **Người phụ trách:** Bùi Quang Vinh
- **Trạng thái:** Open
- **Thời gian:** 2026-11-21 → 2026-11-22

## Mục tiêu tuần

- [x] Đọc hiểu mã nguồn của lab và giải thích được cơ chế từng module
- [x] Dựng môi trường chạy được từ đầu đến cuối (CPU → GPU)
- [x] Baseline đủ 4 mức B0/B1/B2/B3 trên validation có power
- [x] SFT 1 epoch + chọn checkpoint bằng generate, không bằng loss
- [x] GRPO smoke: đo được tín hiệu huấn luyện và thử nghiệm reward mode
- [x] Đánh giá test-ID + 4 trục OOD
- [x] Đưa toàn bộ lên GitHub để đồng nghiệp làm tiếp không cần hỏi
- [ ] Chạy control tách arity/độ sâu cho `ood-arity` (tuần sau)
- [ ] Nhiều seed cho SFT/GRPO để có phương sai (tuần sau)

## Công việc đã làm và bằng chứng

| Công việc | Kết quả | File / commit | Người làm |
|---|---|---|---|
| Trích mã nguồn | 19 file, phát hiện code nằm trong `code-bundle` JSON của HTML, **không** nằm ở repo GitHub | `countdown-code-bundle.json`, `unpack.py` · commit `d83de53` | Vinh |
| Verify bộ lõi | `pytest -q` **45 passed** (Python 3.13), giữ nguyên sau khi đổi torch sang cu124 | `notes/vinh-baseline-v2-001.md` | Vinh |
| Đóng băng dữ liệu | `data/` 240 đề + `data-v2/` **640 đề**, 7 tập đôi một rời nhau | `notes/dataset-card-data-v2.md` | Vinh |
| Môi trường GPU | torch 2.6.0+cu124, bf16 OK, `cap (8,6)`; 6 gói khớp `EXPECTED` | `logs/environment.lock-cuda.txt` | Vinh |
| Baseline B0–B3 | 40 đề, có replicate seed 1234 | `notes/vinh-baseline-v2-001.md` · `52617cd` | Vinh |
| SFT | 4 checkpoint, chọn ckpt-80 trên validation | `notes/vinh-sft-v2-001.md` · `71a7761` | Vinh |
| Control dữ liệu/bước | tách được: format ← số bước, accuracy ← lượng data | `notes/vinh-sft-control96-001.md` | Vinh |
| GRPO binary vs shaped | kết luận reward hacking định lượng | `notes/vinh-grpo-v2-rewardmode-001.md` | Vinh |
| GRPO 240 bước | bác giả thuyết "thiếu bước" | `notes/vinh-grpo-240-001.md` | Vinh |
| Test-ID + OOD | 20 tổ hợp, 1 trục gãy | `notes/vinh-eval-ood-v2-001.md` · `52617cd` | Vinh |
| Chia việc 2 người | `.gitignore` + `.gitattributes` + CONTRIBUTING | commit `55f30ee` | Vinh |

Commit đã push: `d83de53`, `55f30ee`, `cde9496`, `71a7761`, `52617cd`.

## Kết quả định lượng

| | baseline tốt nhất (inference-time) | SFT | GRPO |
|---|---:|---:|---:|
| `pass@1` validation 40 đề | **0.000** (B3 budget-4, 2 seed) | **0.175** (ckpt-80, greedy) | 0.175 (80b) → **0.100** (240b) |
| `valid_output_rate` | 0.900 (cần 4 mẫu) | 0.975 (1 mẫu) | 0.975 |
| Chi phí | 27.1 → 3.6 s/đề (median) | 0.85 s/đề | 0.64–0.76 s/đề |
| test-ID `pass@1` | 0.000 | 0.050 | 0.075 / 0.050 |
| OOD gãy ở đâu | — | **`ood-arity` 0/60, Fisher p = 7.3e-29** | giống SFT |

Self-check đã khớp 3 lần: `B3 == pass@N` trên cùng pool; với reward binary thì
`reward_mean == solve_rate` tới từng chữ số thập phân.

## Lỗi / thí nghiệm thất bại

1. **Kết luận sai vì mẫu nhỏ (tự sửa 2 lần).** Số 0.083 của pilot là 1 câu ăn may trên 12 đề;
   không tái lập trên 40 đề với 2 seed. Lần thứ hai: đoán "SFT làm mất đa dạng output" — sai,
  nguyên nhân thật là khuôn depth-2. Kiểm chứng bằng cách đếm số biểu thức khác nhau trong
  mỗi pool: SFT 4.00/4 mẫu, cao hơn base 3.83/4 — đa dạng không giảm.
2. **GRPO không tiến bộ.** 240 bước làm `pass@1` **giảm** 0.175 → 0.100 (thêm 0 đề, mất 3 đề,
   McNemar p = 0.25). Đường cong 3 khúc phẳng, KL ổn định → không phải nổ, mà là đi lang
   thang do mật độ tín hiệu quá thưa (`group_variance = 0` ở ~78 % nhóm).
3. **`shaped` = reward hacking.** reward tăng 2.7 lần, group variance tăng 2.6 lần,
   `solve_rate` **không đổi 0.050**; 64 % phần thưởng là tiền định dạng.
4. **Lỗi đo lường do máy ngủ, 5/59 log.** `mean_seconds` bị phao tới 10.000 lần. Đã chuyển
   sang median/trimmed mean và ghi thành quy tắc trong README; **không sửa log** vì
   `eval.py` không kiểm tra trường `seconds` nên sửa sẽ không ai phát hiện.
5. **Suýt commit 1.177 MB** (trong đó `model.safetensors` 942 MB, vượt trần 100 MB của
   GitHub) — phát hiện khi đo `git diff --cached`. Đã thêm `.gitignore` và quy trình đo
   trước push.
6. **Bẫy `core.autocrlf=true`** có thể làm lệch sha256 của data so với manifest khi đồng
   nghiệp checkout — đã chặn bằng `.gitattributes` `-text`.
7. Lỗi vận hành của tôi: `--out` thiếu prefix `logs/` khi eval giai đoạn E (đã dọn), và truyền
   `--exclude` chỉ file chưa tồn tại khi sinh OOD (đã sửa thứ tự).

## Quyết định

| Quyết định | Lý do / bằng chứng | Người duyệt |
|---|---|---|
| Giữ `reward_mode: binary` cho protocol cuối | shaped cho +0.086 reward nhưng +0.000 solve | Vinh (chờ GVHD xác nhận) |
| Giao thức chính: greedy 1 mẫu | vừa chính xác hơn sampling (0.175 vs 0.100) vừa rẻ hơn ~3 lần | Vinh |
| Chọn checkpoint bằng `pass@1` trên validation, không bằng `eval_loss` | control chứng minh loss gần bằng nhau (0.4007/0.3964) mà accuracy chênh 2.3 lần | Vinh |
| Tái lập = train lại, không commit adapter/checkpoint | SFT 1 epoch chỉ 176 s; adapter 336 MB không nên ở Git | Vinh |
| `src/` và `data*` coi như bất biến | mọi sửa đổi làm mất nghĩa của hash và số liệu đã công bố | Vinh |

## Rủi ro hiện tại

- Mật độ tín hiệu quá thưa: GRPO khó có cửa tiến bộ nếu init không đạt `solve_rate` cao hơn
  rõ rệt → cần curriculum, không cần thêm bước.
- Power thống kê vẫn yếu: 40 đề, mỗi cấu hình 1 seed.
- `ood-arity` lẫn lộn arity với độ sâu → kết luận tổng hoá mạnh nhất của tuần này **chưa**
  phải là kết luận nhân quả.
- RTX 3050 4 GB giới hạn `num_generations` và batch, nên một số so sánh với tài liệu (G=8/16)
  không thực hiện được trên máy này.

## Mục tiêu tuần sau

- [ ] Control tách arity/độ sâu: thêm profile 5-số-depth-2 với `generator_version` mới, sinh
      `data-v3`, chạy lại 4 tập OOD.
- [ ] Curriculum cho GRPO: bộ đề sao cho init đạt `solve_rate` ≥ 0.15 để `group_variance`
      dương ở đa số nhóm.
- [ ] 3 seed cho SFT và GRPO binary → có error bar thật.
- [ ] Ablation `beta` (0.04 so với 0.2) và `learning_rate` (1e-6 so với 1e-5).
- [ ] Đồng nghiệp: test đối kháng bổ sung cho verifier + phân tích phân bố lỗi theo độ khó.

## Câu hỏi cho GVHD

1. `ood-arity` đổi đồng thời 5 số và depth 3 (`generator.py:89`). Thầy muốn em tách bằng cách
   thêm profile mới (tăng `generator_version`) hay chấp nhận đây là "OOD theo nghĩa rộng"?
2. Với TRL 0.15.2, KL reference là **base model khi tắt adapter**, không phải snapshot SFT
   (`grpo/train.py:50-52`). Em hiểu đúng không, và điều này có làm thay đổi cách đọc `beta`?
3. Phát hiện greedy (0.175) tốt hơn sampling temp 0.8 (0.100) — nhưng GRPO *cần* exploration.
   Thầy khuyên xử lý mâu thuẫn này thế nào: hạ temperature khi train, hay tăng G?
4. Quy mô 0.5B + 40 đề có đủ để viết thành kết quả nghiên cứu, hay đề tài cần đổi trục câu hỏi
   sang phân tích hành vi (error taxonomy) thay vì so sánh `pass@1`?
