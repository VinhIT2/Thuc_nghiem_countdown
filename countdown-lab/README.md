# Countdown RLVR lab

Thực nghiệm so sánh **prompting / SFT / RLVR(GRPO)** trên trò chơi Countdown, dùng
verifier làm nguồn phần thưởng.

Bài toán: cho 4 số và 1 đích, ví dụ `[2,3,7,10] -> 47`. Lời giải hợp lệ phải dùng
**mỗi số đúng một lần**, chỉ dùng `+ - * /` và ngoặc, ví dụ `(10-3)*7-2`.

## Mã nguồn này ở đâu ra

Repo GitHub `hieutachi/rl-reasoning-student-guide` **chỉ có tài liệu**, không có code.
Toàn bộ mã dưới `src/` lấy từ JSON nhúng trong HTML của trang lab
(`https://countdown-lab-8nchu0xfadn.qoder.website/`, thẻ `<script id="code-bundle">`),
giải nén bằng `unpack.py` đúng theo hướng dẫn mục 3.1 của thầy. Bản JSON gốc giữ ở
`countdown-code-bundle.json` để ai cũng kiểm tra được code không bị sửa sau khi tải.

## Yêu cầu máy

- Python 3.11 trở lên (môi trường mẫu 3.13, lõi chạy bằng thư viện chuẩn).
- pytest cho phần CPU. GPU NVIDIA chỉ cần khi train SFT/GRPO; baseline chạy được trên CPU.
- ~1 GB đĩa cho model `Qwen2.5-0.5B-Instruct` (tự tải, đã pin revision).

## Cài đặt

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1   |  Linux/WSL: source .venv/bin/activate
.venv/Scripts/python.exe -m pip install -r requirements-core.txt
```

**Bẫy torch (đã dính thật):** `requirements.txt` ghi `torch==2.6.0` không kèm index, nên
trên Windows pip lấy bản `+cpu`; `training_common.py:21` sẽ chặn training với lỗi
`Training example requires NVIDIA CUDA`. Bản CUDA phải cài tường minh:

```bash
.venv/Scripts/python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python.exe -m pip install transformers==4.49.0 trl==0.15.2 peft==0.14.0 datasets==3.3.2 accelerate==1.4.0 PyYAML==6.0.2
.venv/Scripts/python.exe -m pip check
```

Bản đã kiểm chứng chạy được: torch 2.6.0+cu124, transformers 4.49.0, trl 0.15.2,
peft 0.14.0, datasets 3.3.2, accelerate 1.4.0 — khớp chính xác `EXPECTED` trong
`training_common.py:11-12`. Xem `logs/environment.lock-cuda.txt`.

Ghi chú Windows: model cache mặc định tải về `C:`. Đặt `HF_HOME` về thư mục khác nếu
`C:` hẹp, ví dụ `export HF_HOME=D:/duong/dan/.hf-cache`.

## Bước 1 — CPU trước, GPU sau

```bash
.venv/Scripts/python.exe -m pytest -q                 # 45 passed
```

Đừng cài stack ML trước khi lệnh này xanh: parser/verifier sai thì reward sai, và khi
đó không biết lỗi nằm ở model hay ở bộ chấm.

## Bước 2 — dữ liệu đã đóng băng (không cần sinh lại)

`data/` và `data-v2/` đã sinh xong và **đóng băng**. Muốn xác minh nó chưa bị sửa:

```bash
sha256sum data-v2/train.jsonl    # phai khop files["train.jsonl"].sha256 trong data-v2/manifest-id.json
```

| Bộ | train | validation | test-id | OOD |
|---|---|---|---|---|
| `data/` (pilot, seed 42) | 96 | 12 | 12 | 4 tập x 30 |
| `data-v2/` (seed 100) | 320 | 40 | 40 | 4 tập x 60 |

Muốn sinh bộ mới thì **dùng thư mục mới** và tăng version trong `generator.py`, không
ghi đè: `build()` cố tình raise `FileExistsError`. Mỗi OOD phải `--exclude` toàn bộ file
ID để chống leakage (`--exclude` cũng phải theo thứ tự tồn tại của file).

## Bước 3 — baseline và đánh giá

```bash
R=7ae557604adf67be50417f59c2c2f167def9a775
D=data-v2/validation.jsonl
python -m src.baseline --dataset $D --out logs/v2-b0.jsonl --revision $R --n 1 --style direct   --config-id v2-base-direct
python -m src.baseline --dataset $D --out logs/v2-b1.jsonl --revision $R --n 1 --style fewshot  --config-id v2-base-fewshot
python -m src.baseline --dataset $D --out logs/v2-pool4.jsonl --revision $R --n 4 --style fewshot --config-id v2-base-pool4
python -m src.eval --log logs/v2-b0.jsonl --dataset $D --expected-n 1 --out logs/metrics-v2-b0.json
```

Lệnh `--n 4` tự sinh thêm `*.B2.selected.jsonl` (majority vote) và `*.B3.selected.jsonl`
(best-of-N dùng verifier). Log selected phải eval với `--expected-n 1 --selection-budget 4`,
nếu không `eval.py:46-47` sẽ chặn vì khai báo ngân sách không khớp.

`--revision` bắt buộc đúng 40 ký tự hex (`baseline.py:35`). Không dùng `main`.

## Bước 4 — train

```bash
python -m src.sft.train  --config configs/sft_lora_v2.yaml     # ~3 phut, peak 1.2 GB VRAM
python -m src.grpo.train --config configs/grpo_v2_binary.yaml  # ~9 phut
python -m src.grpo.train --config configs/grpo_v2_shaped.yaml  # ~9 phut, chi khac reward_mode
```

Adapter **không có trong Git** (xem `.gitignore`), nên muốn evaluate phải train trước.
Mỗi run tự tạo `logs/<experiment_id>/manifest.json` chứa hash dữ liệu, version package,
tên GPU; `runtime.json` chứa thời gian và đỉnh VRAM.

## Cách đọc chỉ số chi phí (đọc trước khi trích số `mean_seconds`)

`src.eval` báo `mean_seconds_per_record` / `mean_seconds_per_problem` bằng **trung bình
cộng**, nên một bản ghi lâu bất thường sẽ kéo cả cột đi. Trên máy laptop có sleep, điều này
đã xảy ra **5 lần / 59 log**: `time.perf_counter()` vẫn đếm trong lúc máy suspend.

| Log | mean (báo cáo bởi `src.eval`) | median | trimmed mean |
|---|---:|---:|---:|
| `v2-b1.jsonl` | 3.17 s/mẫu | 1.23 | 1.31 |
| `v2-pool4.jsonl` | **217.22** s/mẫu | 1.25 | 1.50 |
| `v2-pool4.B3.selected` | **868.88** s/đề | 5.03 | 7.05 |
| `v2-E-ood-range-grpo240` | 23.14 s/mẫu | 1.88 | 1.47 |

Quy tắc dùng trong project này:

- **Chỉ số chi phí được trích là `median` hoặc trimmed mean** (loại bản ghi > 20 × median),
  không phải mean. Bảng B0-B3 trong README và trong `notes/` đã đổi sang median.
- **Không sửa log.** `eval.py:36` chỉ chấm lại `verifier` chứ **không kiểm tra `seconds`**,
  nên việc xóa bản ghi lạ sẽ là can thiệp âm thầm vào bằng chứng gốc — đúng thứ bộ audit
  này sinh ra để chống. Muốn số sạch thì **chạy lại ra đường dẫn mới** (`baseline.py:39`
  chặn ghi đè, nên chạy lại tự nhiên thành một lần đo độc lập; ví dụ đã làm:
  `v2-pool4-s1234.jsonl` đo lại đúng 3.56 s/đề).
- `valid_output_rate`, `pass@1`, `pass@k`, `error_counts` **không bị ảnh hưởng** vì verifier
  chấm lại từng bản ghi.
- Khi chạy bộ dài, tắt sleep: `powercfg /change standby-timeout-ac 0` (hoàn tác bằng cùng
  lệnh với giá trị `15`).

## Kết quả đã đo trên `data-v2/validation.jsonl` (40 đề)

| Cấu hình | valid | parse fail | pass@1 | pass@4 | CI 95% |
|---|---|---|---|---|---|
| B0 direct, 1 mẫu | 0.000 | 1.000 | 0.000 | — | [0, 0] |
| B1 few-shot, 1 mẫu | 0.425 | 0.100 | 0.025 | — | [0, 0.075] |
| B2 vote, N=4 | 0.900 | 0.025 | — | — | [0, 0] |
| B3 verifier, N=4 (2 seed) | ~0.90 | ~0.03 | — | **0.000** | [0, 0] |
| SFT 1 epoch, ckpt-80, 1 mẫu | **0.975** | **0.000** | **0.175** | — | [0.075, 0.300] |

Bốn điều rút ra:

1. **Format là điều kiện cần của RLVR.** Với prompt direct, 40/40 output không có dòng
   `FINAL:` hợp lệ, nên tập output hợp lệ rỗng và B2/B3 buộc phải chọn mẫu vô hiệu —
   verifier không chấm được cái nó không parse được.
2. **`B3 == pass@4` của cùng pool** là self-check của verifier: best-of-N dùng verifier
   phải đạt đúng trần `pass@k`. Hai con số lệch nhau nghĩa là parser hoặc selection có bug.
3. **Majority vote không phải oracle.** Trên pool của model đã SFT, B2 = 0.000 trong khi
   B3 = 0.083 (pilot v1): model fine-tune theo một khuôn tham chiếu thì phương án sai
   lặp lại nhiều hơn và thắng phiếu. Chỉ reward kiểm chứng còn giá trị.
4. **Số nhỏ trên 12 đề là nhiễu.** Con số 0.083 của pilot `data/` không tái lập trên 40 đề
   (0.000 ở cả hai seed). Đừng kết luận phương pháp từ validation 12 đề.

## Kết quả trên test-ID và 4 trục OOD (SFT ckpt-80, greedy 1 mẫu)

| Tập | n | `valid` | `pass@1` | Fisher p so với test-ID (về `valid`) |
|---|---:|---:|---:|---:|
| `test-id` | 40 | 1.000 | 0.050 | — |
| `ood-range` (số 11–20) | 60 | 1.000 | 0.133 | 1.00 (không khác) |
| **`ood-arity` (5 số)** | 60 | **0.000** | **0.000** | **7.3e-29 (gãy hoàn toàn)** |
| `ood-fraction` | 60 | 0.967 | 0.033 | 0.52 (không khác) |
| `ood-depth` | 60 | 0.967 | 0.050 | 0.52 (không khác) |

Khoảng cách ngoài phân phối **chỉ tồn tại trên một trục: số lượng số**. Trên 60/60 đề 5 số,
model luôn viết đúng 4 hằng số theo khuôn `((a op b) op (c op d))` và bỏ rơi số còn lại
(`error_code = NUMBERS` 60/60) — tức nó học *khuôn hình thức* của reference, không học
*luật* của trò chơi. Chi tiết và hệ quả ở `notes/vinh-eval-ood-v2-001.md`.

Lưu ý: `ood-arity` đổi đồng thời 2 thuộc tính (5 số **và** depth 3 theo `generator.py:89`),
nên "arity" chưa tách khỏi "độ sâu" — muốn tách phải thêm profile mới và tăng
`generator_version`.

## Hạn chế thật của bộ số liệu này

- Model 0.5B, 1 GPU 4 GB, số bước train là smoke/1-epoch, **không** phải run hội tụ.
- SFT ckpt-80 từng đổi hai biến một lúc (96 → 320 đề **và** 24 → 80 bước); control
  `vinh-sft-control96-001` đã tách: định dạng đến từ **số bước**, độ chính xác theo **lượng
  dữ liệu** nhưng ở mức chưa có ý nghĩa thống kê (Fisher p ≈ 0.36).
- Mỗi cấu hình train **một seed duy nhất**; chưa có phương sai giữa seed cho SFT/GRPO.
- Tái lập GPU không đảm bảo bit-identical dù cùng seed; chỉ tiêu số có thể lệch nhẹ.
- Validation 40 đề cho CI rộng (±0.11 ở mức 0.175). Kết luận mạnh hơn cần đề nhiều hơn.
- Pilot `data/` (12 đề validation) giữ nguyên trong repo làm bằng chứng quá trình, không
  dùng để so sánh cross-method.
