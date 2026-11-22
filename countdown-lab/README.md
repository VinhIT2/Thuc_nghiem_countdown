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

## Hạn chế thật của bộ số liệu này

- Model 0.5B, 1 GPU 4 GB, số bước train là smoke/1-epoch, **không** phải run hội tụ.
- SFT ckpt-80 đồng thời đổi hai biến (96 -> 320 đề **và** 24 -> 80 bước). Chưa có control
  tách riêng ảnh hưởng của lượng dữ liệu.
- Tái lập GPU không đảm bảo bit-identical dù cùng seed; chỉ tiêu số có thể lệch nhẹ.
- Validation 40 đề cho CI rộng (±0.11 ở mức 0.175). Kết luận mạnh hơn cần đề nhiều hơn.
- Pilot `data/` (12 đề validation) giữ nguyên trong repo làm bằng chứng quá trình, không
  dùng để so sánh cross-method.
