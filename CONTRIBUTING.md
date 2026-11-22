# Cách làm việc chung (2 người)

Mục tiêu: hai người chạy thực nghiệm **đồng thời** mà xung đột Git trở nên bất khả thi về
mặt cấu trúc, không phải nhờ nhắc nhở.

## Vì sao ít conflict được

Mã của thầy cố tình **từ chối ghi đè** ở mọi chỗ tạo output:

| Chỗ | Hành vi |
|---|---|
| `generator.py:155` | dataset tồn tại -> `FileExistsError`, phải đổi thư mục có version |
| `baseline.py:39,58` | log tồn tại -> chặn; file mở bằng chế độ `"x"` |
| `training_common.py:24` | `output_dir` tồn tại -> chặn, "never overwrite a run" |
| `eval.py:77` | metrics tồn tại -> chặn |

Nên chỉ cần một luật: **mỗi experiment là một bộ file mới, mang tên bạn**. Không ai sửa
file của experiment đã chạy xong.

## Vùng sở hữu

| Đường dẫn | Ai được sửa | Luật |
|---|---|---|
| `countdown-lab/src/` | **không ai** | Mã của thầy. Cần năng lực mới -> module MỚI cạnh bên, không sửa file cũ. Sửa `parser.py`/`verifier.py`/`reward.py` = đổi thang đo của mọi số liệu đã công bố, bắt buộc tăng `VERSION` và báo cho cả nhóm + PR |
| `countdown-lab/tests/` | cả hai, thêm test mới | Không sửa test đã xanh; test mới là bằng chứng mới |
| `data/`, `data-v2/` | **đóng băng, chỉ đọc** | Dataset mới = `data-v3/` + `generator_version` mới. Không bao giờ regenerate đè |
| `configs/` | mỗi người file của mình | Tên: `<ten>-<method>-<dataset>-<variant>.yaml`. Không sửa config của experiment đã chạy; muốn đổi -> copy thành file mới |
| `logs/` | tự động tách theo `experiment_id` | Đặt `output_dir: logs/<ten>-<exp-id>/` để không đụng nhau |
| `notes/` | mỗi người file của mình | Một experiment = một experiment card, **không có file tổng hợp dùng chung** |

## Quy ước đặt tên (bắt buộc)

```
experiment_id / config_id :  <ten>-<phuong-phap>-<du-lieu>-<bien>-<so>
vid du  :  vinh-sft-v2-full-001
          binh-grpo-v2-shaped-003
duong dan :  logs/vinh-sft-v2-full-001/
             configs/binh-grpo_v2_shaped.yaml
             notes/binh-grpo-v2-shaped-003.md
```

`config_id` phải xuất hiện trong `manifest.json` của run — đó là khóa để đối chiếu log
với config nhiều tuần sau.

## Một experiment hợp lệ gồm những gì

1. `configs/<ten>-....yaml` — chỉ **một** biến số khác với experiment nó so sánh.
2. Lệnh chạy nguyên văn, copy được.
3. `logs/<exp-id>/manifest.json` + `runtime.json` (tự sinh).
4. Log thô `logs/<you>-....jsonl` (tự sinh, đã include output gốc của model).
5. `logs/metrics-<you>-....json` từ `src.eval` với `--expected-n` và `--selection-budget`
   khớp đúng kế hoạch.
6. `notes/<exp-id>.md` — kết luận + **cả kết quả âm**.

Thiếu (5) hoặc khai sai ngân sách thì `eval.py` tự chặn; đó là tính năng, không phải lỗi.

## Những so sánh bị coi là không công bằng

- So model đã train với B0 một mẫu, trong khi model đó được chọn bằng 4 mẫu. Muốn so thì
  so **cùng ngân sách**: B3 cùng `selection_budget`, hoặc khai rõ đang so ở ngân sách nào.
- So `pass@1` của hai cấu hình trên hai tập validation khác nhau.
- Kết luận từ difference nhỏ hơn bề rộng CI. Với 40 đề, CI 95% rộng khoảng ±0.11.

## Luồng Git

```bash
git switch main && git pull
git switch -c <ten>/<chu-de>       # vid: binh/ood-analysis
# làm việc, commit nhỏ và thường xuyên
git push -u origin <ten>/<chu-de>  # mở PR, một người kia duyệt rồi mới merge
```

Không push thẳng vào `main`, không rebase nhánh đã push chung, không `--force`.

## Không commit (đã cấu hình sẵn trong `.gitignore`)

- `.venv/` (4.9 GB) — tạo lại bằng lệnh, xem `countdown-lab/README.md`.
- `.hf-cache/` (954 MB) — tự tải lại, `--revision` đã pin nên vẫn đúng bản.
- `*.safetensors`, `logs/*/checkpoint-*/`, `logs/*/final-adapter/` (336 MB) — train lại
  SFT mất ~3 phút. **Nhưng**: `.gitattributes` đặt `-text` cho `*.jsonl/*.json/*.yaml` vì
  `core.autocrlf=true` sẽ đổi nội dung byte khi checkout, làm lệch sha256 so với manifest
  và vô hiệu hoá chuỗi `--exclude` chống data leakage. Đừng bỏ dòng đó.
- `rl-reasoning-student-guide/` — repo riêng của thầy, mỗi người tự clone.

Trước khi push, luôn kiểm tra:

```bash
git add -A && git diff --cached --name-only | wc -l
git diff --cached --name-only | while read f; do [ -f "$f" ] && stat -c%s "$f"; done \
  | awk '{t+=$1} END {printf "%.1f MB\n", t/1048576}'
```

Một lần suýt commit 1.177 MB (trong đó `model.safetensors` 942 MB, vượt giới hạn 100 MB
của GitHub) chỉ vì bước này bị bỏ qua.

## Gợi ý phân công hiện tại

**Người có GPU (vinh):** chuỗi train — SFT dài hơn, GRPO `binary` vs `shaped`, nhiều seed,
test-id + 4 tập OOD.

**Người không cần GPU (đồng nghiệp):** đây là chỗ đang yếu nhất của nghiên cứu, và không
đòi hỏi máy mạnh:

1. **Control experiment trên giấy + bằng lệnh**: tách ảnh hưởng của *lượng dữ liệu* khỏi
   *số bước train* (hiện hai biến đang đổi cùng lúc ở `sft-v2-001`).
2. **Curriculum**: `reward-diagnostics.jsonl` cho thấy 9/10 nhóm train có
   `group_variance_rate = 0` -> GRPO không có gradient. Cần bộ đề sao cho model sinh ra
   đúng *thỉnh thoảng*. Đây là bài toán **data**, không phải bài toán GPU.
3. **Test đối kháng cho verifier**: `tests/test_verifier.py` đã có 25 case; thêm case ở
   biên (số lặp, chia cho biểu thức bằng 0, độ sâu cây, giới hạn literal).
4. **Phân tích `error_counts`** từ các log thô đã có trong repo: mô hình hoá xem lỗi
   `NUMBERS` và `WRONG_TARGET` phân bố theo độ khó đề thế nào.
