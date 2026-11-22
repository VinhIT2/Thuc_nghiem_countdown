# Dataset Card — data-v2

- **Người phụ trách:** Bùi Quang Vinh
- **Trạng thái:** **Frozen** (2026-11-22)
- **Dataset version:** `generator-v1-reference-profiles` (không đổi so với `data/` pilot;
  khác nhau ở `count` và `seed`)

## Mục đích và phạm vi

Đo so sánh prompting / SFT / RLVR trên Countdown với đủ power thống kê cho `pass@1`.
Pilot `data/` chỉ có 12 đề validation → mọi khác biệt cross-method đều là nhiễu
(chứng minh: con số 0.083 không tái lập). `data-v2` nâng validation lên 40 và thêm OOD 60/tập.

## Đặc tả bài toán

- Số toán hạng: 4 (ID, range, fraction, depth) và **5** (ood-arity)
- Khoảng số: 1–10 (ID) · **11–20** (ood-range)
- Target: số nguyên 1–200, trung vị thực tế 34 (min 1, max 200)
- Toán tử: `+ - * /`, ngoặc đơn, unary minus
- Phân số trung gian **được phép** và bắt buộc với `ood-fraction`; không cho số âm ở đích
- Format output: đúng một dòng `FINAL: <biểu thức>` sau tùy ý phần reasoning

## Cách sinh

- Thuật toán: `method=expression` — ngẫu nhiên lắp 4/5 số thành cây biểu thức, lấy giá trị
  làm đích, rồi lọc theo `matches(profile)` (điều kiện depth/phân số)
- Seed: 100 (ID) và 200/201/202/203 (4 tập OOD)
- **Đảm bảo có nghiệm:** mọi `reference_expression` được chính `verifier.verify()` chấm lại
  trong `assert_dataset_valid()` (`generator.py:29-31`); dataset không chứa lời giải sai
- Difficulty: `reference_has_fraction -> hard`, `depth > 2 -> medium`, còn lại `easy`
  (chú ý: đây là độ khó **theo hình thức reference**, không phải độ khó với model — xem mục
  Hạn chế)

## Split

| Split | Số mẫu | hash (16 ký tự đầu) | Chọn tham số? |
|---|---:|---|---|
| `train.jsonl` | 320 | `5051d7ab15f2461e` | Có |
| `validation.jsonl` | 40 | `0366a98522a5b1b7` | Có |
| `test-id.jsonl` | 40 | `cd2a5250aa9acb60` | **Không** |
| `ood-range.jsonl` | 60 | `39325f3a173cd063` | **Không** |
| `ood-arity.jsonl` | 60 | `7b4f1d5441111f76` | **Không** |
| `ood-fraction.jsonl` | 60 | `1101c522313237ca` | **Không** |
| `ood-depth.jsonl` | 60 | `8a72d5cdda7e8f50` | **Không** |

Tổng **640 đề**, `--exclude` khi sinh OOD ghi lại trong `excluded_hashes` của từng manifest.

## Chống trùng và leakage

- Canonical key: `json.dumps([sorted(numbers), target])` (`generator.py:13-14`) — hoán vị
  số được tính là cùng một đề
- **Kiểm tra overlap đã chạy:** 7 tập **đôi một rời nhau**, 0 cặp trùng trong 21 cặp xét
- ID 80/10/10 theo `generator.build`, xáo bằng `Random(seed+1)`

## Kiểm tra chất lượng

- Tỷ lệ có nghiệm: 100 % (mọi reference được verifier chấm lại lúc sinh)
- `pytest -q` = 45 passed, trong đó `test_split_freeze` kiểm đúng cơ chế freeze + exclude
- Mẫu đối chiếu thủ công: `validation-000000` = `[3,3,4,4] -> 7`, ref `((4 + 4) - (3 / 3))` ✓
- Đã biết: **5 đề của `data/` pilot trùng với `data-v2`**, trong đó 1 đề nằm trong
  `data-v2/train` và cũng thuộc `data/test-id` → số liệu pilot và v2 **không độc lập**, chỉ
  dùng pilot làm minh họa quy trình

## OOD — mỗi tập đổi gì

| Tập | Đổi | Không đổi |
|---|---|---|
| `ood-range` | giá trị số 11–20 | arity 4, depth 2, không phân số |
| `ood-arity` | **số lượng 5** | (nhưng depth ref = 3 → **đổi 2 thứ**, xem Hạn chế) |
| `ood-fraction` | bắt buộc phân số trung gian | arity 4, depth 2 |
| `ood-depth` | depth cây 3 | arity 4, không phân số |

## Hạn chế

1. **`ood-arity` không cô lập được arity**: `generator.py:89` quy định profile này có
   `depth == 3`, nên 5 số luôn đi kèm cây sâu hơn. Kết luận "model gãy vì số lượng số" do
   đó chưa tách khỏi "gãy vì độ sâu". Muốn tách: thêm profile 5-số-depth-2, tức sửa
   `PROFILES`/`matches()` và **tăng `generator_version`**.
2. Không gian đề ID hẹp: chỉ `C(13,4) = 715` bộ số khả dĩ, `data-v2` đã dùng 296 bộ (41 %).
   Đây là lý do sinh thêm đề sẽ sớm trùng, và là lý do `data-v3` nên đổi *kiểu* đề chứ không
   chỉ tăng `count`.
3. Độ khó định nghĩa theo *hình thức reference*, không theo độ khó thực tế với model — bằng
   chứng: `ood-range` (mà generator gắn nhãn như ID) lại có `pass@1` cao nhất 0.133.
