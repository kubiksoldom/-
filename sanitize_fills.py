# sanitize_fills.py
import csv, sys, os, re

SRC = "fills_all.csv"
DST = "fills_all.cleaned.csv"

def main():
    if not os.path.exists(SRC):
        print(f"[ERR] Not found: {SRC}")
        sys.exit(1)

    with open(SRC, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("[ERR] empty file")
            sys.exit(1)

        cols = len(header)
        print(f"[INFO] header cols = {cols}: {header}")

        fixed = 0
        dropped = 0
        total = 0

        with open(DST, "w", encoding="utf-8", newline="") as g:
            w = csv.writer(g)
            w.writerow(header)

            for row in reader:
                total += 1
                # склеим лишние поля, если их больше, чем в header
                if len(row) > cols:
                    # пробуем слить хвостовые пустые/шумные поля
                    row = row[:cols] + [",".join(row[cols:])]
                    # если всё равно длиннее — урежем
                    row = row[:cols]
                    fixed += 1
                elif len(row) < cols:
                    # дополним пустыми
                    row = row + [""] * (cols - len(row))
                    fixed += 1

                # уберем не-ASCII управляющие символы
                row = [re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", c) for c in row]

                # простая валидация: обязательные поля не пустые
                # если знаешь точную схему — можно усилить
                if row[0] == "" or row[2] == "" or row[3] == "":
                    dropped += 1
                    continue

                w.writerow(row)

    print(f"[OK] wrote {DST}")
    print(f"[STATS] total={total}, fixed={fixed}, dropped={dropped}")

if __name__ == "__main__":
    main()
