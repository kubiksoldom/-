import csv
inp='fills_all.csv'; out='fills_all_clean.csv'
with open(inp, newline='', encoding='utf-8') as f, open(out,'w',newline='',encoding='utf-8') as g:
    r=csv.reader(f)
    header=next(r); exp=len(header)
    w=csv.writer(g)
    w.writerow(header)
    fixed=0
    for row in r:
        if len(row)!=exp:
            # если колонок больше — склеиваем «лишнее» в ПОСЛЕДНЮЮ колонку (обычно это orderLinkId)
            if len(row)>exp:
                row=row[:exp-1]+[",".join(row[exp-1:])]
            else:
                # если меньше — дополняем пустыми
                row+=['']*(exp-len(row))
            fixed+=1
        w.writerow(row)
print("fixed", fixed)
