def search_by_keywords(lines, keywords):
    for line in lines:
        line = line.lower()
        status = True
        for key in keywords:
            if key not in line:
                status = False
        if status:
            print(line)

lines = [x.strip() for x in open("samples/gigaword/tmp_seq_fixed/train.formatted.txt").readlines()]
search_by_keywords(lines, ["[unused500]", "[unused57]"])
