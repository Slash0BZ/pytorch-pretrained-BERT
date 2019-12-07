def search_by_keywords(lines, keywords):
    for line in lines:
        line = line.lower()
        status = True
        for key in keywords:
            if key not in line:
                status = False
        if status:
            print(line)

lines = [x.strip() for x in open("samples/gigaword/raw_collection_contextsent_tokenized.txt").readlines()]
search_by_keywords(lines, ["i am running for republican leader because we did n't just lose our majority"])
