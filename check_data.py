# save as check_data.py
en_file = './dataset/data/corpus.train.en'
bn_file = './dataset/data/corpus.train.bn'

with open(en_file, 'r', encoding='utf-8') as f_en, open(bn_file, 'r', encoding='utf-8') as f_bn:
    for i, (en_line, bn_line) in enumerate(zip(f_en, f_bn)):
        if i >= 5:  # Just check the first 5 lines
            break
        print(f"English {i+1}: {en_line.strip()}")
        print(f"Bengali {i+1}: {bn_line.strip()}")
        print()