def countSentencePairs():
    # Paths to the parallel corpus files
    bn_file = "dataset/data/corpus.train.bn"
    en_file = "dataset/data/corpus.train.en"

    # Count lines in the Bengali file
    with open(bn_file, 'r', encoding='utf-8') as f:
        bn_line_count = sum(1 for _ in f)

    # Count lines in the English file
    with open(en_file, 'r', encoding='utf-8') as f:
        en_line_count = sum(1 for _ in f)

    print(f"Bengali file line count: {bn_line_count}")
    print(f"English file line count: {en_line_count}")

    if bn_line_count == en_line_count:
        print(f"Total number of sentence pairs: {bn_line_count}")
    else:
        print("Warning: Files have different line counts!")

def countVocabSize():
    import sentencepiece as spm

    # Load the Bengali SentencePiece model
    bn_sp = spm.SentencePieceProcessor()
    bn_sp.load('dataset/vocab/bn.model')  # replace with your actual path

    # Load the English SentencePiece model
    en_sp = spm.SentencePieceProcessor()
    en_sp.load('dataset/vocab/en.model')  # replace with your actual path

    # Get vocabulary sizes
    bn_vocab_size = bn_sp.get_piece_size()
    en_vocab_size = en_sp.get_piece_size()

    print(f"Bengali vocabulary size: {bn_vocab_size}")
    print(f"English vocabulary size: {en_vocab_size}")


countVocabSize()
countSentencePairs()