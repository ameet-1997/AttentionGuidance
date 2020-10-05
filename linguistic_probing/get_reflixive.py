def load_reflexive_dataset(args, tokenizer):
    lines = open(args.train_data_file, 'r').readlines()
    dataset, labels = [], []
    for line in lines():
        comp = line.strip().split()
        sentence = comp[:-3]
        if len(tokenizer.encode(' '.join(sentence))) == len(sentence) + 2:
            dataset.append(' '.join(sentence))
            labels.append((comp[-3], (comp[-2], comp[-1])))
    return dataset, labels