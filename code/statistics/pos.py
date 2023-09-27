import nltk
import multiprocessing as mp
import json


def get_pos(sent):
        return [tok[1] for tok in nltk.pos_tag(nltk.tokenize.word_tokenize(sent))]

def calc_pos_dist(input_data):
    if not input_data:
        raise ValueError("Input data cannot be empty.")

    article_dist = {}
    reference_dist = {}

   


    references = [pair['reference'].strip() for pair in input_data]
    articles = [pair['article'].strip() for pair in input_data]

    
    with mp.Pool(4) as pool:
        reference_pos = pool.map(get_pos, references)
        article_pos = pool.map(get_pos, articles)

    for doc_pos in reference_pos:
        for pos in doc_pos:
            if pos not in reference_dist.keys():
                reference_dist[pos] = 0
            else:
                reference_dist[pos] += 1

    for doc_pos in article_pos:
        for pos in doc_pos:
            if pos not in article_dist.keys():
                article_dist[pos] = 0
            else:
                article_dist[pos] += 1
    article_sum = sum(article_dist.values())
    reference_sum = sum(reference_dist.values())
    article_dist = {k:round(v/article_sum, 4) for k,v in article_dist.items()}
    reference_dist = {k:round(v/reference_sum, 4) for k,v in reference_dist.items()}
    return {"article_pos_distribution":article_dist, "reference_pos_distribution":reference_dist}


if __name__ == "__main__":
    with open("train.json") as f:
        train_data=json.load(f)

    train_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in train_data]

    #train_length_stat = calc_pos_dist(train_data)
    #print(train_length_stat["article_pos_distribution"], train_length_stat["reference_pos_distribution"])
    # {'NNP': 0.0951, 'NN': 0.1629, 'VBZ': 0.014, 'VBN': 0.0289, 'WRB': 0.0017, 'DT': 0.0724, 'RB': 0.0224, 'JJ': 0.0918, 'TO': 0.0149, 'IN': 0.1057, 'NNS':0.0617, 'CC': 0.0293, 'CD': 0.0508, '.': 0.0379, 'MD': 0.004, 'VB': 0.0159, ',': 0.0451, 'VBG': 0.0152, 'JJR': 0.0026, 'PRP$': 0.0031, 'WDT': 0.0041,'PRP': 0.0069, 'VBP': 0.0114, 'VBD': 0.023, '(': 0.0314, ')': 0.0314, ':': 0.0076, 'RBR': 0.0009, 'RBS': 0.0003, 'FW': 0.0011, 'JJS': 0.0011, 'EX': 0.0005, 'RP': 0.0004, 'WP': 0.0003, 'POS': 0.0002, "''": 0.0004, '$': 0.002, '#': 0.0003, 'NNPS': 0.0006, 'PDT': 0.0002, 'WP$': 0.0001, 'UH': 0.0001, 'SYM': 0.0001, '``': 0.0001, 'LS': 0.0}
    # {'(': 0.0036, 'NNP': 0.0626, ')': 0.0036, 'NN': 0.1515, 'VBD': 0.022, 'NNS': 0.0786, 'IN': 0.1187, 'DT': 0.0952, 'VBN': 0.0248, ',': 0.0466, '``': 0.0079, "''": 0.0079, 'JJ': 0.0892, 'CC': 0.0291, 'VBZ': 0.0237, 'RB': 0.032, 'JJR': 0.004, '.': 0.0362, 'POS': 0.0041, 'TO': 0.023, 'VB': 0.0287, 'PRP$': 0.0069, 'VBG': 0.0189, 'PRP': 0.0156, 'WDT': 0.0082, 'WRB': 0.0047, 'VBP': 0.0186, 'RP': 0.0018, 'CD': 0.0109, 'MD': 0.01, 'WP': 0.0021, 'FW': 0.0002, 'RBR': 0.0022, 'EX': 0.001, 'PDT': 0.0003, 'NNPS': 0.0007, ':': 0.0026, 'RBS': 0.0006, '$': 0.0001, 'JJS': 0.0014, 'WP$': 0.0001, 'UH': 0.0, 'SYM': 0.0, '#': 0.0, 'LS': 0.0}
    with open("val.json") as f:
        val_data=json.load(f)

    val_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in val_data]

    #val_length_stat = calc_pos_dist(val_data)
    #print(val_length_stat["article_pos_distribution"], val_length_stat["reference_pos_distribution"])
    # {'NNP': 0.0948, 'NN': 0.1628, 'NNS': 0.0623, 'VBP': 0.0115, 'JJ': 0.0915, 'IN': 0.1061, ':': 0.0075, ',': 0.0453, 'WDT': 0.0041, 'MD': 0.0041, 'VB': 0.0161, 'TO': 0.015, 'DT': 0.0723, 'RB': 0.0226, 'VBZ': 0.0141, 'CC': 0.0293, '.': 0.038, 'VBN': 0.0288, 'PRP': 0.007, 'WRB': 0.0017, 'CD': 0.0504, '(': 0.031, 'VBG': 0.0153, ')': 0.031, 'RBR': 0.0009, 'VBD': 0.0229, 'PRP$': 0.0031, '$': 0.0019, 'JJR': 0.0027, 'RP': 0.0004, 'EX': 0.0005, 'UH': 0.0001, 'PDT': 0.0002, 'JJS': 0.0011, 'RBS': 0.0003, 'WP$': 0.0001, 'FW': 0.0011, 'NNPS': 0.0006, 'WP': 0.0003, 'POS': 0.0002, 'SYM': 0.0001, '#': 0.0003, "''": 0.0002, '``': 0.0001, 'LS': 0.0}
    # {'IN': 0.1184, 'PRP$': 0.0069, 'NNS': 0.0788, 'VBD': 0.0221, 'RB': 0.0324, 'VBN': 0.0247, 'WRB': 0.0048, 'DT': 0.0951, 'NN': 0.1508, 'VB': 0.0289, ',': 0.0469, 'JJ': 0.089, 'JJR': 0.0041, '.': 0.0362, 'NNP': 0.0625, 'VBG': 0.0189, 'PRP': 0.0157, 'CC': 0.0293, 'TO': 0.0232, 'VBP': 0.0188, 'RP': 0.0019, 'VBZ': 0.0237, 'POS': 0.0041, 'MD': 0.01, 'RBR': 0.0022, ':': 0.0025, 'WDT': 0.0082, 'CD': 0.0109, '``': 0.0079, "''": 0.0079, 'WP': 0.0021, '(': 0.0033, ')': 0.0033, 'EX': 0.001, 'WP$': 0.0001, 'PDT': 0.0003, 'JJS': 0.0014, 'NNPS': 0.0007, 'RBS': 0.0006, '$': 0.0001, 'FW': 0.0002, 'UH': 0.0, '#': 0.0, 'SYM': 0.0}
    with open("test.json") as f:
        test_data=json.load(f)

    test_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in test_data]

    #test_length_stat =  calc_pos_dist(test_data)
    #print(test_length_stat["article_pos_distribution"], test_length_stat["reference_pos_distribution"])
    # {'NNP': 0.0952, 'NNS': 0.0619, 'VBP': 0.0114, 'VBN': 0.029, 'JJ': 0.0914, 'IN': 0.1061, 'NN': 0.1622, ',': 0.0451, 'RB': 0.0225, 'TO': 0.0149, 'VBG': 0.0152, 'DT': 0.0728, '.': 0.0378, 'CD': 0.0509, 'PRP': 0.0069, 'WDT': 0.0041, 'CC': 0.0294, '(': 0.0314, ')': 0.0314, 'MD': 0.004, 'VB': 0.0158, 'VBD': 0.023, 'VBZ': 0.0142, 'JJR': 0.0027, 'JJS': 0.0011, 'PRP$': 0.0031, ':': 0.0074, 'RBR': 0.0009, 'EX': 0.0005, 'WRB': 0.0017, '$': 0.0019, 'RP': 0.0004, 'FW': 0.0011, 'RBS': 0.0003, 'PDT': 0.0002, 'NNPS': 0.0006, 'WP': 0.0003, 'SYM': 0.0001, 'WP$': 0.0001, 'UH': 0.0001, 'POS': 0.0002, '#': 0.0003, '``': 0.0, "''": 0.0003, 'LS': 0.0}
    # {'JJ': 0.0891, 'NNS': 0.0784, 'VBP': 0.0187, 'VBN': 0.0246, 'NN': 0.1508, 'IN': 0.1188, 'DT': 0.0953, ',': 0.0467, 'CC': 0.0291, 'VBG': 0.019, 'TO': 0.0229, 'NNP': 0.0633, 'EX': 0.0009, 'RB': 0.0322, 'WDT': 0.0081, 'VB': 0.0286, 'WRB': 0.0046, '.': 0.0363, 'VBZ': 0.0237, 'VBD': 0.0218, 'CD': 0.0109, 'MD': 0.01, ':': 0.0025, 'PRP': 0.0157, 'JJR': 0.004, 'JJS': 0.0014, 'WP': 0.0021, '``': 0.0079, 'RP': 0.0019, "''": 0.0079, 'RBR': 0.0021, 'PRP$': 0.0071, 'PDT': 0.0003, 'POS': 0.0042, '(': 0.0036, ')': 0.0036, 'NNPS': 0.0007, 'RBS': 0.0006, '$': 0.0001, 'WP$': 0.0001, 'FW': 0.0002, '#': 0.0, 'SYM': 0.0, 'UH': 0.0, 'LS': 0.0}

    with open("statistics_results/pos.json", 'w') as f:
        data = train_data+val_data+test_data
        data_pos_stat = calc_pos_dist(data)
        json.dump(data_pos_stat, f)