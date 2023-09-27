import spacy
import json

import numpy as np

import multiprocessing as mp



nlp = spacy.load("en_core_web_sm")


def get_dependency_labels(input_data):
    # avg sentence number & word number
    if not input_data:
        raise ValueError("Input data cannot be empty.")



    references = [pair['reference'].strip() for pair in input_data]
    articles = [pair['article'].strip() for pair in input_data]
    
    article_docs = nlp.pipe(articles, n_process=4, disable=['tagger', 'ner', 'lemmatizer', 'textcat', 'custom'])
    reference_docs = nlp.pipe(references, n_process=4, disable=['tagger', 'ner', 'lemmatizer', 'textcat', 'custom'])
    
    article_label_list = []
    reference_label_list = []

    reference_dict  = {}
    article_dict = {}

    all_keys = []
    for doc in article_docs:
        label_dict = {}
        for token in doc:
            if token.dep_ not in label_dict.keys():
                label_dict[token.dep_] = 0               
            else:
                label_dict[token.dep_] += 1
            if token.dep_ not in article_dict.keys():
                article_dict[token.dep_] = 0               
            else:
                article_dict[token.dep_] += 1
            if token.dep_ not in all_keys:
                all_keys.append(token.dep_)
        article_label_list.append(label_dict)
    
    for doc in reference_docs:
        label_dict = {}
        for token in doc:
            if token.dep_ not in label_dict.keys():
                label_dict[token.dep_] = 0               
            else:
                label_dict[token.dep_] += 1
            if token.dep_ not in reference_dict.keys():
                reference_dict[token.dep_] = 0               
            else:
                reference_dict[token.dep_] += 1
            if token.dep_ not in all_keys:
                all_keys.append(token.dep_)
        reference_label_list.append(label_dict)

    for label_dict in article_label_list:
        for label in all_keys:
            if label not in label_dict.keys():
                label_dict[label] = 0
    
    for label_dict in reference_label_list:
        for label in all_keys:
            if label not in label_dict.keys():
                label_dict[label] = 0
    
    for label in all_keys:
        if label not in article_dict.keys():
            article_dict[label] = 0
        if label not in reference_dict.keys():
            reference_dict[label] = 0

    return {"article_label_list": article_label_list, 'reference_label_list': reference_label_list, "reference_dict": reference_dict, "article_dict": article_dict}

if __name__ == "__main__":
    with open("train.json") as f:
        train_data=json.load(f)

    train_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in train_data]

    #train_length_stat = get_dependency_labels(train_data)
    #print(train_length_stat["article_dict"], train_length_stat["reference_dict"])
    # {'compound': 23880801, 'nsubjpass': 4291822, 'case': 288771, 'advmod': 7124419, 'auxpass': 4407306, 'ROOT': 9591120, 'det': 19301280, 'amod': 22753747, 'advcl': 2906982, 'prep': 28414458, 'pobj': 28596338, 'cc': 8249064, 'conj': 10369011, 'nummod': 9521064, 'punct': 45414469, 'nsubj': 8947595, 'aux': 3680715, 'npadvmod': 3493313, 'poss': 1126613, 'acomp': 1427192, 'mark': 2081277, 'agent': 921208, 'nmod': 4763338, 'ccomp': 1688084, 'pcomp': 1075473, 'dobj': 9036075, 'appos': 9032411, 'quantmod': 448941, 'relcl': 1538932, 'acl': 2531915, 'parataxis': 258501, 'neg': 527695, 'attr': 1031752, 'xcomp': 1230713, 'dep': 753859, 'prt': 177141, 'expl': 125488, 'oprd': 154426, 'csubj': 120216, 'meta': 278604, 'predet': 67292, 'preconj': 191870, 'csubjpass': 22514, 'dative': 29246, 'intj': 49429} {'punct': 3012651, 'nsubj': 1567324, 'relcl': 302467, 'dobj': 1124253, 'prep': 2730771, 'det': 2260210, 'compound': 1589992, 'pobj': 2630904, 'acl': 241136, 'agent': 71499, 'ROOT': 897302, 'amod': 2066304, 'cc': 733030, 'conj': 806596, 'ccomp': 423270, 'advmod': 972254, 'acomp': 203159, 'mark': 286547, 'advcl': 319872, 'nmod': 115390, 'poss': 267576, 'case': 96606, 'appos': 235410, 'attr': 145830, 'aux': 795830, 'xcomp': 220740, 'prt': 55431, 'nummod': 212044, 'npadvmod': 122704, 'quantmod': 50251, 'nsubjpass': 230395, 'auxpass': 255075, 'neg': 71640, 'csubj': 29194, 'pcomp': 165372, 'predet': 10210, 'preconj': 15082, 'expl': 23667, 'parataxis': 7352, 'oprd': 30107, 'dep': 14573, 'csubjpass': 1304, 'dative': 7917, 'intj': 1197, 'meta': 4272}
    #with open("train_dependency_labels.json", 'w') as f:
    #    json.dump(train_length_stat, f)

    with open("val.json") as f:
        val_data=json.load(f)

    val_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in val_data]

    #val_length_stat = get_dependency_labels(val_data)
    #print(val_length_stat["article_dict"], val_length_stat["reference_dict"])
    # {'compound': 2967575, 'nummod': 1174196, 'nsubj': 1126726, 'punct': 5648221, 'ROOT': 1199968, 'acomp': 179750, 'prep': 3556119, 'amod': 2845654, 'pobj': 3576286, 'appos': 1123495, 'aux': 464810, 'relcl': 194199, 'det': 2402000, 'conj': 1296624, 'cc': 1032450, 'advmod': 894688, 'dobj': 1132303, 'mark': 261868, 'npadvmod': 433569, 'ccomp': 211445, 'quantmod': 54953, 'advcl': 362818, 'auxpass': 546758, 'nsubjpass': 531665, 'acl': 318252, 'poss': 143140, 'nmod': 585249, 'pcomp': 135397, 'xcomp': 153884, 'meta': 34533, 'dep': 92892, 'intj': 5937, 'parataxis': 32012, 'agent': 115490, 'csubj': 15329, 'attr': 128785, 'neg': 66492, 'oprd': 19742, 'preconj': 23942, 'prt': 22017, 'expl': 15904, 'predet': 8348, 'csubjpass': 2743, 'case': 36497, 'dative': 3521} {'mark': 36102, 'poss': 33085, 'nsubjpass': 28342, 'auxpass': 31429, 'advmod': 122118, 'advcl': 40575, 'det': 280136, 'nsubj': 196111, 'aux': 99613, 'neg': 9006, 'punct': 373149, 'ROOT': 111506, 'dobj': 140644, 'amod': 255770, 'prep': 338011, 'pobj': 325532, 'appos': 28625, 'compound': 196454, 'npadvmod': 15271, 'ccomp': 52849, 'cc': 91737, 'conj': 100875, 'prt': 7150, 'acl': 29738, 'acomp': 25746, 'case': 11883, 'quantmod': 6431, 'nummod': 26076, 'pcomp': 20570, 'dep': 1768, 'relcl': 37624, 'xcomp': 27598, 'attr': 17976, 'dative': 933, 'agent': 8746, 'expl': 2946, 'nmod': 13935, 'csubj': 3652, 'predet': 1284, 'parataxis': 861, 'csubjpass': 144, 'oprd': 3698, 'preconj': 1895, 'meta': 513, 'intj': 147}
    #with open("val_dependency_labels.json", 'w') as f:
    #    json.dump(val_length_stat, f)

    with open("test.json") as f:
        test_data=json.load(f)

    test_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in test_data]

    #test_length_stat = get_dependency_labels(test_data)
    #print(test_length_stat["article_dict"], test_length_stat["reference_dict"])
    # {'amod': 2835380, 'nsubj': 1115485, 'aux': 458463, 'ROOT': 1191585, 'dobj': 1117316, 'prep': 3546106, 'npadvmod': 434203, 'punct': 5629776, 'compound': 2967423, 'pobj': 3569127, 'advmod': 886285, 'pcomp': 134119, 'cc': 1029225, 'det': 2415685, 'poss': 139032, 'case': 34881, 'advcl': 361244, 'nummod': 1190008, 'nmod': 590145, 'appos': 1119464, 'conj': 1291694, 'relcl': 191891, 'mark': 259758, 'acl': 316094, 'ccomp': 211096, 'attr': 130034, 'neg': 65783, 'auxpass': 550805, 'predet': 8561, 'nsubjpass': 536570, 'xcomp': 152609, 'acomp': 178131, 'expl': 15820, 'quantmod': 54977, 'csubj': 14784, 'dep': 94143, 'parataxis': 32178, 'meta': 33914, 'prt': 22166, 'agent': 116194, 'dative': 3635, 'intj': 5949, 'preconj': 23879, 'oprd': 18803, 'csubjpass': 2705} {'amod': 258626, 'nsubj': 196847, 'aux': 99879, 'ROOT': 112698, 'compound': 199072, 'dobj': 140084, 'prep': 342334, 'det': 283278, 'pobj': 330332, 'punct': 377900, 'cc': 91886, 'appos': 29353, 'conj': 101263, 'expl': 2881, 'ccomp': 53297, 'attr': 18673, 'acl': 30109, 'advmod': 122234, 'auxpass': 31802, 'xcomp': 27417, 'advcl': 40087, 'mark': 35846, 'csubj': 3610, 'relcl': 37808, 'nummod': 26394, 'preconj': 1917, 'nsubjpass': 28781, 'neg': 9008, 'pcomp': 20376, 'acomp': 25433, 'prt': 7285, 'nmod': 14410, 'quantmod': 6390, 'agent': 9052, 'poss': 34006, 'predet': 1311, 'oprd': 3740, 'case': 12174, 'intj': 133, 'npadvmod': 15013, 'dep': 1809, 'dative': 981, 'parataxis': 920, 'meta': 484, 'csubjpass': 131}
    #with open("test_dependency_labels.json", 'w') as f:
    #    json.dump(test_length_stat, f)

    with open("statistics_results/dependency_label.json", 'w') as f:
        data = train_data+val_data+test_data
        data_dependency_stat = get_dependency_labels(data)
        json.dump(data_dependency_stat, f)