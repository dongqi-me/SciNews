import textstat

def calc_readability(input_data):
    #  FKGL GFI SMOG ARI CLI
    if not input_data:
        raise ValueError("Input data cannot be empty.")

    article_fkgl_list = []
    reference_fkgl_list = []

    article_gfi_list = []
    reference_gfi_list = []

    article_smog_list = []
    reference_smog_list = []

    article_ari_list = []
    reference_ari_list = []

    article_cli_list = []
    reference_cli_list = []
    
    for pair in input_data:

        try:
            reference = pair['reference'].strip()
            article = pair['article'].strip()
        except KeyError:
            raise ValueError("Each dictionary in input data should contain 'reference' and 'prediction' keys.")
        
        article_fkgl_list.append(textstat.flesch_kincaid_grade(article))
        article_gfi_list.append(textstat.gunning_fog(article))
        article_smog_list.append(textstat.smog_index(article))
        article_ari_list.append(textstat.automated_readability_index(article))
        article_cli_list.append(textstat.coleman_liau_index(article))

        reference_fkgl_list.append(textstat.flesch_kincaid_grade(reference))
        reference_gfi_list.append(textstat.gunning_fog(reference))
        reference_smog_list.append(textstat.smog_index(reference))
        reference_ari_list.append(textstat.automated_readability_index(reference))
        reference_cli_list.append(textstat.coleman_liau_index(reference))
    
    doc_num = len(input_data)
    return {"article_fkgl":round(sum(article_fkgl_list)/doc_num, 4), "article_gfi":round(sum(article_gfi_list)/doc_num, 4), "article_smog":round(sum(article_smog_list)/doc_num, 4), "article_ari":round(sum(article_ari_list)/doc_num, 4), "article_cli":round(sum(article_cli_list)/doc_num, 4),
            "reference_fkgl":round(sum(reference_fkgl_list)/doc_num, 4), "reference_gfi":round(sum(reference_gfi_list)/doc_num, 4), "reference_smog":round(sum(reference_smog_list)/doc_num, 4), "reference_ari":round(sum(reference_ari_list)/doc_num, 4), "reference_cli":round(sum(reference_cli_list)/doc_num, 4)}


if __name__ == "__main__":
    import json
    
    with open("train.json") as f:
        train_data=json.load(f)

    train_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in train_data]

    #train_length_stat = calc_readability(train_data)
    #print(train_length_stat)
    #     {'article_fkgl': 14.6068, 'article_gfi': 12.9178, 'article_smog': 14.8399, 'article_ari': 17.9963, 'article_cli': 13.9581, 'reference_fkgl': 13.323, 'reference_gfi': 14.109, 'reference_smog': 14.8533, 'reference_ari': 16.3352, 'reference_cli': 13.9404}

    with open("val.json") as f:
        val_data=json.load(f)

    val_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in val_data]

    #val_length_stat = calc_readability(val_data)
    #print(val_length_stat)
    # {'article_fkgl': 14.5401, 'article_gfi': 12.8818, 'article_smog': 14.7519, 'article_ari': 17.8989, 'article_cli': 13.9372, 'reference_fkgl': 13.3059, 'reference_gfi': 14.0982, 'reference_smog': 14.822, 'reference_ari': 16.3122, 'reference_cli': 13.9137}

    with open("test.json") as f:
        test_data=json.load(f)

    test_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in test_data]

    #test_length_stat = calc_readability(test_data)
    #print(test_length_stat)
    
    # {'article_fkgl': 14.285, 'article_gfi': 12.6756, 'article_smog': 14.8245, 'article_ari': 17.4842, 'article_cli': 13.7474, 'reference_fkgl': 13.2739, 'reference_gfi': 14.0676, 'reference_smog': 14.815, 'reference_ari': 16.2608, 'reference_cli': 13.9111}


    with open("statistics_results/readability.json", 'w') as f:
        data = train_data+val_data+test_data
        data_readability_stat = calc_readability(data)
        json.dump(data_readability_stat, f)