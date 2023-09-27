def get_topic_dist(input_data):
    if not input_data:
        raise ValueError("Input data cannot be empty.")
    topic_list = [pair['Topic'].strip() for pair in input_data]

    topic_dist = {}
    for topic in topic_list:
        if topic not in topic_dist.keys():
            topic_dist[topic] = 0
        topic_dist[topic] += 1
    topic_dist = {k:round(v/len(topic_list), 4) for k, v in topic_dist.items()}
    return {"topic_dist": topic_dist, "topic_list": topic_list}

if __name__ == "__main__":
    import json
    with open("train.json") as f:
        train_data = json.load(f)
    print(get_topic_dist(train_data)["topic_dist"]) # {'Space': 0.0259, 'Medicine': 0.3512, 'Nano': 0.0595, 'Other': 0.0465, 'Chemistry': 0.0763, 'Biology': 0.2108, 'Earth': 0.093, 'Computer': 0.026, 'Physics': 0.1108}

    with open("val.json") as f:
        val_data = json.load(f)
    print(get_topic_dist(val_data)["topic_dist"]) # {'Chemistry': 0.0764, 'Earth': 0.093, 'Medicine': 0.3514, 'Biology': 0.2108, 'Nano': 0.0595, 'Physics': 0.1108, 'Computer': 0.0259, 'Other': 0.0464, 'Space': 0.0259}

    with open("test.json") as f:
        test_data = json.load(f)
    print(get_topic_dist(test_data)["topic_dist"]) # {'Earth': 0.0931, 'Medicine': 0.3508, 'Physics': 0.1109, 'Other': 0.0465, 'Space': 0.0261, 'Chemistry': 0.0762, 'Biology': 0.2107, 'Nano': 0.0596, 'Computer': 0.0261}

    with open("statistics_results/topic_distribution.json", 'w') as f:
        data = train_data+val_data+test_data
        data_topic_stat = get_topic_dist(data)
        json.dump(data_topic_stat, f)