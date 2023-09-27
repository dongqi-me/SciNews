def get_total_num(input_data):
    if not input_data:
        raise ValueError("Input data cannot be empty.")
    return len(input_data)

if __name__ == "__main__":
    import json
    with open("train.json") as f:
        train_data = json.load(f)
    print(get_total_num(train_data)) # 35532

    with open("val.json") as f:
        val_data = json.load(f)
    print(get_total_num(val_data)) # 4440

    with open("test.json") as f:
        test_data = json.load(f)
    print(get_total_num(test_data)) # 4447