import json # import json module

vocab_path = "./ch-6000-wpm-22000-pretty"

vocab_file = '../data/vocab-v3.txt'
f = open(vocab_file,'w', encoding='utf-8')
with open(vocab_path, encoding='utf-8') as json_file:
    json_data = json.load(json_file)
    for item in json_data["model"]["vocab"].keys():
        f.write(item+'\n')

    f.close()