import csv
from tqdm import tqdm

# convert p2m data list to shapenet dats csv
headers = ['id', 'synsetId', 'subSynsetId', 'modelId', 'split']
rows = []

list_name = ['train_list.txt', 'test_list.txt']
with open(list_name[0], 'r') as f:
    lines = f.readlines()

    print('reading ', list_name[0])
    for line in tqdm(lines):
        line_split = line.split('/')
        if(line_split[-1].strip() == '04.dat'):
            synset_id, model_id  = line_split[2:4]
            rows.append({
                headers[0]:synset_id, 
                headers[1]:synset_id, 
                headers[2]:synset_id, 
                headers[3]:model_id,
                headers[4]:'train'
            })

with open(list_name[1], 'r') as f:
    lines = f.readlines()

    print('reading ', list_name[1])
    for line in tqdm(lines):
        line_split = line.split('/')
        if(line_split[-1].strip() == '04.dat'):
            synset_id, model_id  = line_split[2:4]
            rows.append({
                headers[0]:synset_id, 
                headers[1]:synset_id, 
                headers[2]:synset_id, 
                headers[3]:model_id,
                headers[4]:'test'
            })

with open('p2m.csv','w') as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(rows)      