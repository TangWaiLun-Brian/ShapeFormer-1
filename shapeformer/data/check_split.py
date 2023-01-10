import json
split = "/data/brian22/ShapeFormer/datasets/PartNet_Split/partnet_03001627_train.json"
with open(split, 'r') as f:
    info = json.load(f)
names = [itm['model_id'] for itm in info]
target = '58a1e9909542abbae48dacf1789d97b'
print(target in names)