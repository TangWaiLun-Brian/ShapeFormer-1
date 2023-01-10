import os

# names1 = os.listdir("/research/d4/spc/wltang1/ShapeFormer/datasets/IMNet2_fixed_data/train/03001627/")
# names2 = os.listdir("/research/d4/spc/wltang1/ShapeFormer/datasets/IMNet2_fixed_data/test/03001627/")
#
# overlap = [name for name in names2 if name in names1]
# print(len(overlap))

split1 = "/data/brian22/ShapeFormer/datasets/IMNet2_packed/shape_name_train.txt"
with open(split1, 'r') as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]
split2 = "/data/brian22/ShapeFormer/datasets/IMNet2_packed/shape_name_test.txt"
with open(split2, 'r') as f:
    lines2 = f.readlines()
lines2 = [line.strip() for line in lines2]

for line in lines2:
    if line != '' and line in lines:
        print('overlap:', line)