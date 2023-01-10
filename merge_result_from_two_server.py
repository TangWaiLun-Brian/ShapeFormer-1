import os
dir1 = "/data/brian22/ShapeFormer/experiments/shapeformer/partnet_scale_chair/results/VisShapeFormer/eval_partnet_chair_mesh/"
dir2 = "/data/brian22/ShapeFormer/experiments/shapeformer/partnet_scale_chair/results/VisShapeFormer/proj52_partnet_chair_mount/"
dir3 = "/data/brian22/ShapeFormer/experiments/shapeformer/partnet_scale_chair/results/VisShapeFormer/merge_partnet_chair"
os.mkdir(dir3)

names1 = os.listdir(dir1)
names2 = os.listdir(dir2)

names2_non_overlap = [name for name in names2 if name not in names1]
total_names = names1 + names2_non_overlap
print('number of shapes:', len(total_names))

for name in total_names:
    print(name)
    if name not in names1:
        base_path = dir2
    else:
        base_path = dir1
    base_path = os.path.join(base_path, name)
    save_root = os.path.join(dir3, name)
    os.mkdir(save_root)
    cmd = f'cp {os.path.join(base_path, "raw.ply")} {save_root}'
    os.system(cmd)
    for i in range(10):
        cmd = f'cp {os.path.join(base_path, f"fake-z{i}.obj")} {save_root}'
        os.system(cmd)
