import numpy as np


OBJ_NAME = 'obj'
SCENE_NAME = 'scene'
DATA_NAME = 'data1'

def load(file):
    v = []
    f = open(file, 'r')
    line = f.readline()
    while line:
        x, y = map(lambda n: np.double(n), line.strip('\n').split(' '))
        v.append([x, y])
        line = f.readline()
    f.close()
    return np.array(v)


def save(data, file):
    f = open(file, 'w')
    for datum in data:
        f.write(f'{datum[0]} {datum[1]}\n')
    f.close()


def sample(v, n):
    total_length = 0
    edge_length = np.zeros(len(v), dtype=np.double)
    for i in range(len(v)):
        length = np.linalg.norm(v[(i + 1) % len(v)] - v[i])
        edge_length[i] = length
        total_length += length
    edge_portion = edge_length / total_length
    edge_portion *= n
    edge_num = np.around(edge_portion).astype(int)

    direction = (v[1] - v[0])
    d = np.random.uniform(0, 1, size=(edge_num[0], 1))
    boundary_points = v[0] + d * direction
    for i in range(1, len(v)):
        direction = (v[(i + 1) % len(v)] - v[i])
        d = np.random.uniform(0, 1, size=(edge_num[i], 1))
        boundary_points = np.concatenate(
            (boundary_points, v[i] + d * direction), axis=0)
    return boundary_points


def normalize(v):
    g = np.mean(v, axis=0)
    return v - g


if __name__ == "__main__":
    scene = load(f'raw/{SCENE_NAME}.txt')
    obj = load(f'raw/{OBJ_NAME}.txt')
    obj_normalized = normalize(obj)
    scene_centered = scene - np.array([0.5, 0.5])
    save(obj_normalized, f'raw/{OBJ_NAME}_norm.txt')
    save(scene_centered, f'raw/{SCENE_NAME}_norm.txt')
    scene_pc = sample(scene_centered, 4000)
    obj_pc = sample(obj_normalized, 4000)
    save(scene_pc, f'data/{DATA_NAME}/scene_pc/{SCENE_NAME}_pc.txt')
    save(obj_pc, f'data/{DATA_NAME}/obj_pc/{OBJ_NAME}_pc.txt')
    save(scene_pc, f'data/{DATA_NAME}_test/scene_pc/{SCENE_NAME}_pc.txt')
    save(obj_pc, f'data/{DATA_NAME}_test/obj_pc/{OBJ_NAME}_pc.txt')
