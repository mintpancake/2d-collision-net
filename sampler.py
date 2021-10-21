import numpy as np


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


if __name__ == "__main__":
    scn = load('scene.txt')
    obj = load('object.txt')
    sc = sample(scn, 6000)
    oc = sample(obj, 2000)
    save(sc, 'scene_point_cloud.txt')
    save(oc, 'object_point_cloud.txt')
