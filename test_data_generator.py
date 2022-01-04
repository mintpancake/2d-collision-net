import numpy as np
from utils import ensure_dir, read_config


CFG = read_config()
OBJ_NAME = CFG['obj_name']
SCENE_NAME = CFG['scene_name']
DATA_NAME = CFG['data_name']
N_PAIR = CFG['test_data_size']
CUT_SIZE = CFG['cut_size']


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


def save_points(file, data):
    ensure_dir(file)
    f = open(file, 'w')
    for datum in data:
        f.write(f'{datum[0]} {datum[1]}\n')
    f.close()


def save_gt(file, data):
    ensure_dir(file)
    f = open(file, 'w')
    for datum in data:
        f.write(f'{datum}\n')
    f.close()


def inside(p, scn):
    angle_sum = 0
    for i in range(len(scn)):
        a = scn[i]
        b = scn[(i + 1) % len(scn)]
        angle_sum += np.arctan2(np.cross(a - p, b - p), np.dot(a - p, b - p))
    return abs(angle_sum) > 1


def trajectory(x):
    y = -x
    return y


if __name__ == "__main__":
    DATA_NAME = f'{DATA_NAME}_test'
    scene = load(f'raw/{SCENE_NAME}_norm.txt')
    # scene_pc = load(f'data/{DATA_NAME}/scene_pc/{SCENE_NAME}_pc.txt')
    obj = load(f'raw/{OBJ_NAME}_norm.txt')
    obj_pc = load(f'data/{DATA_NAME}/obj_pc/{OBJ_NAME}_pc.txt')
    grid_size = 1 / CUT_SIZE

    n = N_PAIR
    poses = np.linspace(-0.499, 0.499, n, dtype=np.float32)
    for i in range(n):
        print(f'Generating pair {i+1}...')

        pos = np.array([poses[i], trajectory(poses[i])])

        obj_pc_moved = obj_pc + pos
        obj_moved = obj + pos
        cut_size = CUT_SIZE
        cut = np.linspace(-0.5, 0.5, cut_size+1, dtype=np.float32)
        gt = np.zeros([cut_size, cut_size], dtype=int)
        for j in range(cut_size):
            for k in range(cut_size):
                samples = np.array(
                    [[cut[j], cut[k]], [cut[j+1], cut[k]], [cut[j], cut[k+1]], [cut[j+1], cut[k+1]]])
                for point in samples:
                    if inside(point, scene) and inside(point, obj_moved):
                        gt[j, k] = 1
                        break

        save_points(
            f'data/{DATA_NAME}/obj_pc/moved/{OBJ_NAME}_pc_{i}.txt', obj_pc_moved)
        save_points(f'data/{DATA_NAME}/pos/pos_{i}.txt', [pos])
        save_gt(f'data/{DATA_NAME}/gt/gt_{i}.txt', gt.reshape(-1))
