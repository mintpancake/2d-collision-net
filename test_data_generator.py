import numpy as np
from shapely.geometry import Polygon
from utils import ensure_dir, read_config


CFG = read_config()
OBJ_NAME = CFG['obj_name']
SCENE_NAME = CFG['scene_name']
DATA_NAME = f"{CFG['data_name']}_test"
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


def trajectory(x):
    y = x
    # y = -x
    # y = 4*pow(x, 2)-0.5
    return y


def run():
    scene = load(f'raw/{SCENE_NAME}_norm.txt')
    # scene_pc = load(f'data/{DATA_NAME}/scene_pc/{SCENE_NAME}_pc.txt')
    obj = load(f'raw/{OBJ_NAME}_norm.txt')
    obj_pc = load(f'data/{DATA_NAME}/obj_pc/{OBJ_NAME}_pc.txt')
    scene_poly = Polygon(scene)
    n = N_PAIR
    cut_size = CUT_SIZE
    poses = np.linspace(-0.499, 0.499, n, dtype=np.float32)
    print('Generating', end=' ')
    for i in range(n):
        print(i, end=' ')
        gt = np.zeros([cut_size, cut_size], dtype=int)
        pos = np.array([poses[i], trajectory(poses[i])])
        obj_pc_moved = obj_pc + pos
        obj_moved = obj + pos
        obj_moved_poly = Polygon(obj_moved)
        inter_poly = scene_poly.intersection(obj_moved_poly)

        if not inter_poly.is_empty:
            cut = np.linspace(-0.5, 0.5, cut_size+1, dtype=np.float32)
            for j in range(cut_size):
                for k in range(cut_size):
                    grid = np.array(
                        [[cut[j], cut[k]], [cut[j+1], cut[k]], [cut[j+1], cut[k+1]], [cut[j], cut[k+1]]])
                    grid_poly = Polygon(grid)
                    if inter_poly.intersects(grid_poly):
                        gt[j, k] = 1

        save_points(
            f'data/{DATA_NAME}/obj_pc/moved/{OBJ_NAME}_pc_{i}.txt', obj_pc_moved)
        save_points(f'data/{DATA_NAME}/pos/pos_{i}.txt', [pos])
        save_gt(f'data/{DATA_NAME}/gt/gt_{i}.txt', gt.reshape(-1))


if __name__ == "__main__":
    run()
