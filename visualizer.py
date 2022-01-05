import cv2
import numpy as np
import glob
from PIL import Image
from utils import ensure_dir, read_config

CFG = read_config()
SIDE_LENGTH = 800
CANVAS_SIZE = (801, 801, 3)
CUT_SIZE = CFG['cut_size']
N = CFG['test_data_size']

DATA_NAME = f"{CFG['data_name']}_test"
OBJ_NAME = CFG['obj_name']
SCENE_NAME = CFG['scene_name']

MODE = 'pred'

PRED = f'data/{DATA_NAME}/{MODE}'
POS = f'data/{DATA_NAME}/pos'
OBJ = f'raw/{OBJ_NAME}_norm.txt'
SCENE = f'raw/{SCENE_NAME}_norm.txt'


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


def load_pred(file):
    pred = []
    f = open(file, 'r')
    line = f.readline()
    while line:
        pred.append(int(line.strip('\n')))
        line = f.readline()
    f.close()
    return np.array(pred)


def scale(v):
    v[:, 1] = -v[:, 1]
    v += np.array([0.5, 0.5])
    v *= SIDE_LENGTH
    return np.around(v).astype(int)


def draw(idx):
    canvas = np.zeros(CANVAS_SIZE, np.uint8)

    pred = load_pred(f'{PRED}/{MODE}_{idx}.txt')
    pred = pred.reshape([CUT_SIZE, CUT_SIZE])
    grid_size = int(SIDE_LENGTH / CUT_SIZE)
    grid_num = CUT_SIZE
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if pred[i, j] == 1:
                upper_left = np.array(
                    [i * grid_size, (grid_num-j-1) * grid_size])
                upper_right = np.array(
                    [(i+1) * grid_size, (grid_num-j-1) * grid_size])
                lower_left = np.array(
                    [i * grid_size, (grid_num-j) * grid_size])
                lower_right = np.array(
                    [(i+1) * grid_size, (grid_num-j) * grid_size])
                cv2.fillPoly(canvas, np.array(
                    [[upper_left, upper_right, lower_right, lower_left]]), (255, 255, 255))

    cut = np.linspace(0, 800, grid_num + 1, dtype=int)
    for c in cut:
        cv2.line(canvas, [c, 0], [c, SIDE_LENGTH], (127, 127, 127))
        cv2.line(canvas, [0, c], [SIDE_LENGTH, c], (127, 127, 127))

    scene = scale(load(SCENE))
    obj = scale(load(OBJ)+load(f'{POS}/pos_{idx}.txt'))
    cv2.polylines(canvas, [scene], True, (255, 255, 0), 2)
    cv2.polylines(canvas, [obj], True, (0, 0, 255), 2)
    ensure_dir(f'images/{DATA_NAME}/map_{str(idx).zfill(3)}.png')
    cv2.imwrite(f'images/{DATA_NAME}/map_{str(idx).zfill(3)}.png', canvas)


if __name__ == '__main__':
    error = 0
    for i in range(N):
        draw(i)
    fp_in = f'./images/{DATA_NAME}/map_*.png'
    fp_out = f'./images/{DATA_NAME}/animation.gif'

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)
