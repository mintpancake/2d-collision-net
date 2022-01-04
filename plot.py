import matplotlib.pyplot as plt
from utils import ensure_dir, read_config

CFG = read_config()
DATA = CFG['data_name']


def read(file):
    epoch = []
    val = []
    f = open(file, 'r')
    line = f.readline()
    line = f.readline()
    while line:
        x, y = line.strip('\n').split(',')
        x, y = int(x), float(y)
        epoch.append(x)
        val.append(y)
        line = f.readline()
    f.close()
    return epoch, val


if __name__ == "__main__":
    print('Ploting graphs...')
    x, y = read(f'logs/{DATA}/train_loss.csv')
    plt.plot(x, y, color='orange', linestyle='solid',
             marker='None', label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('BCEWithLogitsLoss')
    plt.grid()
    ensure_dir(f'graphs/{DATA}/train_loss.png')
    plt.savefig(f'graphs/{DATA}/train_loss.png', bbox_inches='tight')
    plt.close('all')

    x, y = read(f'logs/{DATA}/train_error.csv')
    plt.plot(x, y, color='orange', linestyle='solid',
             marker='None', label="Training Error")
    plt.xlabel('Epoch')
    plt.ylabel('Average missclassified grids')
    plt.grid()
    ensure_dir(f'graphs/{DATA}/train_error.png')
    plt.savefig(f'graphs/{DATA}/train_error.png', bbox_inches='tight')
    plt.close('all')

    x, y = read(f'logs/{DATA}/val_loss.csv')
    plt.plot(x, y, color='orange', linestyle='solid',
             marker='None', label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('BCEWithLogitsLoss')
    plt.grid()
    ensure_dir(f'graphs/{DATA}/val_loss.png')
    plt.savefig(f'graphs/{DATA}/val_loss.png', bbox_inches='tight')
    plt.close('all')

    x, y = read(f'logs/{DATA}/val_error.csv')
    plt.plot(x, y, color='orange', linestyle='solid',
             marker='None', label="Validation Error")
    plt.xlabel('Epoch')
    plt.ylabel('Average missclassified grids')
    plt.grid()
    ensure_dir(f'graphs/{DATA}/val_error.png')
    plt.savefig(f'graphs/{DATA}/val_error.png', bbox_inches='tight')
    plt.close('all')

    print('Done!')
