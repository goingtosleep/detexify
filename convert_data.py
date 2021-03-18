import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

## Load data
df = pd.read_csv('data.csv')
df = df.drop(['Unnamed: 0'], axis=1)

df.key
y = df.key.values
np.save('y.npy', y)

y


## plot symbol of data
def plot_strokes(strokes_string):
    strokes = eval(strokes_string)
    for stroke in strokes:
        stroke = np.array(stroke)
        plt.plot(stroke[:, 0], stroke[:, 1], 'k')
        plt.axis('off')

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.uint8)

def strokes_to_array(strokes):
    plot_strokes(strokes)
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(32/DPI, 32/DPI)
    fig.canvas.draw()
    arr = np.array(fig.canvas.buffer_rgba())
    arr = rgb2gray(arr)
    fig.clear()
    return arr



##
def test(S, all=False):
    image_shape = (256, 256)

    if all:
        n = S.shape[0]
    else:
        n = 10000

    list_of_arrays = []
    for ids in tqdm(range(n)):
        strokes = S[ids]
        list_of_arrays.append(strokes_to_array(strokes))

        if (ids%10_000==0 or ids==n-1) and ids>0:
            np.save(f'X_{ids}', np.stack(list_of_arrays))
            print(f'[INFO] [Iteration {ids}] File saved!')
            list_of_arrays = []
            print(f'[INFO] [Iteration {ids}] Array deleted!')


# X = test(coords.strokes, all=True)


##
def merge_files():
    files = np.arange(10_000, 220_000, 10_000).astype(str).tolist() + ['210453']
    L = []
    for ids in files:
        filename = f'X_{ids}.npy'
        L.append( np.load(filename) )
    X = np.concatenate(L)
    np.save('X.npy', X)
    return X





