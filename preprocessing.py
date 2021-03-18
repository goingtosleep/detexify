import numpy as np
import pandas as pd
import matplotlib.pyplot as pl


## Load data
X = np.load('X.npy')
y = np.load('y.npy', allow_pickle=True)


## Utils
def plot_grid(n=3, start=0):
    for ids in range(n**2):
        t = X[ids+start]
        pl.subplot(n, n, ids+1)
        pl.imshow(t, cmap='gray')
        pl.title(y[ids+start], fontsize=8, y=-0.2)
        pl.axis('off')
    pl.show()


## Random data visualization
start_ids = np.random.randint(0, 210_000)
plot_grid(8, start_ids)


## Filter top 500 classes
def summarize_classes():
    L = [(item, np.sum(y==item)) for item in np.unique(y)]
    return pd.DataFrame.from_records(L, columns=['labels', 'count'])

classes_summary = summarize_classes()
top_classes = classes_summary.sort_values('count')[lambda x: x['count']>100]

##
t = classes_summary.sort_values('count')[lambda x: x['count']>50]
t = t.sort_values('count', ascending=False)[:500]


##
def ids_in_top_500_labels():
    classes_summary = summarize_classes()
    t = classes_summary.sort_values('count')[lambda x: x['count']>50]
    t = t.sort_values('count', ascending=False)[:500]
    top_500 = t.labels.values

    ids = []
    for i, label in enumerate(y):
        if label in top_500:
            ids.append(i)
    return ids

ids = ids_in_top_500_labels()


## Save data
# np.save('X_500.npy', X[ids])
# np.save('y_500.npy', y[ids])

## Processing for pytorch
# X = np.load('X_500.npy')
# y = np.load('y_500.npy', allow_pickle=True)
# y_str, y = np.unique(y, return_inverse=True)

# X = np.invert(X).astype(np.float32)
# X[X<10] = 0
# X[X>245] = 255
# X = X[:, None] / 255

# np.save('X_500_processed.npy', X)
# np.save('y_500_processed.npy', y)



