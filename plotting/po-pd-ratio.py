import numpy as np
import matplotlib.pyplot as plt

thresh = 4 # survival threshold (vmax)
doub_width = 7.48
figsize = (doub_width,.8*doub_width)

x_range = np.r_[.05:1:.05]
y_range = x_range.copy()

def xy_ratio(x,y):
    return x/y


zgrid = np.zeros((len(x_range),len(y_range)))

col = -1
for x in x_range:
    col += 1
    row = -1
    for y in y_range:
        row += 1
        zgrid[row,col] = xy_ratio(x,y)

fig,ax = plt.subplots(figsize=figsize)

im = ax.pcolormesh(x_range,y_range,zgrid,cmap="Greys",vmin=1,vmax=thresh)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(im,ax=ax,label="x/y",extend="both")

ax.axes.set_aspect('equal')

plt.savefig("figures/po-pd-ratio.pdf")

plt.show()

