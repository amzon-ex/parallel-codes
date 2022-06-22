import sys
import matplotlib.pyplot as plt
from numpy import fromfile, float64, diff, nonzero

try:
    filename = sys.argv[1]
except IndexError:
    filename = 'paralap_out'

data = fromfile(filename, dtype = float64)
ydata = data[::3]
xdata = data[1::3]
zdata = data[2::3]

# heuristically infer shape of XY-grid
nx = nonzero(diff(xdata))[0][0] + 1
ny = ydata.size // nx

print(f'Heuristically inferred shape: nx={nx}, ny={ny}')

ydata = ydata.reshape(ny, -1)
xdata = xdata.reshape(-1, nx)
zdata = zdata.reshape(ny, nx)

# plt.grid(True, alpha = 0.3)
# plt.figure(figsize = (10, 10))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.xlim(0.45, 0.55)
# plt.ylim(0.9, 1.0)
# plt.pcolormesh(xdata, ydata, zdata)

fig = plt.figure(figsize = (10, 10))
ax = plt.axes(projection = '3d')
ax.set_xlabel('x', fontsize = 12)
ax.set_ylabel('y', fontsize = 12)
ax.set_xlim(0.45, 0.55)
ax.set_ylim(0.9, 1.0)
ax.plot_surface(xdata, ydata, zdata, cmap = 'viridis')

plt.show()
