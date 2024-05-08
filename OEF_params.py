import math
import numpy as np
import OEFrunmodel
#from OEFrunmodel import dx, L, spongedrag1, spongedrag2
import settings
#from settings import l, r, l2, r2, dx

dx = 1 / 5
dt = 1 / (2**8)
dtinv = 1 / dt
sampfreq = 5
tpl = sampfreq * dtinv

N = math.ceil(L / dx)
L = N * dx

l = np.concatenate((np.array(N), np.arange(1, N)), axis=None)
l2 = np.concatenate((np.arange(N - 1, N), np.arange(1, N - 1)), axis=None)
r = np.concatenate((np.arange(2, N), np.array(1)), axis=None)
r2 = np.concatenate((np.arange(3, N), np.arange(1, 2)), axis=None)

# h B grid
x, y = np.meshgrid(
    np.arange(0.5, N + 0.5) * dx - L / 2, np.arange(0.5, N + 0.5) * dx - L / 2
)
H = 1 + 0 * x
eta = 0 * x
h1 = 0 * x + 1
h2 = 0 * x + 1

# u grid
x, y = np.meshgrid(np.arange(0, N) * dx - L / 2, np.arange(0.5, N + 0.5) * dx - L / 2)
u1 = 0 * np.multiply(x, y)
u2 = u1

# v grid
x, y = np.meshgrid(np.arange(0.5, N + 0.5) * dx - L / 2, np.arange(0, N) * dx - L / 2)
v1 = 0 * np.multiply(x, y)
v2 = v1

# zeta grid
x, y = np.meshgrid(np.arange(0, N) * dx - L / 2, np.arange(0, N) * dx - L / 2)
rdist = np.sqrt(np.power(x, 2) + np.power(y, 2))
outerlim = L / 2 - 0.5
rlim = 1 * (rdist <= outerlim)  # 1* converts the Boolean values to integers 1 or 0


sponge1 = np.multiply(np.ones(N), np.maximum(rdist - outerlim, 0))
sponge1 = np.divide(sponge1, np.max(np.max(sponge1, axis=0)))
spdrag1 = np.multiply(spongedrag1, sponge1)


sponge2 = np.multiply(np.ones(N), np.maximum(rdist - outerlim, 0))
sponge2 = np.divide(sponge1, np.max(np.max(sponge1, axis=0)))
spdrag2 = np.multiply(spongedrag2, sponge2)
