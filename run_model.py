import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import helpers 
from parameters import *

locs = helpers.paircountN2(num, N - 1)  # paircountN2 is a helper function
mode = 1  # set mode to 1

match mode:
    case 1:
        pulse = "off"
        wlayer = helpers.pairshapeN2(locs, x, y, Br2, Wsh, N, dx)
        Wmat = helpers.pairfieldN2(L, dx, h1, wlayer)
        Wmatorig = Wmat
        tpulseper = tstpf
        tpulsedur = tstf
        tclock = 0
        FreeGrid = np.sum(spdrag1 == 0) / (N**2)

        #  FreeGrid = np.count_nonzero(spdrag1 == 0) / (N**2)

        # h B grid
        """
        x, y = np.meshgrid(
            np.arange(0.5, N + 0.5) * dx - L / 2, np.arange(0.5, N + 0.5) * dx - L / 2
        )
        H = 1 + 0 * x
        if layers == 2:
            h = 0*x + 0.5
            h10 = 0.5
        else:
            eta = 0*x
            h1 = 0*x + 1
            h2 = 0*x + 1

        # u grid
        x, y = np.meshgrid(
            np.arange(0, N) * dx - L / 2, np.arange(0.5, N + 0.5) * dx - L / 2
        )

        # v grid
        x, y = np.meshgrid(
            np.arange(0.5, N + 0.5) * dx - L / 2, np.arange(0, N) * dx - L / 2
        )
        """

t = 0
tc = 0


uhatvec = 0
del2psivec = 0
psi2vec = 0
CFL1vec = 0
CFL2vec = 0


# TIME STEPPING
#########################################
if AB == 2:
    u1_p = u1
    v1_p = v1
    h1_p = h1
    u2_p = u2
    v2_p = v2
    h2_p = h2
# ---------------------------------------
if AB == 3:
    u1_p, v1_p, h1_p = 0, 0, 0
    u2_p, v2_p, h2_p = 0, 0, 0
#########################################


ts = []
hm = []  # height max min!
psi2 = np.zeros(np.shape(x))
dhdt = psi2
pv1 = psi2
pv2 = psi2
zeta1 = psi2
zeta2 = psi2
B2 = psi2
B1p = B2
pv1 = B2
pv2 = B2

#  u  h,B
#  z  v
#

##
# -------------------------------------------------
# figure
ii = 0
zeta1mat = []
zeta2mat = []
hmat = []
Wpulsemat = []
u1mat = []
u2mat = []
v1mat = []
v2mat = []
h1mat = []
h2mat = []

while t <= tmax + dt / 2:
    if AB == 2:
        tmp = u1
        u1 = 1.5 * u1 - 0.5 * u1_p
        u1_p = tmp
        tmp = u2
        u2 = 1.5 * u2 - 0.5 * u2_p
        u2_p = tmp
        tmp = v1
        v1 = 1.5 * v1 - 0.5 * v1_p
        v1_p = tmp
        tmp = v2
        v2 = 1.5 * v2 - 0.5 * v2_p
        v2_p = tmp
        tmp = h1
        h1 = 1.5 * h1 - 0.5 * h1_p
        h1_p = tmp
        if layers == 2.5:
            tmp = h2
            h2 = 1.5 * h2 - 0.5 * h2_p
            h2_p = tmp
    if AB == 3:
        if tc > 1:
            u1s, u1, u1_pp, u1_p = (
                u1,
                23 / 12 * u1 - 16 / 12 * u1_p + 5 / 12 * u1_pp,
                u1_p,
                u1s,
            )
            v1s, v1, v1_pp, v1_p = (
                v1,
                23 / 12 * v1 - 16 / 12 * v1_p + 5 / 12 * v1_pp,
                v1_p,
                v1s,
            )
            h1s, h1, h1_pp, h1_p = (
                h1,
                23 / 12 * h1 - 16 / 12 * h1_p + 5 / 12 * h1_pp,
                h1_p,
                h1s,
            )
            u2s, u2, u2_pp, u2_p = (
                u2,
                23 / 12 * u2 - 16 / 12 * u2_p + 5 / 12 * u2_pp,
                u2_p,
                u2s,
            )
            v2s, v2, v2_pp, v2_p = (
                v2,
                23 / 12 * v2 - 16 / 12 * v2_p + 5 / 12 * v2_pp,
                v2_p,
                v2s,
            )
        if layers == 2.5:
            h2s, h2, h2_pp, h2_p = (
                h2,
                23 / 12 * h2 - 16 / 12 * h2_p + 5 / 12 * h2_pp,
                h2_p,
                h2s,
            )

    du1dt = helpers.viscN2(u1, Re, n)  # helper
    du2dt = helpers.viscN2(u2, Re, n)
    dv1dt = helpers.viscN2(v1, Re, n)
    dv2dt = helpers.viscN2(v2, Re, n)

    if spongedrag1 > 0:
        du1dt = du1dt - spdrag1 * u1
        du2dt = du2dt - spdrag2 * u2
        dv1dt = dv1dt - spdrag1 * v1
        dv2dt = dv2dt - spdrag2 * v2

    zeta1 = (
        1
        - Bt * (rdist**2)
        + (1 / dx) * (v1 - np.roll(v1, 1, axis=1) + np.roll(u1, 1, axis=0) - u1)
    )
    zeta2 = (
        1
        - Bt * (rdist**2)
        + (1 / dx) * (v2 - np.roll(v2, 1, axis=1) + np.roll(u2, 1, axis=0) - u2)
    )

    zv1 = zeta1 * (v1 + np.roll(v1, 1, axis=1))
    zv2 = zeta2 * (v2 + np.roll(v2, 1, axis=1))

    du1dt = du1dt + 0.25 * (zv1 + np.roll(zv1, -1, axis=0))
    du2dt = du2dt + 0.25 * (zv2 + np.roll(zv2, -1, axis=0))

    zu1 = zeta1 * (u1 + np.roll(u1, 1, axis=0))
    zu2 = zeta2 * (u2 + np.roll(u2, 1, axis=0))

    dv1dt = dv1dt - 0.25 * (zu1 + np.roll(zu1, -1, axis=1))
    dv2dt = dv2dt - 0.25 * (zu2 + np.roll(zu2, -1, axis=1))

    B1p, B2p = helpers.BernN2(u1, v1, u2, v2, gm, c22h, c12h, h1, h2, ord)  # helper

    du1dtsq = du1dt - (1 / dx) * (B1p - np.roll(B1p, 1, axis=1))
    du2dtsq = du2dt - (1 / dx) * (B2p - np.roll(B2p, 1, axis=1))

    dv1dtsq = dv1dt - (1 / dx) * (B1p - np.roll(B1p, 1, axis=0))
    dv2dtsq = dv2dt - (1 / dx) * (B2p - np.roll(B2p, 1, axis=0))

    if AB == 2:
        u1sq = u1_p + dt * du1dtsq
        u2sq = u2_p + dt * du2dtsq

        v1sq = v1_p + dt * dv1dtsq
        v2sq = v2_p + dt * dv2dtsq

    if mode == 1:
        if t % tpulseper == 0 and t != 0:
            tclock = t
            locs = helpers.paircountN2(num, N - 1)  # helper
            wlayer = helpers.pairshapeN2(locs, x, y, Br2, Wsh, N, dx)
            # helper
            newWmat = helpers.pairfieldN2(L, dx, h1, wlayer)
            # helper
        if tclock + tpulsedur > t and tclock != 0:
            Wmat = newWmat
        elif t > tpulsedur:
            Wmat = 0 * np.multiply(x, y)
            tclock = 0

    Fx1 = helpers.xflux(h1, u1, dx, dt) - kappa / dx * (h1 - np.roll(h1, 1, axis=1))  # helpers
    Fy1 = helpers.yflux(h1, v1, dx, dt) - kappa / dx * (h1 - np.roll(h1, 1, axis=0))
    dh1dt = -(1 / dx) * (
        np.roll(Fx1, -1, axis=1) - Fx1 + np.roll(Fy1, -1, axis=0) - Fy1
    )

    if layers == 2.5:
        Fx2 = helpers.xflux(h2, u2, dx, dt) - kappa / dx * (h2 - np.roll(h2, 1, axis=1))
        Fy2 = helpers.yflux(h2, v2, dx, dt) - kappa / dx * (h2 - np.roll(h2, 1, axis=0))
        dh2dt = -(1 / dx) * (
            np.roll(Fx2, -1, axis=1) - Fx2 + np.roll(Fy2, -1, axis=0) - Fy2
        )

    if tradf > 0:
        dh1dt = dh1dt - (1 / tradf) * (h1 - 1)
        dh2dt = dh2dt - (1 / tradf) * (h2 - 1)

    if mode == 1:
        dh1dt = dh1dt + Wmat
        if layers == 2.5:
            dh2dt = dh2dt - H1H2 * Wmat

    if AB == 3:
        if tc <= 1:
            du1dt1 = u1sq + dt * du1dtsq
            du2dt1 = u2sq + dt * du2dtsq
            dv1dt1 = v1sq + dt * dv1dtsq
            dv2dt1 = v2sq + dt * dv2dtsq
            dh1dt1 = h1 + dt * dh1dt
            if layers == 2.5:
                dh2dt1 = h2 + dt * dh2dt
            u1_pp = u1_p
            u1_p = u1
            v1_pp = v1_p
            v1_p = v1
            u2_pp = u2_p
            u2_p = u2
            v2_pp = v2_p
            v2_p = v2
            h1_pp = h1_p
            h1_p = h1
            if layers == 2.5:
                h2_pp = h2_p
                h2_p = h2
            u1sq = u1sq + dt / 2 * (du1dtsq + du1dt1)
            u2sq = u2sq + dt / 2 * (du2dtsq + du2dt1)
            v1sq = v1sq + dt / 2 * (dv1dtsq + dv1dt1)
            v2sq = v2sq + dt / 2 * (dv2dtsq + dv2dt1)
            h1 = h1 + dt / 2 * (dh1dt + dh1dt1)
            if layers == 2.5:
                h2 = h2 + dt / 2 * (dh2dt + dh2dt1)
        else:
            h1 = h1_p + dt * dh1dt
            if layers == 2.5:
                h2 = h2_p + dt * dh2dt

    if AB == 2:
        h1 = h1_p + dt * dh1dt
        if layers == 2.5:
            h2 = h2_p + dt * dh2dt

    u1 = u1sq
    u2 = u2sq
    v1 = v1sq
    v2 = v2sq

    if tc % tpl == 0:
        print("mean h1 is " + str(np.mean(h1)))
        ii += 1
        ts.append(t)

        u1mat.append(u1)
        u2mat.append(u2)
        v1mat.append(v1)
        v2mat.append(v2)
        h1mat.append(h1)
        h2mat.append(h2)

        frames.append(zeta2)
        time = fig.text(0.19, 0.9, 't = ' + str(t))
        times.append(time)

    if math.isnan(h1[0, 0]):
        print("break")
        break

    tc += 1
    t = tc * dt


#print(zeta2)
#print(np.sum(zeta2))
#df = pd.DataFrame(zeta2)
#df.to_csv('file2.csv',index=False)

#control = pd.read_csv('file.csv', sep=',', header=None)
#control = np.asarray(control)
#testzeta = frames[-1][39:110,49:105] 
#print(np.max(testzeta))
#print(np.min(testzeta))
#testzeta_max = np.max(testzeta)
#testzeta_min = np.min(testzeta)
#control_max = np.max(control)
#control_min = np.min(control)
#numerator = testzeta_max - testzeta_min
#denominator = control_max - control_min
#slope = numerator/denominator
#intercept = testzeta_min - slope * control_min
#control = control * slope + intercept 
#print(np.max(control))
#print(np.min(control))
#print(np.sum(np.sum((np.abs(testzeta - control)))))

cv0 = frames[0]
im = ax.imshow(cv0, origin='lower') # Here make an AxesImage rather than contour
cb = fig.colorbar(im, cax=cax)

def animate(i):
    arr = frames[i]
    vmax = np.max(arr)
    vmin = np.min(arr)
    for txt in fig.texts:
        txt.set_visible(False)
    fig.text(0.3, 0.9, 'a0Ld = ' + str(aOLd)).set_visible(True)
    fig.text(0.45, 0.9, 'Ephat = ' + str(Ephat)).set_visible(True)
    times[i].set_visible(True)
    im.set_data(arr)
    im.set_clim(vmin, vmax)

#ani = ArtistAnimation(fig, frames, interval=250, repeat=True, blit=True)
ani = animation.FuncAnimation(fig, animate, interval=ani_interval, frames=int(tmax/5)+1)
ani.save("test.mp4")
plt.show()


