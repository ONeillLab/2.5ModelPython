import math
import numpy as np
import helperfunc as hf
import OEFrunmodel
from OEF_params import dt, u1, u2, v1, v2, spdrag1, spdrag2, rdist, dx, l, r, tpl
from OEF_modes import h1, h2, x, mode, tpulseper, tclock, tpulsedur, y

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
    u1_p, v1_p, h1_p = u1, v1, h1
    u2_p, v2_p, h2_p = u2, v2, h2
# ---------------------------------------
if AB == 3:
    u1_p, v1_p, h1_p = 0, 0, 0
    u2_p, v2_p, h2_p = 0, 0, 0
#########################################


ts = []
hm = []  # height max min!
psi2 = 0 * x
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
        tmp, u1, u1_p = u1, 1.5 * u1 - 0.5 * u1_p, tmp
        tmp, u2, u2_p = u2, 1.5 * u2 - 0.5 * u2_p, tmp
        tmp, v1, v1_p = v1, 1.5 * v1 - 0.5 * v1_p, tmp
        tmp, v2, v2_p = v2, 1.5 * v2 - 0.5 * v2_p, tmp
        tmp, h1, h1_p = h1, 1.5 * h1 - 0.5 * h1_p, tmp
        if layers == 2.5:
            tmp, h2, h2_p = h2, 1.5 * h2 - 0.5 * h2_p, tmp
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

    du1dt = hf.viscN2(u1, Re, n)  # helper
    du2dt = hf.viscN2(u2, Re, n)
    dv1dt = hf.viscN2(v1, Re, n)
    dv2dt = hf.viscN2(v2, Re, n)

    if spongedrag1 > 0:
        du1dt = du1dt - spdrag1 * u1
        du2dt = du2dt - spdrag2 * u2
        dv1dt = dv1dt - spdrag1 * v1
        dv2dt = dv2dt - spdrag2 * v2

    zeta1 = (
        1 - Bt * np.power(rdist, 2) + (1 / dx) * (v1 - v1[:, l - 1] + u1[l - 1, :] - u1)
    )
    zeta2 = (
        1
        - Bt * np.power(rdist, 2)
        + (1 / dx) * (v2 - v2[:, l - 1][:, np.newaxis] + u2[l - 1, :] - u2)
    )

    zv1 = np.multiply(zeta1, v1 + v1[:, l - 1][:, np.newaxis])
    zv2 = np.multiply(zeta2, v2 + v2[:, l - 1][:, np.newaxis])

    du1dt = du1dt + 0.25 * (zv1 + zv1[r - 1, :])
    du2dt = du2dt + 0.25 * (zv2 + zv2[r - 1, :])

    zu1 = np.multiply(zeta1, u1 + u1[:, l - 1][:, np.newaxis])
    zu2 = np.multiply(zeta2, u2 + u2[:, l - 1][:, np.newaxis])

    dv1dt = dv1dt + 0.25 * (zu1 + zu1[r - 1, :])
    dv2dt = dv2dt + 0.25 * (zu2 + zu2[r - 1, :])

    B = hf.BernN2(u1, v1, u2, v2, gm, c22h, c12h, h1, h2, ord)  # helper

    B1p = B[0]
    B2p = B[1]

    du1dtsq = du1dt - (1 / dx) * (B1p - B1p[:, l - 1])
    du2dtsq = du2dt - (1 / dx) * (B2p - B2p[:, l - 1])

    dv1dtsq = dv1dt - (1 / dx) * (B1p - B1p[l - 1, :])
    dv2dtsq = dv2dt - (1 / dx) * (B2p - B2p[l - 1, :])

    if AB == 2:
        u1sq = u1_p + dt * du1dtsq
        u2sq = u2_p + dt * du2dtsq

        v1sq = v1_p + dt * dv1dtsq
        v2sq = v2_p + dt * dv2dtsq

    if mode == 1:
        if t % tpulseper == 0 and t != 0:
            tclock = t
            locs = hf.paircountN2(num, N)  # helper
            wlayer = hf.pairshapeN2(locs, x, y, Br2, Wsh, N, dx)
            # helper
            newWmat = hf.pairfieldN2(L, dx, h1, wlayer)
            # helper
        if tclock + tpulsedur > t and tclock != 0:
            Wmat = newWmat
        elif t > tpulsedur:
            Wmat = 0 * np.multiply(x, y)
            tclock = 0

    Fx1 = hf.xflux(h1, u1, dx, dt) - np.divide(
        kappa, dx * (h1 - h1[:, l - 1])
    )  # helpers
    Fy1 = hf.yflux(h1, v1, dx, dt) - np.divide(kappa, dx * (h1 - h1[l - 1, :]))
    dh1dt = (1 / dx) * (Fx1[:, r - 1] - Fx1 + Fy1[r - 1, :] - Fy1)

    if layers == 2.5:
        Fx2 = hf.xflux(h2, u2, dx, dt) - np.divide(kappa, dx * (h2 - h2[:, l - 1]))
        Fy2 = hf.yflux(h2, v2, dx, dt) - np.divide(kappa, dx * (h2 - h2[l - 1, :]))
        dh2dt = (1 / dx) * (Fx2[:, r - 1] - Fx2 + Fy2[r - 1, :] - Fy2)

    if tradf > 0:
        dh1dt = dh1dt - 1 / tradf * (h1 - 1)
        dh2dt = dh2dt - 1 / tradf * (h2 - 1)

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
            u1_pp, u1_p, v1_pp, v1_p = u1_p, u1, v1_p, v1
            u2_pp, u2_p, v2_pp, v2_p = u2_p, u2, v2_p, v2
            h1_pp, h1_p = h1_p, h1
            if layers == 2.5:
                h2_pp, h2_p = h2_p, h2
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

    u1, u2, v1, v2 = u1sq, u2sq, v1sq, v2sq

    if tc % tpl == 0:
        print("mean h1 is " + np.mean(h1))
        ii += 1
        ts.append(t)

        u1mat.append(u1)
        u2mat.append(u2)
        v1mat.append(v1)
        v2mat.append(v2)
        h1mat.append(h1)
        h2mat.append(h2)

    tc += 1
    t = tc * dt
