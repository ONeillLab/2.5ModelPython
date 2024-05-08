import math
import numpy as np
#from OEF_params import l, r, l2, r2, dx


def pairshapeN2(locs, x, y, Br2, Wsh, N, dx):
    rad = math.ceil(np.sqrt(Br2 ** (-1)) / dx)
    xg, yg = np.meshgrid(np.arange(-rad, rad + 1), np.arange(-rad, rad + 1))
    gaus = Wsh * np.exp(
        -(Br2 * dx**2) / 0.3606 * (np.power(xg + 0.5, 2) + np.power(yg + 0.5, 2))
    )

    wlayer = np.zeros(np.shape(x))

    buf = rad
    bufmat = np.zeros((N + 2 * rad, N + 2 * rad))
    nlocs = locs + rad

    jj = 1
    corners = nlocs - rad
    while jj <= np.shape(locs)[0]:
        bufmat[
            int(corners[jj - 1, 0] - 1) : int(corners[jj - 1, 0] + 2 * rad),
            int(corners[jj - 1, 1] - 1) : int(corners[jj - 1, 1] + 2 * rad),
        ] = (
            bufmat[
                int(corners[jj - 1, 0] - 1) : int(corners[jj - 1, 0] + 2 * rad),
                int(corners[jj - 1, 1] - 1) : int(corners[jj - 1, 1] + 2 * rad),
            ]
            + gaus
        )
        jj += 1

    wlayer = bufmat[buf : N + buf, buf : N + buf]

    addlayer1 = np.zeros(np.shape(wlayer))
    addlayer2, addlayer3, addlayer4 = addlayer1, addlayer1, addlayer1

    addcorn1 = addlayer1
    addcorn2 = addcorn1
    addcorn3 = addcorn1
    addcorn4 = addcorn1

    addlayer1[0:buf, :] = bufmat[buf + N :, buf : N + buf]
    addlayer2[:, 0:buf] = bufmat[buf : N + buf, buf + N :]
    addlayer3[N - buf : N, :] = bufmat[0:buf, buf : N + buf]
    addlayer4[:, N - buf : N] = bufmat[buf : N + buf, 0:buf]

    addcorn1[0:buf, 0:buf] = bufmat[buf + N :, buf + N :]
    addcorn2[N - buf :, N - buf :] = bufmat[0:buf, 0:buf]
    addcorn3[0:buf, N - buf :] = bufmat[buf + N :, 0:buf]
    addcorn4[N - buf :, 0:buf] = bufmat[0:buf, buf + N :]

    wlayer = (
        wlayer
        + addlayer1
        + addlayer2
        + addlayer3
        + addlayer4
        + addcorn1
        + addcorn2
        + addcorn3
        + addcorn4
    )

    return wlayer


def pairfieldN2(L, dx, h1, wlayer):
    voldw = np.sum(wlayer) * dx**2
    area = L**2
    wcorrect = np.divide(voldw, area)
    Wmat = wlayer - wcorrect

    return Wmat


def paircountN2(num, N):
    locs = np.ceil(np.random.random((num, 2)) * N)
    return locs


def yflux(f, v, dx, dt):
    fl = np.roll(f, 1, axis=0)
    fr = f
    fa = 0.5 * v * (fl + fr)

    return fa


def xflux(f, u, dx, dt):
    fl = f[:, l - 1]
    fr = f
    fa = 0.5 * u * (fl + fr)

    return fa


def viscN2(vel, Re, n):

    if n == 1:

        field = np.roll(vel, 1, axis=0) + vel[r - 1, :] + vel[:, l - 1] + vel[:, r - 1] - 4 * vel
        field = (
            n / dx**2
        ) * field  # in Morgan's code the n in this line is 'nu', but that's never defined; I think it's a typo

    if n == 2:

        field = (
            2 * vel[l - 1, l - 1]
            + 2 * vel[l - 1, r - 1]
            + 2 * vel[r - 1, l - 1]
            + 2 * vel[r - 1, r - 1]
            - 8 * vel[l - 1, :]
            - 8 * vel[r - 1, :]
            - 8 * vel[:, l - 1]
            - 8 * vel[:, r - 1]
            + vel[l2 - 1, :]
            + vel[r2 - 1, :]
            + vel[:, l2 - 1]
            + vel[:, r2 - 1]
            + 20 * vel
        )

        field = -1 / Re * (1 / dx**4) * field

    return field


def gauss(x, y, L):
    g = np.exp(-0.5 * (np.power(x, 2) + np.power(x, 2)) / L**2)

    return g


def BernN2(u1, v1, u2, v2, gm, c22h, c12h, h1, h2, ord):
    if ord == 1:
        B1 = "broke"
        B2 = "broke"
    else:
        B1 = (
            c12h
            + h1
            + c22h * h2
            + 0.25
            * (
                np.power(u1, 2)
                + np.power(u1[:, r - 1], 2)[:, np.newaxis]
                + np.power(v1, 2)
                + np.power(v1[r - 1, :], 2)[:, np.newaxis]
            )
        )

        B2 = (
            gm * c12h * h1
            + c22h * h2
            + 0.25
            * (
                np.power(u1, 2)
                + np.power(u1[:, r - 1], 2)[:, np.newaxis]
                + np.power(v1, 2)
                + np.power(v1[r - 1, :], 2)[:, np.newaxis]
            )
        )

    return B1, B2
