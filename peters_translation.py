import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

# 2.5 layer SW polar cap model as described in:
#
# O'Neill et al. 2015, Nature Geosci.: http://dx.doi.org/10.1038/ngeo2459
# O'Neill et al. 2016, J. Atmos. Sci.: http://dx.doi.org/10.1175/JAS-D-15-0314.1
#
# Send questions or comments to Morgan O'Neill: morgan.e.oneill@gmail.com
#


# This model has:
# 2 and 2.5 model option
# hyperviscosity
# radiative relaxation (Rayleigh damping) on height fields
# 2nd and 3rd order Adams-Bashforth time stepping
# sponge drag on velocity to damp square Cartesian corner dynamics
# baroclinic drag

################# Metaparameters ##############################

simnumber = 2  # number of simulations you want to run
tmax = 1000

#fig, ax = plt.subplots()
#frames = []

fig = plt.figure()
ax = fig.add_subplot(111)
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')
frames = []
times = []

################# Controlling parameters ######################
# these are the only parameters you should vary to explore physical space

### ND = nondimensional

c22h = 3#9  # ND 2nd baroclinic gravity wave speed squared
c12h = 4#10  # ND 1st baroclinic gravity wave speed squared
H1H2 = 1  # ND upper to lower layer height ratio
Bt = (1**2) / 2 / (50**2)  # ND scaled beta Ld2^2/4a^2 ### adjust this
Br2 = 1#1.5  # ND scaled storm size: Burger number Ld2^2/Rst^2
p1p2 = 0.95  # ND upper to lower layer density ratio
tstf = 6#48  # ND storm duration tst*f0
tstpf = 15#60  # ND period between forced storms tstp*f0
tradf = 2000  # ND Newtonian damping of layer thickness trad*f0
Ar = 0.15  # ND areal storm coverage
Re = 5e4  # ND Reynolds number
Wsh = 0.03 / 2  # ND convective Rossby number


################## derived quantities ##########################

gm = p1p2 * c22h / c12h * H1H2  # ND reduced gravity
aOLd = np.sqrt(1 / Bt / 2)  # ND planetary radius to deformation radius ratio
print(aOLd)
L = 3 * np.pi / 9 * aOLd  ###???  # ND num = ceil(numfrc.*L.^2./Br2)
num = round(Ar * (L**2) * Br2 / np.pi)  # number of storms
deglim = 90 - 3 * L / 2 * aOLd * 180 / np.pi  # domain size [degrees]


################## engineering params ##########################

AB = 2  # order of Adams-Bashforth scheme (2 or 3)
layers = 2.5  # # of layers (2 or 2.5)
n = 2  # order of Laplacian '2' is hyperviscosity
kappa = 1e-6
ord = (
    2  # must equal 1 for Glenn's order, otherwise for Sadourney's (squares before avgs)
)
spongedrag1 = 0.1
spongedrag2 = 0.1

########### OEF_params ##############
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################

dx = 1 / 5
dt = 1 / (2**8)
dtinv = 1 / dt
sampfreq = 5
tpl = sampfreq * dtinv

N = math.ceil(L / dx)
L = N * dx

l = np.concatenate((np.array([N]), np.arange(1, N)), axis=None)

l2 = np.concatenate((np.arange(N - 1, N + 1), np.arange(1, N - 1)), axis=None)

r = np.concatenate((np.arange(2, N + 1), np.array([1])), axis=None)

r2 = np.concatenate((np.arange(3, N + 1), np.arange(1, 3)), axis=None)

# l = [158, 1, ..., 157]
# l2 = [157, 158, 1, ..., 156]
# r = [2, 3, ..., 158, 1]
# r2 = [3, 4, ..., 158, 1, 2]

# h B grid
"""
N +0.5 or N?

"""
x, y = np.meshgrid(
    np.arange(0.5, N + 0.5) * dx - L / 2, np.arange(0.5, N + 0.5) * dx - L / 2
)
H = 1 + 0 * x
eta = 0 * x
h1 = 0 * x + 1
h2 = 0 * x + 1

# u grid
x, y = np.meshgrid(np.arange(0, N) * dx - L / 2, np.arange(0.5, N + 0.5) * dx - L / 2)
u1 = 0 * x * y
u2 = u1

# v grid
x, y = np.meshgrid(np.arange(0.5, N + 0.5) * dx - L / 2, np.arange(0, N) * dx - L / 2)
v1 = 0 * x * y
v2 = v1

# zeta grid
x, y = np.meshgrid(np.arange(0, N) * dx - L / 2, np.arange(0, N) * dx - L / 2)
rdist = np.sqrt((x**2) + (y**2))
outerlim = L / 2 - 0.5
rlim = (rdist <= outerlim).astype(
    float
)  # 1* converts the Boolean values to integers 1 or 0


sponge1 = np.ones(N) * np.maximum(rdist - outerlim, 0)
sponge1 = sponge1 / np.max(sponge1)  #####
spdrag1 = spongedrag1 * sponge1


sponge2 = np.ones(N) * np.maximum(rdist - outerlim, 0)
sponge2 = sponge2 / np.max(sponge1)
spdrag2 = spongedrag2 * sponge2

########### helpers ##############
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################


def pairshapeN2(locs, x, y, Br2, Wsh, N, dx):
    rad = math.ceil(np.sqrt(1 / Br2) / dx)
    xg, yg = np.meshgrid(np.arange(-rad, rad + 1), np.arange(-rad, rad + 1))
    gaus = Wsh * np.exp(-(Br2 * dx**2) / 0.3606 * ((xg + 0.5) ** 2 + (yg + 0.5) ** 2))

    wlayer = np.zeros(np.shape(x))

    buf = rad
    bufmat = np.zeros((N + 2 * rad, N + 2 * rad))
    nlocs = locs + rad

    jj = 1
    corners = nlocs - rad
    corners = corners.astype(np.int64)
    while jj <= np.shape(locs)[0]:
        bufmat[
            (corners[jj - 1, 0] - 1) : (corners[jj - 1, 0] + 2 * rad),
            (corners[jj - 1, 1] - 1) : (corners[jj - 1, 1] + 2 * rad),
        ] += gaus

        # bufmat[(corners[jj - 1, 0] - 1):(corners[jj - 1, 0] + 2*rad),(corners[jj - 1, 1] - 1):(corners[jj - 1, 1] + 2*rad)] + gaus

        jj += 1

    wlayer = bufmat[buf : N + buf, buf : N + buf]

    addlayer1 = np.zeros(np.shape(wlayer))
    addlayer2 = addlayer1
    addlayer3 = addlayer1
    addlayer4 = addlayer1

    addcorn1 = addlayer1
    addcorn2 = addcorn1
    addcorn3 = addcorn1
    addcorn4 = addcorn1

    addlayer1[0:buf, :] = bufmat[buf + N :, buf : N + buf]
    addlayer2[:, 0:buf] = bufmat[buf : N + buf, buf + N :]
    addlayer3[-buf:, :] = bufmat[0:buf, buf : N + buf]
    addlayer4[:, -buf:] = bufmat[buf : N + buf, 0:buf]

    addcorn1[0:buf, 0:buf] = bufmat[buf + N :, buf + N :]
    addcorn2[-buf:, -buf:] = bufmat[0:buf, 0:buf]
    addcorn3[0:buf, -buf:] = bufmat[buf + N :, 0:buf]
    addcorn4[-buf:, 0:buf] = bufmat[0:buf, buf + N :]

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
    wcorrect = voldw / area
    Wmat = wlayer - wcorrect

    return Wmat


def paircountN2(num, N):
    locs = np.ceil(np.random.random((num, 2)) * N).astype(int)

    return locs


def yflux(f, v, dx, dt):
    fl = np.roll(f, 1, axis=0)
    fr = f
    fa = 0.5 * v * (fl + fr)

    return fa


def xflux(f, u, dx, dt):
    fl = np.roll(f, 1, axis=1)
    fr = f
    fa = 0.5 * u * (fl + fr)

    return fa


def viscN2(vel, Re, n):

    if n == 1:

        field = (
            -4 * vel
            + np.roll(vel, 1, axis=0)
            + np.roll(vel, -1, axis=0)
            + np.roll(vel, 1, axis=1)
            + np.roll(vel, -1, axis=1)
        )
        field = (
            n / dx**2
        ) * field  # in Morgan's code the n in this line is 'nu', but that's never defined; I think it's a typo

    if n == 2:

        field = (
            20 * vel
            + 2 * np.roll(np.roll(vel, 1, axis=0), 1, axis=1)
            + 2 * np.roll(np.roll(vel, 1, axis=0), -1, axis=1)
            + 2 * np.roll(np.roll(vel, -1, axis=0), 1, axis=1)
            + 2 * np.roll(np.roll(vel, -1, axis=0), -1, axis=1)
            - 8 * np.roll(vel, 1, axis=0)
            - 8 * np.roll(vel, -1, axis=0)
            - 8 * np.roll(vel, 1, axis=1)
            - 8 * np.roll(vel, -1, axis=1)
            + np.roll(vel, 2, axis=0)
            + np.roll(vel, -2, axis=0)
            + np.roll(vel, 2, axis=0)  # TYPO ?
            + np.roll(vel, -2, axis=1)
        )

        field = -1 / Re * (1 / dx**4) * field

    return field


def gauss(x, y, L):
    g = np.exp(-0.5 * (x**2) + (x**2) / L**2)

    return g


def BernN2(u1, v1, u2, v2, gm, c22h, c12h, h1, h2, ord):
    if ord == 1:
        B1 = "broke"
        B2 = "broke"
    else:
        B1 = (
            c12h * h1
            + c22h * h2
            + 0.25
            * (
                (u1**2)
                + (np.roll(u1, -1, axis=1) ** 2)
                + (v1**2)
                + (np.roll(v1, -1, axis=0) ** 2)
            )
        )

        B2 = (
            gm * c12h * h1
            + c22h * h2
            + 0.25
            * (
                (u1**2)
                + (np.roll(u1, -1, axis=1) ** 2)
                + (v1**2)
                + (np.roll(v1, -1, axis=0) ** 2)
            )
        )

    return B1, B2


########### OEF_modes ##############
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################

locs = paircountN2(num, N - 1)  # paircountN2 is a helper function
mode = 1  # set mode to 1

match mode:
    case 1:
        pulse = "off"
        wlayer = pairshapeN2(locs, x, y, Br2, Wsh, N, dx)
        Wmat = pairfieldN2(L, dx, h1, wlayer)
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

########### OEF_difeqs ##############
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################

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

    du1dt = viscN2(u1, Re, n)  # helper
    du2dt = viscN2(u2, Re, n)
    dv1dt = viscN2(v1, Re, n)
    dv2dt = viscN2(v2, Re, n)

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

    B1p, B2p = BernN2(u1, v1, u2, v2, gm, c22h, c12h, h1, h2, ord)  # helper

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
            locs = paircountN2(num, N - 1)  # helper
            wlayer = pairshapeN2(locs, x, y, Br2, Wsh, N, dx)
            # helper
            newWmat = pairfieldN2(L, dx, h1, wlayer)
            # helper
        if tclock + tpulsedur > t and tclock != 0:
            Wmat = newWmat
        elif t > tpulsedur:
            Wmat = 0 * np.multiply(x, y)
            tclock = 0

    Fx1 = xflux(h1, u1, dx, dt) - kappa / dx * (h1 - np.roll(h1, 1, axis=1))  # helpers
    Fy1 = yflux(h1, v1, dx, dt) - kappa / dx * (h1 - np.roll(h1, 1, axis=0))
    dh1dt = -(1 / dx) * (
        np.roll(Fx1, -1, axis=1) - Fx1 + np.roll(Fy1, -1, axis=0) - Fy1
    )

    if layers == 2.5:
        Fx2 = xflux(h2, u2, dx, dt) - kappa / dx * (h2 - np.roll(h2, 1, axis=1))
        Fy2 = yflux(h2, v2, dx, dt) - kappa / dx * (h2 - np.roll(h2, 1, axis=0))
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
        time = fig.text(0.2, 0.9, 't = ' + str(t))
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
control = pd.read_csv('file.csv', sep=',', header=None)
control = np.asarray(control)
testzeta = frames[-1][39:110,49:105] 
print(np.max(testzeta))
print(np.min(testzeta))
testzeta_max = np.max(testzeta)
testzeta_min = np.min(testzeta)
control_max = np.max(control)
control_min = np.min(control)
numerator = testzeta_max - testzeta_min
denominator = control_max - control_min
slope = numerator/denominator
intercept = testzeta_min - slope * control_min
control = control * slope + intercept 
print(np.max(control))
print(np.min(control))
print(np.sum(np.sum((np.abs(testzeta - control)))))

cv0 = frames[0]
im = ax.imshow(cv0, origin='lower') # Here make an AxesImage rather than contour
cb = fig.colorbar(im, cax=cax)

def animate(i):
    arr = frames[i]
    vmax = np.max(arr)
    vmin = np.min(arr)
    for txt in fig.texts:
        txt.set_visible(False)
    times[i].set_visible(True)
    im.set_data(arr)
    im.set_clim(vmin, vmax)

#ani = ArtistAnimation(fig, frames, interval=250, repeat=True, blit=True)
ani = animation.FuncAnimation(fig, animate, interval=200, frames=int(tmax/5))
ani.save("test.mp4")
plt.show()



########### helpers ##############
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
