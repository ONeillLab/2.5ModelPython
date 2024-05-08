import numpy as np
import helperfunc as hf
import OEFrunmodel
import OEF_params
#from OEFrunmodel import num, Br2, Wsh, tstpf, tstf, layers
#from OEF_params import N, dx, x, y, L, h1, spdrag1

locs = hf.paircountN2(num, N)  # paircountN2 is a helper function
mode = 1  # set mode to 1

match mode:
    case 1:
        pulse = "off"
        wlayer = hf.pairshapeN2(locs, x, y, Br2, Wsh, N, dx)
        Wmat = hf.pairfieldN2(L, dx, h1, wlayer)
        Wmatorig = Wmat
        tpulseper = tstpf
        tpulsedur = tstf
        tclock = 0
        FreeGrid = np.count_nonzero(spdrag1 == 0) / (N**2)

        # h B grid
        x, y = np.meshgrid(
            np.arange(0.5, N + 0.5) * dx - L / 2, np.arange(0.5, N + 0.5) * dx - L / 2)
        H = 1 + 0 * x
        if layers == 2:
            h = np.multiply(0, x) + 0.5
            h10 = 0.5
        else:
            eta = np.multiply(0, x)
            h1 = np.multiply(0, x) + 1
            h2 = np.multiply(0, x) + 1

        # u grid
        x, y = np.meshgrid(
            np.arange(0, N) * dx - L / 2, np.arange(0.5, N + 0.5) * dx - L / 2)

        # v grid
        x, y = np.meshgrid(
            np.arange(0.5, N + 0.5) * dx - L / 2, np.arange(0, N) * dx - L / 2)
