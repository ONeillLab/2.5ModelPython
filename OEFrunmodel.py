import numpy as np


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

################# Controlling parameters ######################
# these are the only parameters you should vary to explore physical space

### ND = nondimensional

c22h = 9  # ND 2nd baroclinic gravity wave speed squared
c12h = 10  # ND 1st baroclinic gravity wave speed squared
H1H2 = 1  # ND upper to lower layer height ratio
Bt = (1**2) / 2 / (30**2)  # ND scaled beta Ld2^2/4a^2
Br2 = 1.5  # ND scaled storm size: Burger number Ld2^2/Rst^2
p1p2 = 0.95  # ND upper to lower layer density ratio
tstf = 48  # ND storm duration tst*f0
tstpf = 60  # ND period between forced storms tstp*f0
tradf = 2000  # ND Newtonian damping of layer thickness trad*f0
Ar = 0.15  # ND areal storm coverage
Re = 5**4  # ND Reynolds number
Wsh = 0.03 / 2  # ND convective Rossby number


################## derived quantities ##########################

gm = p1p2 * c22h / c12h * H1H2  # ND reduced gravity
aOLd = np.sqrt(1 / Bt / 2)  # ND planetary radius to deformation radius ratio
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
tmax = 8000
spongedrag1 = 0.1
spongedrag2 = 0.1


# run simulations below

# simcounter = 1 renamed 'iind' to 'simcounter'
#while simcounter < (simnumber + 1):
    # OEF_params
    # OEF_modes
    # OEF_difeqs
    #simcounter += 1
