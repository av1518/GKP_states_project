import strawberryfields as sf
from strawberryfields.ops import *
from cutoff_opt import min_cutoff



# cutoff = min_cutoff(dB)

def target(dB):

    cutoff = min_cutoff(dB)

    prog_gkp = sf.Program(1)
    e = 10**(-dB/10)
    with prog_gkp.context as q:
        GKP(epsilon = e) | q[0]
        
    eng_gkp = sf.Engine('fock', backend_options = {'cutoff_dim': cutoff}) #
    gkp = eng_gkp.run(prog_gkp).state
    target_dm = gkp.dm()

    return target_dm, cutoff