#%%
from ctypes import cast
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf
from qutip import wigner, Qobj, wigner_cmap, fidelity
import matplotlib as mpl
from matplotlib import cm
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def circuit(params, q):
    """
    Args:
        params (list[float]): list containing the parameters for the circuit
        q (list[RegRef]): list of Strawberry Fields quantum modes the circuit
            is to be applied to
    """
    N = len(q)
    
    a = params[0]
    s = params[1]
    theta = params[2:4]
    s2 = params[4]

    Catstate(a) | q[0]
    Catstate(a) | q[1]
    Sgate(s) | q[0]
    Sgate(s) | q[1]

    for i in range(N):
        Rgate(np.pi/2) | q[i]

    BSgate(theta[0], theta[1]) | (q[0], q[1])
    Sgate(s2) | q[1] #GKP state mode
    MeasureHomodyne(0, select = 0) | q[0]


def init_weights():
    """
    Initialize a TensorFlow Variable containing
    random weights for a 2-mode quantum circuit.
    """
    
    # Create the TensorFlow variables
    a = tf.constant(2.0, shape=[1,1])
    s1 = tf.random.uniform(shape=[1,1], minval=0, maxval=1)
    theta = tf.random.uniform(shape=[1,1], minval=0, maxval=np.pi)
    phi = tf.random.uniform(shape=[1,1], minval=0, maxval=np.pi)
    s2 = tf.random.uniform(shape=[1,1], minval=0, maxval=1)

    weights = tf.concat([a, s1, theta, phi, s2], axis=1)

    weights = tf.Variable(weights)

    return weights

# set the random seed
# tf.random.set_seed(77)
# np.random.seed(77)

# define number of modes and cutoff
modes = 2
cutoff_dim = 10

# defining target state
prog_gkp = sf.Program(1)
e = 0.1 #corresponding to delta = 10 dB
with prog_gkp.context as q:
    GKP(epsilon = e) | q[0]
    
eng_gkp = sf.Engine('fock', backend_options = {'cutoff_dim': cutoff_dim}) #
gkp = eng_gkp.run(prog_gkp).state

target_state = gkp
target_dm = gkp.dm()
target_dm_q = Qobj(target_dm)
target_dm = tf.constant(target_dm, dtype=tf.complex64)

# initialize engine and program
eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
qnn = sf.Program(modes)

# initialize weights
weights = init_weights() # our TensorFlow weights
num_params = np.prod(weights.shape)   # total number of parameters in our model

# Create array of Strawberry Fields symbolic gate arguments, matching
# the size of the weights Variable.
sf_params = np.arange(num_params).reshape(weights.shape).astype(str)
sf_params = np.array([qnn.params(*i) for i in sf_params])

# Construct the symbolic Strawberry Fields program by
# looping and applying our layer to the program.
with qnn.context as q:
    circuit(sf_params[0], q)

# mapping = {p.name: w for p, w in zip(sf_params.flatten(), tf.reshape(weights, [-1]))}
# state = eng.run(qnn, args=mapping).state
# dm = state.reduced_dm(1)
# dm_q = Qobj(dm.numpy())

# fid = fidelity(dm_q, target_dm_q)
# a = tf.math.multiply(target_dm,tf.math.sqrt(dm))
# b = tf.math.multiply(tf.math.sqrt(dm),a)
# c = tf.math.sqrt(b)
# fid2 = tf.math.real(tf.linalg.trace(c))
# cos = tf.math.subtract(1,fid2)
# trace_dist = 0.5*tf.linalg.trace(tf.abs(dm - target_dm))

#%%

def cost(weights):
    # Create a dictionary mapping from the names of the Strawberry Fields
    # symbolic gate parameters to the TensorFlow weight values.
    mapping = {p.name: w for p, w in zip(sf_params.flatten(), tf.reshape(weights, [-1]))}
    state = eng.run(qnn, args=mapping).state
    dm = state.reduced_dm(1)
    dm_q = Qobj(dm.numpy())

    fid = fidelity(dm_q, target_dm_q)
    #cos = tf.math.subtract(1.0,tf.math.real(tf.linalg.trace(tf.math.sqrt\
        #(tf.math.multiply(tf.math.sqrt(dm),tf.math.multiply(target_dm,tf.math.sqrt(dm)))))))
    trace_dist = 0.5*tf.linalg.trace(tf.abs(dm - target_dm))

    return trace_dist, fid, dm, tf.math.real(state.trace())

# set up the optimizer
rate = 0.001
opt = tf.keras.optimizers.Adam(learning_rate = rate)
cost_before, fidelity_before, _, _ = cost(weights)

fid_progress = []
cost_progress = []

# Perform the optimization
for i in range(1600):

    # reset the engine if it has already been executed
    if eng.run_progs:
        eng.reset()

    with tf.GradientTape() as tape:
        loss, fid, red_dm, trace = cost(weights)

    fid_progress.append(fid)
    cost_progress.append(loss)

    # print(type(weights))
    # one repetition of the optimization
    gradients = tape.gradient(loss, weights)
    # print(gradients)
    print(weights)
    opt.apply_gradients(zip([gradients], [weights]))

    # Prints progress at every rep
    if i % 1 == 0:
        print("Rep: {} Cost: {:.4f} Fidelity: {:.4f} Trace: {:.4f}".format(i, loss, fid, trace))


print("\nFidelity before optimization: ", fidelity_before)
print("Fidelity after optimization: ", fid)
print("\nTarget density matrix: ", target_dm)
print("Output reduced density matrix: ", np.round(red_dm, decimals=3))
print("\nCircuit parameters: ", weights)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.sans-serif"] = ["Computer Modern Roman"]
plt.style.use("default")

plt.figure()
plt.plot(fid_progress, label  = 'Fidelity Progress')
plt.plot(cost_progress, label = 'Cost Progress')
plt.ylabel("Fidelity")
plt.xlabel("Step")
plt.legend()

grid = 800
xvec = np.linspace(-5,5, grid)
Wp = wigner(Qobj(red_dm.numpy()), xvec, xvec)
wmap = wigner_cmap(Wp)
sc1 = np.max(Wp)
nrm = mpl.colors.Normalize(-sc1, sc1)
fig, axes = plt.subplots(1, 1, figsize=(5, 4))
plt1 = axes.contourf(xvec, xvec, Wp, 60,  cmap=cm.RdBu, norm=nrm)
axes.contour(xvec, xvec, Wp, 60,  cmap=cm.RdBu, norm=nrm)
axes.set_title("Wigner function of the learnt state")
cb1 = fig.colorbar(plt1, ax=axes)
fig.tight_layout()

Wp2 = wigner(Qobj(target_dm.numpy()), xvec, xvec)
wmap2 = wigner_cmap(Wp2)
sc12 = np.max(Wp2)
nrm2 = mpl.colors.Normalize(-sc12, sc12)
fig2, axes2 = plt.subplots(1, 1, figsize=(5, 4))
plt2 = axes2.contourf(xvec, xvec, Wp2, 60,  cmap=cm.RdBu, norm=nrm2)
axes2.contour(xvec, xvec, Wp2, 60,  cmap=cm.RdBu, norm=nrm2)
axes2.set_title("Wigner function of the target state")
cb12 = fig2.colorbar(plt2, ax=axes2)
fig2.tight_layout()

#%% Plot cut of Wigner function at p = 0

plt.figure()
plt.title('Wigner cut of learnt state')
plt.plot(xvec, Wp[grid//2,:], label = f'Fidelity:{fid}, learning_rate = {rate}')
plt.ylabel(r"W(q,0)")
plt.xlabel(r"q")

plt.figure()
plt.title('Wigner cut of target state')
plt.plot(xvec, Wp2[grid//2,:])
plt.xlabel(r"q")
plt.ylabel(r"W(q,0)")
plt.legend()
plt.show()
# %%
