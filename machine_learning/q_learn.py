#%%
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import operation
import tensorflow as tf
from gkp_target import psi
from qutip import wigner, Qobj, wigner_cmap
import matplotlib as mpl
from matplotlib import cm
from thewalrus.quantum import state_vector

#%%

def interferometer(params, q):
    """Parameterised interferometer acting on ``N`` modes.

    Args:
        params (list[float]): list of length ``max(1, N-1) + (N-1)*N`` parameters.

            * The first ``N(N-1)/2`` parameters correspond to the beamsplitter angles
            * The second ``N(N-1)/2`` parameters correspond to the beamsplitter phases
            * The final ``N-1`` parameters correspond to local rotation on the first N-1 modes

        q (list[RegRef]): list of Strawberry Fields quantum registers the interferometer
            is to be applied to
    """
    N = len(q)
    theta = params[:N*(N-1)//2]
    phi = params[N*(N-1)//2:N*(N-1)]
    rphi = params[-N+1:]

    if N == 1:
        # the interferometer is a single rotation
        Rgate(rphi[0]) | q[0]
        return

    n = 0  # keep track of free parameters

    # Apply the rectangular beamsplitter array
    # The array depth is N
    for l in range(N):
        for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
            # skip even or odd pairs depending on layer
            if (l + k) % 2 != 1:
                BSgate(theta[n], phi[n]) | (q1, q2)
                n += 1

    # apply the final local phase shifts to all modes except the last one
    for i in range(max(1, N - 1)):
        Rgate(rphi[i]) | q[i]

def layer(params, q):
    """
    Args:
        params (list[float]): list of length ``2*(max(1, N-1) + N**2 + n)`` containing
            the number of parameters for the layer
        q (list[RegRef]): list of Strawberry Fields quantum registers the layer
            is to be applied to
    """
    N = len(q)
    M = int(N * (N - 1)) + max(1, N - 1)
    
    a = params[:N]
    s = params[N:2*N]
    int1 = params[2*N:2*N+M]
    s2 = params[M+2*N]

    # begin layer
    for i in range(N):
        Catstate(a[i]) | q[i]
        Sgate(s[i]) | q[i]

    interferometer(int1, q)

    Sgate(s2) | q[N-1] #GKP state mode


def init_weights(modes, layers, active_sd=0.0001, passive_sd=0.1):
    """Initialize a 2D TensorFlow Variable containing normally-distributed
    random weights for an ``N`` mode quantum neural network with ``L`` layers.

    Args:
        modes (int): the number of modes in the quantum neural network
        layers (int): the number of layers in the quantum neural network
        active_sd (float): the standard deviation used when initializing
            the normally-distributed weights for the active parameters
            (displacement, squeezing, and Kerr magnitude)
        passive_sd (float): the standard deviation used when initializing
            the normally-distributed weights for the passive parameters
            (beamsplitter angles and all gate phases)

    Returns:
        tf.Variable[tf.float32]: A TensorFlow Variable of shape
        ``[layers, 2*(max(1, modes-1) + modes**2 + modes)]``, where the Lth
        row represents the layer parameters for the Lth layer.
    """
    # Number of interferometer parameters:
    M = int(modes * (modes - 1)) + max(1, modes - 1)

    # Create the TensorFlow variables
    a_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
    s_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
    int1_weights = tf.random.normal(shape=[layers, M], stddev=passive_sd)
    s2_weight = tf.random.normal(shape=[layers, 1], stddev=active_sd)

    weights = tf.concat(
        [a_weights, s_weights, int1_weights, s2_weight], axis=1
    )

    weights = tf.Variable(weights)

    return weights

#%%

# set the random seed
tf.random.set_seed(77)
np.random.seed(77)

# define width and depth of CV quantum neural network
modes = 3
layers = 1
cutoff_dim = 25

# defining desired state
target_state = psi
target_state = tf.constant(target_state, dtype=tf.complex64)

# initialize engine and program
eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
qnn = sf.Program(modes)

# initialize QNN weights
weights = init_weights(modes, layers) # our TensorFlow weights
num_params = np.prod(weights.shape)   # total number of parameters in our model

#%%

# Create array of Strawberry Fields symbolic gate arguments, matching
# the size of the weights Variable.
sf_params = np.arange(num_params).reshape(weights.shape).astype(np.str)
sf_params = np.array([qnn.params(*i) for i in sf_params])

# Construct the symbolic Strawberry Fields program by
# looping and applying layers to the program.
with qnn.context as q:
    for k in range(layers):
        layer(sf_params[k], q)

def cost(weights):
    # Create a dictionary mapping from the names of the Strawberry Fields
    # symbolic gate parameters to the TensorFlow weight values.
    mapping = {p.name: w for p, w in zip(sf_params.flatten(), tf.reshape(weights, [-1]))}

    # run the engine
    state = eng.run(qnn, args=mapping).state
    #c = state.is_pure
    #ket = state.ket()
    mu, cov = state.means(), state.cov()

    ket = state_vector(mu, cov, post_select={0: 5, 1: 7}, normalize=False, cutoff=cutoff_dim)
    p_psi = np.linalg.norm(ket)
    ket = ket / p_psi

    difference = tf.reduce_sum(tf.abs(ket - target_state))
    fidelity = tf.abs(tf.reduce_sum(tf.math.conj(ket) * target_state)) ** 2
    return difference, fidelity, ket, tf.math.real(state.trace())

# set up the optimizer
opt = tf.keras.optimizers.Adam()
cost_before, fidelity_before, _, _ = cost(weights)

fid_progress = []

# Perform the optimization
for i in range(800):

    # reset the engine if it has already been executed
    if eng.run_progs:
        eng.reset()

    with tf.GradientTape() as tape:
        loss, fid, ket, trace = cost(weights)

    fid_progress.append(fid.numpy())

    # one repetition of the optimization
    gradients = tape.gradient(loss, weights)
    opt.apply_gradients(zip([gradients], [weights]))

    # Prints progress at every rep
    if i % 1 == 0:
        print("Rep: {} Cost: {:.4f} Fidelity: {:.4f} Trace: {:.4f}".format(i, loss, fid, trace))


print("\nFidelity before optimization: ", fidelity_before.numpy())
print("Fidelity after optimization: ", fid.numpy())
print("\nTarget state: ", target_state.numpy())
print("Output state: ", np.round(ket.numpy(), decimals=3))

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.sans-serif"] = ["Computer Modern Roman"]
plt.style.use("default")

plt.figure()
plt.plot(fid_progress)
plt.ylabel("Fidelity")
plt.xlabel("Step")

grid = 800
xvec = np.linspace(-5,5, grid)
Wp = wigner(Qobj(ket.numpy()), xvec, xvec)
wmap = wigner_cmap(Wp)
sc1 = np.max(Wp)
nrm = mpl.colors.Normalize(-sc1, sc1)
fig, axes = plt.subplots(1, 1, figsize=(5, 4))
plt1 = axes.contourf(xvec, xvec, Wp, 60,  cmap=cm.RdBu, norm=nrm)
axes.contour(xvec, xvec, Wp, 60,  cmap=cm.RdBu, norm=nrm)
axes.set_title("Wigner function of the heralded state");
cb1 = fig.colorbar(plt1, ax=axes)
fig.tight_layout()
plt.show()