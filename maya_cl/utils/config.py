# config.py — Maya-CL hyperparameters
# Shared across Paper 3 (TIL) and Paper 4 (CIL).

# ── Reproducibility ───────────────────────────────────────────────
SEED = 42

# ── SNN Temporal ──────────────────────────────────────────────────
T_STEPS = 4

# ── Network Architecture ──────────────────────────────────────────
CONV1_CHANNELS = 32
CONV2_CHANNELS = 64
FC1_SIZE = 2048
NUM_CLASSES = 10

# ── LIF Neuron Parameters ─────────────────────────────────────────
TAU_MEMBRANE = 2.0
V_THRESHOLD = 1.0
V_RESET = 0.0

# ── Affective State Time Constants ────────────────────────────────
TAU_SHRADDHA = 10.0    # Trust — slow integrator
TAU_BHAYA = 3.0        # Fear — fast responder
TAU_VAIRAGYA = 20.0    # Wisdom — slowest dimension
TAU_SPANDA = 5.0       # Aliveness — mid-rate

# ── Hebbian Learning ──────────────────────────────────────────────
HEBBIAN_LR = 0.01

# ── Lability ──────────────────────────────────────────────────────
LABILITY_INIT = 1.0
LABILITY_PAIN_BOOST = 5.0
LABILITY_DECAY_RATE = 0.95
PAIN_CONFIDENCE_THRESHOLD = 0.25

# ── Vairagya ──────────────────────────────────────────────────────
VAIRAGYA_DECAY_RATE = 0.002
VAIRAGYA_PROTECTION_THRESHOLD = 0.3
VAIRAGYA_ACCUMULATE_RATE = 0.003

# ── Benchmark ─────────────────────────────────────────────────────
NUM_TASKS = 5
CLASSES_PER_TASK = 2
BATCH_SIZE = 128
EPOCHS_PER_TASK = 20

# ── Paths ─────────────────────────────────────────────────────────
DATA_DIR = "data/"
RESULTS_DIR = "results/"

# ── Paper 4: CIL + Affective-Gated Replay ─────────────────────────
REPLAY_BUFFER_SIZE = 50
REPLAY_RATIO = 0.3
REPLAY_VAIRAGYA_PARTIAL_LIFT = 0.8
REPLAY_PAIN_EXEMPT = True
CIL_BOUNDARY_DECAY = 0.60
CIL_MAX_VFOUT_PROTECTION = 0.70

# ── Paper 4: Buddhi (Discriminative Intellect) ────────────────────
# Buddhi = experience × (1 - Bhaya)
# Collapses at task boundaries (no experience, high fear) — Viparita state.
# Recovers as the network accumulates batches and Bhaya settles.
# Gates Vairagya erosion: erosion is heavy when Buddhi is low,
# approaches zero as Buddhi rises.
TAU_BUDDHI = 200.0              # experience accumulation rate (batches)
VAIRAGYA_PAIN_EROSION_RATE = 0.005  # max erosion per batch in Viparita state