ROWS = 10
COLS = 10

EPISODES = 5000
MAX_STEPS = 300

LEARNING_RATE = 0.001
GAMMA = 0.99

BUFFER_SIZE = 50000
BATCH_SIZE = 64

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

TARGET_UPDATE = 1000

HIDDEN_SIZE = 128

MODEL_PATH = "models/final_model.pth"

# Curriculum learning: train on a small, fixed pool of warehouse
# maps before asking the agent to generalize to fully random maps.
# This isolates "is the DQN implementation broken" from "is the
# task (arbitrary map generalization) too hard right now."
#
#   1    -> train on a single fixed map (sanity check the algorithm
#           can solve navigation at all)
#   5    -> a small pool, next step up in difficulty
#   20   -> a larger pool
#   None -> fully random map every episode (the original setting)
NUM_FIXED_MAPS = 1