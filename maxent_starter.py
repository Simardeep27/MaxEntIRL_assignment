import numpy as np
import numpy.random as rand
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import os
import csv

        
def build_trans_mat_gridworld():
  # 5x5 gridworld laid out like:
  # 0  1  2  3  4
  # 5  6  7  8  9 
  # 9  10 11 12 13
  # 14 15 16 17 18
  # 20 21 22 23 24
  # where 24 is a goal state that always transitions to a 
  # special zero-reward terminal state (25) with no available actions
  trans_mat = np.zeros((26,4,26))
  
  # NOTE: the following iterations only happen for states 0-23.
  # This means terminal state 25 has zero probability to transition to any state, 
  # even itself, making it terminal, and state 24 is handled specially below.
  
  # Action 0 = down
  for s in range(24):
    if s < 20:
      trans_mat[s,0,s+5] = 1
    else:
      trans_mat[s,0,s] = 1
      
  # Action 1 = up
  for s in range(24):
    if s >= 5:
      trans_mat[s,1,s-5] = 1
    else:
      trans_mat[s,1,s] = 1
      
  # Action 2 = left
  for s in range(24):
    if s%5 > 0:
      trans_mat[s,2,s-1] = 1
    else:
      trans_mat[s,2,s] = 1
      
 # Action 3 = right
  for s in range(24):
    if s%5 < 4:
      trans_mat[s,3,s+1] = 1
    else:
      trans_mat[s,3,s] = 1

  # Finally, goal state always goes to zero reward terminal state
  for a in range(4):
    trans_mat[24,a,25] = 1  
      
  return trans_mat


def build_state_features_gridworld():
  # There are 4 features and only one is active at any given state, represented 1-hot vector at each state, with the layout as follows:
  # 0 0 0 0 0
  # 0 1 1 1 1
  # 0 0 2 0 0
  # 0 0 0 0 0 
  # 0 0 0 0 4
  # And the special terminal state (25) has all zero state features.

  sf = np.zeros((26,4))  
  sf[0,0] = 1
  sf[1,0] = 1
  sf[2,0] = 1
  sf[3,0] = 1
  sf[4,0] = 1
  sf[5,0] = 1
  sf[6,1] = 1
  sf[7,1] = 1
  sf[8,1] = 1
  sf[9,1] = 1
  sf[10,0] = 1
  sf[11,0] = 1
  sf[12,2] = 1
  sf[13,0] = 1
  sf[14,0] = 1
  sf[15,0] = 1
  sf[16,0] = 1
  sf[17,0] = 1
  sf[18,0] = 1
  sf[19,0] = 1
  sf[20,0] = 1
  sf[21,0] = 1
  sf[22,0] = 1
  sf[23,0] = 1
  sf[24,3] = 1
  return sf


           
def calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features, term_index):
  """
  Implement steps 1-3 of Algorithm 1 in Ziebart et al.
  
  For a given reward function and horizon, calculate the MaxEnt policy that gives equal weight to equal reward trajectories
  
  trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  r_weights: a size F array of the weights of the current reward function to evaluate
  state_features: an S x F array that lists F feature values for each state in S
  term_index: the index of the special terminal state
  
  return: an S x A policy in which each entry is the probability of taking action a in state s
  """
  n_states = trans_mat.shape[0]
  n_actions = trans_mat.shape[1]

  rewards = np.dot(state_features, r_weights)
  exp_rewards = np.exp(rewards)

  z_s = np.zeros(n_states)
  z_s[term_index] = 1.0
  z_a = np.zeros((n_states, n_actions))

  for _ in range(horizon):
    z_a = np.tensordot(trans_mat, exp_rewards * z_s, axes=([2], [0]))
    z_s = np.sum(z_a, axis=1)
    z_s[term_index] = 1.0

  policy = np.zeros((n_states, n_actions))
  nonzero = z_s > 0
  policy[nonzero, :] = z_a[nonzero, :] / z_s[nonzero, None]
  policy[term_index, :] = 0.0
  return policy


  
def calcExpectedStateFreq(trans_mat, horizon, start_dist, policy):
  """
  Implement steps 4-6 of Algorithm 1 in Ziebart et al.
  
  Given a MaxEnt policy, begin with the start state distribution and propagate forward to find the expected state frequencies over the horizon
  
  trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  start_dist: a size S array of starting start probabilities - must sum to 1
  policy: an S x A array array of probabilities of taking action a when in state s
  
  return: a size S array of expected state visitation frequencies
  """
  
  n_states = np.shape(trans_mat)[0]
  n_actions = np.shape(trans_mat)[1]

  if horizon <= 0:
    return np.zeros(n_states)

  d_st = np.zeros((n_states, horizon))
  d_st[:,0] = start_dist

  for t in range(horizon-1):
    sa_dist = d_st[:,t][:,None] * policy
    d_st[:,t+1] = np.tensordot(sa_dist, trans_mat, axes=([0,1], [0,1]))

  state_freq = np.sum(d_st, axis=1)
  return state_freq
  


def maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate, term_index, return_gaps=False):
  """
  Implement the outer loop of MaxEnt IRL that takes gradient steps in weight space
  
  Compute a MaxEnt reward function from demonstration trajectories
  
  trans_mat: an S x A x S' array that describes transition probabilities from state s to s' if action a is taken
  state_features: an S x F array that lists F feature values for each state in S
  demos: a list of lists containing D demos of varying lengths, where each demo is series of states (ints)
  seed_weights: a size F array of starting reward weights
  n_epochs: how many times (int) to perform gradient descent steps
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  learning_rate: a multiplicative factor (float) that determines gradient step size
  term_index: the index of the special terminal state
  
  return: a size F array of reward weights
  """
  
  n_states = np.shape(state_features)[0]
  n_features = np.shape(state_features)[1]
  n_demos = len(demos)

  r_weights = np.array(seed_weights, dtype=float).copy()

  feature_exp_demo = np.zeros(n_features)
  start_dist = np.zeros(n_states)
  for demo in demos:
    if len(demo) == 0:
      continue
    start_dist[demo[0]] += 1.0
    for s in demo:
      feature_exp_demo += state_features[s]

  feature_exp_demo /= float(n_demos)
  start_dist /= float(n_demos)

  gaps = []
  for epoch in range(n_epochs):
    policy = calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features, term_index)
    state_freq = calcExpectedStateFreq(trans_mat, horizon, start_dist, policy)
    feature_exp_model = np.dot(state_features.T, state_freq)
    grad = feature_exp_demo - feature_exp_model
    gaps.append(np.linalg.norm(grad))

    r_weights += learning_rate * grad

  if return_gaps:
    return r_weights, np.array(gaps)
  return r_weights


def build_reward_grid(r_weights, state_features):
  reward_fxn = []
  for s_i in range(25):
    reward_fxn.append(np.dot(r_weights, state_features[s_i]))
  return np.reshape(reward_fxn, (5, 5))


def save_reward_plot(reward_fxn, title, output_path):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  x = np.arange(0, 5, 1)
  y = np.arange(0, 5, 1)
  x, y = np.meshgrid(x, y)
  ax.plot_surface(x, y, reward_fxn, rstride=1, cstride=1, cmap=cm.coolwarm,
                  linewidth=0, antialiased=False)
  ax.set_title(title)
  plt.tight_layout()
  plt.savefig(output_path)
  plt.close(fig)


def run_learning_rate_experiment(trans_mat, state_features, demos, learning_rates, seed_weights,
                                 n_epochs, horizon, term_index, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  results = []

  for lr in learning_rates:
    r_weights, gaps = maxEntIRL(trans_mat,state_features,demos,seed_weights,n_epochs,horizon,lr,term_index,return_gaps=True)

    reward_fxn = build_reward_grid(r_weights, state_features)
    lr_tag = str(lr).replace('.', 'p')
    plot_path = os.path.join(output_dir, f"reward_lr_{lr_tag}.png")
    gap_path = os.path.join(output_dir, f"gaps_lr_{lr_tag}.npy")

    save_reward_plot(reward_fxn, f"Reward Surface (lr={lr})", plot_path)
    np.save(gap_path, gaps)

    max_abs_reward = float(np.max(np.abs(reward_fxn)))
    reward_range = float(np.max(reward_fxn) - np.min(reward_fxn))
    weight_l2 = float(np.linalg.norm(r_weights))
    max_abs_weight = float(np.max(np.abs(r_weights)))

    result = {
      "learning_rate": lr,
      "final_gap": float(gaps[-1]),
      "min_gap": float(np.min(gaps)),
      "max_abs_reward": max_abs_reward,
      "reward_range": reward_range,
      "weight_l2": weight_l2,
      "max_abs_weight": max_abs_weight,
      "plot_path": plot_path,
      "gap_path": gap_path,
    }
    results.append(result)
    print(
      f"lr={lr:<6} final_gap={result['final_gap']:.6f} "
      f"max|R|={max_abs_reward:.4f} |w|_2={weight_l2:.4f} plot={plot_path}"
    )

  summary_path = os.path.join(output_dir, "learning_rate_summary.csv")
  with open(summary_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(
      csvfile,
      fieldnames=[
        "learning_rate",
        "final_gap",
        "min_gap",
        "max_abs_reward",
        "reward_range",
        "weight_l2",
        "max_abs_weight",
        "plot_path",
        "gap_path",
      ],
    )
    writer.writeheader()
    writer.writerows(results)

  return results, summary_path
  
 
 
if __name__ == '__main__':
  
  # Build domain, features, and demos
  trans_mat = build_trans_mat_gridworld()
  state_features = build_state_features_gridworld() 
  demos = [[4,9,14,19,24,25],[3,8,13,18,19,24,25],[2,1,0,5,10,15,20,21,22,23,24,25],[1,0,5,10,11,16,17,22,23,24,25]]
  seed_weights = np.zeros(4)
  term_index = 25
  
  # Parameters
  n_epochs = 10
  horizon = 15
  # learning_rate_experiment = [0.001, 0.005, 0.1, 0.3, 0.5, 0.8, 1.0, 2.0]
  # output_dir = os.path.join(os.path.dirname(__file__), "learning_rate_outputs")

  # Q1
  # results, summary_path = run_learning_rate_experiment(trans_mat,state_features,demos,learning_rate_experiment,seed_weights,n_epochs,horizon,term_index,output_dir)

  # best = min(results, key=lambda r: r["final_gap"])
  # print(f"\nBest learning rate by final gap: {best['learning_rate']} (gap={best['final_gap']:.6f})")
  # print(f"Summary CSV: {summary_path}")

  learning_rate = 1.0
  
  # Main algorithm call
  r_weights = maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate, term_index)
  
  # Construct reward function from weights and state features
  reward_fxn = []
  for s_i in range(25):
    reward_fxn.append( np.dot(r_weights, state_features[s_i]) )
  reward_fxn = np.reshape(reward_fxn, (5,5))
  
  # Plot reward function
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  X = np.arange(0, 5, 1)
  Y = np.arange(0, 5, 1)
  X, Y = np.meshgrid(X, Y)
  surf = ax.plot_surface(X, Y, reward_fxn, rstride=1, cstride=1, cmap=cm.coolwarm,
			linewidth=0, antialiased=False)
  plt.show()
