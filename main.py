# import mkl
import torch
from torch_geometric.nn import GCNConv
import torch_geometric as tg
import numpy as np
from torch_geometric.data import Data


# map log, model or synchronous move to weight in the graph
def cost(action):
    if "<<" in action:
        return 1+10**-5
    return 10**-10


# Edge pairs
edges = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 3),
    (2, 3),
    (2, 4),
    (3, 5),
    (2, 5),
    (4, 5)
]

# Convert edge pairs to tensor for PyG graph
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Create edge to tuple mapping
edge_to_tuple = {
    0: ('<<', 'A'),  # edge 0-1
    1: ('A', '<<'),  # edge 0-2
    2: ('A', 'A'),   # edge 0-3
    3: ('A', '<<'),  # edge 1-3
    4: ('<<', 'A'),  # edge 2-3
    5: ('B', '<<'),  # edge 2-4
    6: ('B', '<<'),  # edge 3-5
    7: ('B', 'B'),   # edge 2-5
    8: ('<<', 'A')   # edge 4-5
}

# Edge weights (you can use any numerical data you prefer here)
edge_weight = torch.tensor([cost(edge_to_tuple[0]), cost(edge_to_tuple[1]), cost(edge_to_tuple[2]),
                            cost(edge_to_tuple[3]), cost(edge_to_tuple[4]), cost(edge_to_tuple[5]),
                            cost(edge_to_tuple[6]), cost(edge_to_tuple[7]), cost(edge_to_tuple[8])], dtype=torch.float)

# Create PyG graph
data = Data(x=torch.ones(6, 1), edge_index=edge_index, edge_attr=edge_weight)

# Reconstruct the Path: Given a path (list of edge indices), put in map to get the alignment.
path = [0, 2, 6]  # <-- example
path_tuples = [edge_to_tuple[edge_index] for edge_index in path]

# data preprocessing. #1 load dataset. #2. Inductive miner to get process model.
#                     #3 Convert to PyG friendly format. #4 Figure out how to construct on the fly.
# dataset = 'prBm6'
#
# # load dataset:
# df = load_dataset(dataset, unique_traces=True)
# df = df[['case:concept:name', 'concept:name']]
# df_no_loops = df.reset_index()
#
# # inductive miner:
# has_model = ['pr_1151_clean', 'pr_1151_noise', 'pr_1912_clean', 'pr_1912_noise', 'prAm6', 'prBm6', 'prCm6', 'prDm6', 'prEm6', 'prFm6', 'prGm6']
#     if dataset in has_model:
#         has_model_flag = True
#         df_no_loops['concept:name'] = df_no_loops['concept:name'] + '+complete'
#         net, init_marking, final_marking = load_model(dataset)
#     else:
#         has_model_flag = False
#         df_train, df_test = train_test_log_split(df_no_loops, n_traces=n_train_traces,
#                                                  random_selection=random_trace_selection,
#                                                  sample_from_each_trace_family=sample_from_each_trace_family,
#                                                  n=n_samples_from_each_family, random_seed=random_seed)
#         net, init_marking, final_marking = inductive_miner.apply(df_train)
#
#     model = from_discovered_model_to_PetriNet2(net, init_marking, final_marking, has_model_flag,
#                                                non_sync_move_penalty=non_sync_penalty, cost_function=deterministic,
#                                                conditioned_prob_compute=conditioned_prob_compute)


# simple 2-layer GCN
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        return x


# define policy network
class PolicyNetwork(torch.nn.Module):
    def __init__(self, gcn):
        super(PolicyNetwork, self).__init__()
        self.gcn = gcn
        self.action_head = torch.nn.Sequential(
            torch.nn.Linear(num_classes, num_actions),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, data):
        x = self.gcn(data)
        action_probs = self.action_head(x)
        return action_probs


# Define epsilon-greedy strategy
def select_action(policy_network, data, epsilon=0.1):
    with torch.no_grad():
        action_probs = policy_network(data)
    if np.random.uniform(0, 1) < epsilon:
        action = torch.tensor([np.random.choice(num_actions)])
    else:
        action = action_probs.argmax().unsqueeze(0)
    return action


# Modify reinforce function to handle step-by-step rewards and generate episodes on-the-fly
def reinforce(policy_network, optimizer, num_episodes, discount_factor=0.99):
    for i_episode in range(num_episodes):
        log_probs = []
        rewards = []
        state = env.reset()

        # Generate an episode
        # make range(100) = |trace|+|model|
        for t in range(100):  # assuming maximum length of episode is 100
            action = select_action(policy_network, state)
            next_state, reward, done, _ = env.step(action)
            log_prob = torch.log(policy_network(state)[action])
            log_probs.append(log_prob)
            rewards.append(reward)
            if done:
                break
            state = next_state

        # Update policy
        discounts = [discount_factor ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()


# Define your environment here
class GraphEnv:
    def __init__(self, initial_state, final_state):
        self.initial_state = initial_state
        self.current_state = initial_state
        self.final_state = final_state

    def reset(self):
        self.current_state = self.initial_state
        return self.current_state

    def step(self, action):
        """
        Apply the given action and update the current_state,
        calculate the reward and check if the episode is done.
        """
        next_state = action.endNode  # Update your state based on the action
        reward = action.weight  # Calculate the reward for the taken action
        # Check if the episode has finished
        done = False
        if self.current_state == self.final_state:
            done = True
        self.current_state = next_state

        return next_state, reward, done


# Create network and optimizer
# The number of features for each node in your graph.
# This needs to be the same as the dimension of the feature vectors that you associate with each node.
num_node_features = 10
hidden_channels = 16
num_classes = 10
num_actions = 3
initial_state = 0
final_state = 10
env = GraphEnv(initial_state, final_state)
gcn = GCN(num_node_features, hidden_channels, num_classes)
policy_network = PolicyNetwork(gcn)
optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.01)


# Train
reinforce(policy_network, optimizer, num_episodes=100)
