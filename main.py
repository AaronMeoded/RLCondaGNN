import torch
from torch_geometric.nn import GCNConv
import torch_geometric as tg
import numpy as np
from torch_geometric.data import Data


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
    def __init__(self, gcn, num_actions):
        super(PolicyNetwork, self).__init__()
        self.gcn = gcn
        self.action_head = torch.nn.Linear(num_classes, num_actions)
        # self.action_head = torch.nn.Sequential(
        #     torch.nn.Linear(num_classes, num_actions),
        #     torch.nn.Softmax(dim=-1)
        # )

    def forward(self, state, valid_moves):
        x = self.gcn(data)[state]
        action_probs = self.action_head(x)[valid_moves]
        # action_probs = action_probs / action_probs.sum()  # Renormalize so the probabilities sum to 1
        return action_probs


# Define epsilon-greedy strategy
def select_action(policy_network, state, valid_moves, epsilon=0.1):
    # print(valid_moves)
    with torch.no_grad():
        action_probs = policy_network(state, valid_moves)
    action_probs = action_probs / action_probs.sum()  # Renormalize so the probabilities sum to 1
    if torch.rand(1).item() < epsilon:
        action = np.random.choice(valid_moves)  # change here: select from the range of valid_moves
        # action = valid_moves[torch.randint(len(valid_moves), (1,)).item()]
    else:
        action = valid_moves[action_probs.argmax().item()]
    return action


# Modify reinforce function to handle step-by-step rewards and generate episodes on-the-fly
def reinforce(policy_network, optimizer, num_episodes, discount_factor=0.99):
    for i_episode in range(num_episodes):
        log_probs = []
        rewards = []
        state = env.reset()
        # valid_moves = env.outgoing_edges()
        # print(valid_moves)

        # Generate an episode
        # make range(100) = |trace|+|model|
        for t in range(100):  # assuming maximum length of episode is 100
            valid_moves = env.outgoing_edges()
            action = select_action(policy_network, state, valid_moves)  # action means edge to take next
            # print('action: ')
            # print(action)
            # print('valid_moves: ')
            # print(valid_moves)
            # print('PN: ')
            # print(policy_network(state, valid_moves)[valid_moves.index(action)])
            # print(torch.log(policy_network(state, valid_moves)[valid_moves.index(action)]))
            log_prob = torch.log(policy_network(state, valid_moves)[valid_moves.index(action)])
            # log_prob = torch.log(policy_network(state, valid_moves)[action].unsqueeze(0))
            log_probs.append(log_prob)
            next_state, reward, done = env.step(action)
            state = next_state
            # print('new state: ')
            # print(state)
            rewards.append(reward)
            if done:
                break

        # Update policy

        # discounts = [discount_factor ** i for i in range(len(rewards))]
        # R = sum([a * b for a, b in zip(discounts, rewards)])
        # policy_loss = []
        # for log_prob in log_probs:
        #     policy_loss.append(-log_prob * R)
        # policy_loss = torch.stack(policy_loss).sum()

        # Calculate returns from each time step
        returns = []
        G = 0
        for reward in rewards[::-1]:
            G = reward + discount_factor * G
            returns.insert(0, G)

        # Normalize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Calculate policy loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.stack(policy_loss).sum()

        print(f"Summed Rewards Episode #{i_episode}: {sum(rewards)}")
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

    # Zero-epsilon Run
    rewards = []
    state = env.reset()
    for t in range(100):  # assuming maximum length of episode is 100
        valid_moves = env.outgoing_edges()
        action = select_action(policy_network, state, valid_moves, epsilon=0)  # action means edge to take next
        log_prob = torch.log(policy_network(state, valid_moves)[valid_moves.index(action)])
        # log_prob = torch.log(policy_network(state, valid_moves)[action].unsqueeze(0))
        log_probs.append(log_prob)
        next_state, reward, done = env.step(action)
        state = next_state
        # print('new state: ')
        # print(state)
        rewards.append(reward)
        if done:
            break
    print(f"Zero-epsilon Summed Rewards Episode #{i_episode}: {sum(rewards)}")


# Define your environment here
class GraphEnv:
    def __init__(self, data, initial_state, final_state):
        self.data = data
        self.initial_state = initial_state
        self.current_state = initial_state
        self.final_state = final_state

    def reset(self):
        self.current_state = self.initial_state
        return self.current_state

    def outgoing_edges(self):
        # Find the indices where the source node is the given node, return as list.
        edge_indices_source = (self.data.edge_index[0] == self.current_state).nonzero(as_tuple=True)[0]
        outgoing_edge_indices = edge_indices_source.tolist()
        return outgoing_edge_indices

    def step(self, action):
        """
        Apply the given action and update the current_state,
        calculate the reward and check if the episode is done.
        """
        next_state = self.data.edge_index[1][action].item()  # Update your state based on the action
        reward = self.data.edge_attr[action].item()  # Calculate the reward for the taken action
        # Check if the episode has finished
        done = False
        self.current_state = next_state
        if self.current_state == self.final_state:
            done = True

        return next_state, reward, done


# map log, model or synchronous move to weight in the graph
def cost(action):
    if "<<" in action:
        # return (1+10**-5)*-1
        return -1
    # return (10**-10)*-1
    return 2


# Example of a graph:
# Edge pairs
edges = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]

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


# Create network and optimizer
# The number of features for each node in your graph.
# This needs to be the same as the dimension of the feature vectors that you associate with each node.
num_node_features = 1
hidden_channels = 16
num_classes = 16
initial_state = 0
final_state = 5

env = GraphEnv(data, initial_state, final_state)
gcn = GCN(num_node_features, hidden_channels, num_classes)
num_actions = data.edge_index.shape[1]  # Number of edges in the graph
policy_network = PolicyNetwork(gcn, num_actions)
optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.03)


# Train
reinforce(policy_network, optimizer, num_episodes=500)
