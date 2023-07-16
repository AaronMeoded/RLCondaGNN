import torch
from torch_geometric.nn import GCNConv
import torch_geometric as tg
import numpy as np

# Preprocess your graph data into PyTorch Geometric format
# This is a place holder and you will need to replace it with your own data
data = tg.data.Data(x=torch.randn(100, num_node_features), edge_index=torch.randint(100, (2, 500)))

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
env = ...

# Create network and optimizer
# The number of features for each node in your graph.
# This needs to be the same as the dimension of the feature vectors that you associate with each node.
num_node_features = 10
hidden_channels = 16
num_classes = 10
num_actions = 3
gcn = GCN(num_node_features, hidden_channels, num_classes)
policy_network = PolicyNetwork(gcn)
optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.01)


# Train
reinforce(policy_network, optimizer, num_episodes=100)




# import torch
# from torch_geometric.nn import GCNConv
#

# # implement reinforce
# def reinforce(policy_network, optimizer, episodes):
#     for episode in episodes:
#         total_reward = 0
#         log_probs = []
#
#         # Collect trajectory
#         for step in episode:
#             data, action, reward = step
#             action_probs = policy_network(data)
#             log_prob = torch.log(action_probs[action])
#             log_probs.append(log_prob)
#             total_reward += reward
#
#         # Calculate loss
#         loss = (-total_reward) * torch.cat(log_probs).sum()
#
#         # Update weights
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()import torch
# from torch_geometric.nn import GCNConv
#
# # simple 2-layer GCN
# class GCN(torch.nn.Module):
#     def __init__(self, num_node_features, hidden_channels, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(num_node_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, num_classes)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = self.conv2(x, edge_index)
#
#         return x
#
#
# # define policy network
# class PolicyNetwork(torch.nn.Module):
#     def __init__(self, gcn):
#         super(PolicyNetwork, self).__init__()
#         self.gcn = gcn
#         self.action_head = torch.nn.Sequential(
#             torch.nn.Linear(num_classes, num_actions),
#             torch.nn.Softmax(dim=-1)
#         )
#
#     def forward(self, data):
#         x = self.gcn(data)
#         action_probs = self.action_head(x)
#         return action_probs
#
#
# # implement reinforce
# def reinforce(policy_network, optimizer, episodes):
#     for episode in episodes:
#         total_reward = 0
#         log_probs = []
#
#         # Collect trajectory
#         for step in episode:
#             data, action, reward = step
#             action_probs = policy_network(data)
#             log_prob = torch.log(action_probs[action])
#             log_probs.append(log_prob)
#             total_reward += reward
#
#         # Calculate loss
#         loss = (-total_reward) * torch.cat(log_probs).sum()
#
#         # Update weights
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#
# # train model
#
# # Create network and optimizer
# # The number of features for each node in your graph.
# # This needs to be the same as the dimension of the feature vectors that you associate with each node.
# num_node_features = 10
# hidden_channels = 16
# num_classes = 10
# num_actions = 3
# gcn = GCN(num_node_features, hidden_channels, num_classes)
# policy_network = PolicyNetwork(gcn)
# optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.01)
#
# # Generate your episodes here
# episodes = ...
#
#
# # Train
# reinforce(policy_network, optimizer, episodes)
#
#
# # train model
#
# # Create network and optimizer
# # The number of features for each node in your graph.
# # This needs to be the same as the dimension of the feature vectors that you associate with each node.
# num_node_features = 10
# hidden_channels = 16
# num_classes = 10
# num_actions = 3
# gcn = GCN(num_node_features, hidden_channels, num_classes)
# policy_network = PolicyNetwork(gcn)
# optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.01)
#
# # Generate your episodes here
# episodes = ...
#
#
# # Train
# reinforce(policy_network, optimizer, episodes)