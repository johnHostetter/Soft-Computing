import gym
import torch

from torch import optim
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fuzzy.common.flc import FLC as torchFLC
from fuzzy.self_adaptive.cart_pole import evaluate_on_environment


class mimoFLC:
    def __init__(self, n_inputs, n_outputs, antecedents, rules):
        self.flcs = []
        self.optimizers = []
        for flc_idx in range(n_outputs):
            flc = torchFLC(n_inputs, 1, antecedents, rules)
            self.flcs.append(flc)
            # https://stackoverflow.com/questions/42966393/is-it-good-learning-rate-for-adam-method
            self.optimizers.append(optim.Adam(flc.parameters(), lr=3e-4))

    def predict(self, x):
        output = []
        for flc in self.flcs:
            output.append(list(flc.predict(x).detach().numpy()))
        return torch.tensor(output).T

    def train(self, mode):
        for flc in self.flcs:
            flc.train(mode)

    def zero_grad(self):
        for flc in self.flcs:
            flc.zero_grad()

    def offline_update(self, states, target_q_values, action_indices):
        avg_loss = 0.
        all_q_values = self.predict(states)
        for flc_idx, flc in enumerate(self.flcs):
            # Compute the loss and its gradients
            q_values = flc.predict(states)
            # print(torch.argmax(q_values, dim=1))
            loss = loss_fn(all_q_values, q_values, target_q_values, action_indices, flc_idx)
            loss.backward()

            # Adjust learning weights
            self.optimizers[flc_idx].step()
            avg_loss += loss.item()
        return avg_loss / len(self.flcs)


def loss_fn(all_q_values, pred_qvalues, target_qvalues, action_indices, flc_idx):
    cql_alpha = 0.5
    logsumexp_qvalues = torch.logsumexp(all_q_values, dim=-1)

    tmp_pred_qvalues = all_q_values.gather(
        1, action_indices.reshape(-1, 1)).squeeze()
    cql_loss = logsumexp_qvalues - tmp_pred_qvalues

    new_targets = target_qvalues[:, flc_idx]
    # new_targets = target_qvalues.gather(1, action_indices.reshape(-1, 1)).squeeze()
    loss = torch.mean((pred_qvalues - new_targets) ** 2)
    loss = loss + cql_alpha * torch.mean(cql_loss)
    return loss


def make_target_q_values(model, transition, gamma):
    q_values = model.predict(transition['state']).float()
    q_values_next = model.predict(transition['next state'])
    # to index with action_indices, we need to convert to long data type
    action_indices = transition['action'].long()
    rewards = transition['reward'].float()
    terminals = transition['terminal']
    # detach first and then clone is slightly more efficient than clone first and then detach
    # https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
    q_values[torch.arange(len(q_values)), action_indices.long()] = rewards.detach().clone()
    max_q_values_next, max_q_values_next_indices = torch.max(q_values_next, dim=1)
    max_q_values_next *= gamma
    q_values[torch.arange(len(q_values)), action_indices] += torch.tensor(
        [0.0 if done else float(max_q_values_next[idx]) for idx, done in enumerate(terminals)])
    return transition['state'], q_values, action_indices


### OLD
import numpy as np
def extract_data_from_trajectories(model, batch, gamma):
    states = batch[0].detach().numpy()
    next_states = batch[3].detach().numpy()
    action_indices = batch[1].detach().numpy()
    q_values = model.predict(torch.Tensor(states))
    q_values_next = model.predict(torch.Tensor(next_states))
    rewards = batch[2].detach().numpy().astype('float32')
    dones = batch[4].detach().numpy()
    q_values[torch.arange(len(q_values)), action_indices] = torch.tensor(rewards).double()
    max_q_values_next, max_q_values_next_indices = torch.max(q_values_next, dim=1)
    max_q_values_next *= gamma
    q_values[torch.arange(len(q_values)), action_indices] += torch.tensor(
        [0.0 if done else float(max_q_values_next[idx]) for idx, done in enumerate(dones)])
    targets = q_values.detach().numpy()

    return states, targets, action_indices

# END OF OLD


def train_one_epoch(training_loader, optimizer, model, epoch_index, tb_writer, gamma):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, transition in enumerate(training_loader):
        # Every transition is {'state', 'action', 'reward', 'next state', 'terminal'}

        # Zero your gradients for every batch!
        model.zero_grad()
        # optimizer.zero_grad()

        # old_states, old_targets, old_action_indices = extract_data_from_trajectories(model, np.array(list(transition.values())), gamma)
        states, target_q_values, action_indices = make_target_q_values(model, transition, gamma)
        # print(torch.max(target_q_values).item())

        # print(torch.max(states - old_states).item())
        # print(torch.max(target_q_values - torch.Tensor(old_targets)).item())
        # print(torch.max(action_indices - old_action_indices).item())

        # for param in model.parameters():
        # #     print(param)
        # #     print(param.requires_grad)
        #     param.requires_grad = False

        loss = model.offline_update(transition['state'], target_q_values, action_indices)

        # original update
        # # Compute the loss and its gradients
        # q_values = model.predict(transition['state'])
        # # print(torch.argmax(q_values, dim=1))
        # loss = loss_fn(q_values, target_q_values, action_indices)
        # loss.backward()
        #
        # # Adjust learning weights
        # optimizer.step()

        # Gather data and report
        running_loss += loss
        # running_loss += loss.item()
        if i % 10 == 0:
            last_loss = running_loss / 10  # loss per batch
            # print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def offline_q_learning(model, training_dataset, validation_dataset, max_epochs, batch_size, gamma=0.9):
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    env = gym.make('CartPole-v1')

    EPOCHS = 100
    optimizer = None
    # optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)
    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(train_loader, optimizer, model, epoch_number, writer, gamma)

        # We don't need gradients on to do reporting
        model.train(False)
        # print(torch.argmax(model.consequences, dim=1))

        running_vloss = 0.0
        for i, validation_transition in enumerate(val_loader):
            _, target_q_values, action_indices = make_target_q_values(model, validation_transition, gamma)
            vloss = 0.
            # vloss = loss_fn(model.predict(validation_transition['state']), target_q_values, action_indices)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            # torch.save(model.state_dict(), model_path)

        if epoch % 10 == 0:
            avg_score, std_score, curr_rules_during_end = evaluate_on_environment(env)(model)
            print((avg_score, std_score))

        epoch_number += 1
    return model, [avg_loss], [avg_vloss]
