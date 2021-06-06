import wandb
import torch
import horovod.torch as hvd


class NetworkTrainer:
    """# TODO: Docstring"""

    def __init__(self, nn_model, train_set_loader, val_set_loader, config):
        """# TODO: Docstring"""
        self.nn_model = nn_model
        self.train_set_loader = train_set_loader
        self.val_set_loader = val_set_loader
        self.loss_function = config.loss_function
        self.optimizer = config.optimizer
        self.config = config

    def start_training(self):
        """# TODO: Docstring"""

        for epoch in range(1, self.config.num_epochs + 1):
            for batch_idx, (model, label) in enumerate(self.train_set_loader):

                # Make sure network is in training mode
                self.nn_model.train()

                # Copy to GPU if available
                if self.config.device.type == 'cuda':
                    model, label = model.cuda(), label.cuda()

                # Set gradients to zero
                self.optimizer.zero_grad()

                # Calculate output
                output = self.nn_model(model)
                output = torch.reshape(output, (-1, 1))

                # Calculate loss
                loss = self.loss_function(output, label)

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                if batch_idx % self.config.plot_frequency == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(model), len(self.config.train_sampler),
                               100. * batch_idx / len(self.train_set_loader), loss.item()))

            self.validate()

    def metric_average(self, val, name):
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    def validate(self):

        self.nn_model.eval()
        test_loss = 0.
        test_accuracy = 0.

        for model, label in self.val_set_loader:

            # Copy to GPU if available
            if self.config.device.type == 'cuda':
                model, label = model.cuda(), label.cuda()

            # Calculate output
            output = self.nn_model(model)
            output = torch.reshape(output, (-1, 1))

            # Sum up batch loss
            test_loss += self.loss_function(output, label).item()
            test_accuracy += (output > 0.5).float().sum()

        test_loss /= len(self.config.validation_sampler)
        test_accuracy /= len(self.config.validation_sampler)

        # Horovod: average metric values across workers.
        test_loss = self.metric_average(test_loss, 'avg_loss')
        test_accuracy = self.metric_average(test_accuracy, 'avg_accuracy')

        # Horovod: print output only on first rank.
        if hvd.rank() == 0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
                test_loss, 100. * test_accuracy))




    def training_log(self, loss, mini_batch, train=True):
        """# TODO: Docstring"""
        if train:
            wandb.log({'batch': mini_batch, 'loss': loss})
        else:
            wandb.log({'batch': mini_batch, 'loss': loss})

    def validation_phase(self, nn_model, val_set_loader, loss_function, epoch):
        """# TODO: Docstring"""

        print(f"[INFO] Validating.")

        nn_model.eval()

        mini_batches = 0
        loss_val = 0

        for batch_id, (model, label) in enumerate(val_set_loader, 1):

            if self.config.device.type == 'cuda':
                model, label = model.cuda(), label
            else:
                model, label = model, label

            output = nn_model(model)
            loss = loss_function(output, label)

            mini_batches += 1
            loss_val += float(loss)

        loss_val = loss_val / mini_batches
        return loss_val
