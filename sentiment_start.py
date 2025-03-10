########################################################################
########################################################################
##                                                                    ##
##                      ORIGINAL _ DO NOT PUBLISH                     ##
##                                                                    ##
########################################################################
########################################################################

import torch as tr
import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld

import matplotlib.pyplot as plt
plot_graphs = True

pre = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cpu")

batch_size = 32
output_size = 2
hidden_size = 128        # to experiment with

run_recurrent = True    # else run Token-wise MLP
use_RNN = False          # otherwise GRU
atten_size = 0          # atten > 0 means using restricted self atten

reload_model = False
num_epochs = 10
learning_rate = 0.001
test_interval = 50

# Loading sataset, use toy = True for obtaining a smaller dataset

train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size)

# Special matrix multipication layer (like torch.Linear but can operate on arbitrary sized
# tensors and considers its last two indices as the matrix.)

class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias = True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels,out_channels)), requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1,1,out_channels), requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):        
        x = torch.matmul(x,self.matrix) 
        if self.use_bias:
            x = x+ self.bias 
        return x
        
# Implements RNN Unit

class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()

        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid

        # RNN Cell weights
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, output_size)

    def name(self):
        return "RNN"

    def forward(self, x, hidden_state):
        hidden = self.sigmoid(self.in2hidden(torch.cat((x, hidden_state), dim=1)))
        output = self.sigmoid(self.hidden2out(hidden))
        
        return output, hidden_state

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size).to(device)

# Implements GRU Unit

class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size

        # GRU Cell weights
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)  # Update gate
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)  # Reset gate
        self.W = nn.Linear(input_size + hidden_size, hidden_size)  # Current memory content

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def name(self):
        return "GRU"

    def forward(self, x, hidden_state):
        z_t = torch.sigmoid(self.W_z(torch.cat((x, hidden_state), dim=1)))
        r_t = torch.sigmoid(self.W_r(torch.cat((x, hidden_state), dim=1)))
        h_tilde_t = torch.tanh(self.W(torch.cat((x, r_t * hidden_state), dim=1)))
        hidden = (1-z_t) * hidden_state + z_t * h_tilde_t
        output = torch.sigmoid(self.fc(hidden))

        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size).to(device)


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()

        self.ReLU = torch.nn.ReLU()

        self.intermediate_size = 32

        # Token-wise MLP network weights
        self.layer1 = MatMul(input_size,hidden_size)
        # additional layer(s)
        self.layer2 = MatMul(hidden_size, self.intermediate_size)

        self.layer3 = MatMul(self.intermediate_size, output_size)
        

    def name(self):
        return "MLP"

    def forward(self, x):

        # Token-wise MLP network implementation
        
        x = torch.relu(self.layer1(x))
        # rest
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))

        return x


class ExRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRestSelfAtten, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(2)
        
        # Token-wise MLP + Restricted Attention network implementation

        self.layer1 = MatMul(input_size, hidden_size)


        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=True)

        self.layer2 = MatMul(hidden_size, 32)
        self.layer3 = MatMul(32, output_size)

        # Positional Encoding
        self.positional_encoding = self.create_positional_encoding(hidden_size, 100)

    def create_positional_encoding(self, hidden_size, max_len):
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-np.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return nn.Parameter(pe, requires_grad=False)

    def name(self):
        return "MLP_atten"

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Token-wise MLP + Restricted Attention network implementation

        x = self.layer1(x)
        x = self.ReLU(x)
        x = x + self.positional_encoding[:, :seq_len, :]

        # generating x in offsets between -atten_size and atten_size 
        # with zero padding at the ends

        padded = pad(x,(0,0,atten_size,atten_size,0,0))

        x_nei = []
        for k in range(-atten_size,atten_size+1):
            x_nei.append(torch.roll(padded, k, 1))

        x_nei = torch.stack(x_nei,2)
        x_nei = x_nei[:,atten_size:-atten_size,:]
        
        # x_nei has an additional axis that corresponds to the offset

        # Applying attention layer

        query = self.W_q(x).unsqueeze(2)
        keys = self.W_k(x_nei)
        vals = self.W_v(x_nei)
        attention_scores = torch.matmul(query, keys.transpose(-2, -1)) / self.sqrt_hidden_size
        atten_weights = torch.softmax(attention_scores, dim=3)
        context = torch.matmul(atten_weights, vals).squeeze(2)
        output = self.layer3(self.ReLU(self.layer2(context)))

        return output, atten_weights


# prints portion of the review (20-30 first words), with the sub-scores each work obtained
# prints also the final scores, the softmaxed prediction values and the true label values

def print_review(rev_text, sbs1, sbs2, lbl1, lbl2):
            
    # implement
    pass

# select model to use
if __name__ == '__main__':
    if run_recurrent:
        if use_RNN:
            model = ExRNN(input_size, output_size, hidden_size)
        else:
            model = ExGRU(input_size, output_size, hidden_size)
    else:
        if atten_size > 0:
            model = ExRestSelfAtten(input_size, output_size, hidden_size)
        else:
            model = ExMLP(input_size, output_size, hidden_size)

    print("Using model: " + model.name())

    if reload_model:
        print("Reloading model")
        model.load_state_dict(torch.load(model.name() + ".pth"))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = 1.0
    test_loss = 1.0

    train_losses = list()
    test_losses = list()
    train_accuracies = list()
    test_accuracies = list()

    total_epoch_test_accuracies = list()
    total_epoch_train_accuracies = list()

    total_epoch_test_losses = list()
    total_epoch_train_losses = list()


    # training steps in which a test step is executed every test_interval

    for epoch in range(num_epochs):

        epoch_test_accuracies = list()
        epoch_train_accuracies = list()

        epoch_test_losses = list()
        epoch_train_losses = list()

        itr = 0 # iteration counter within each epoch

        for labels, reviews, reviews_text in train_dataset:   # getting training batches

            itr = itr + 1

            labels_train, reviews_train, reviews_text_train = labels, reviews, reviews_text

            if (itr + 1) % test_interval == 0:
                test_iter = True
                labels, reviews, reviews_text = next(iter(test_dataset)) # get a test batch
            else:

                test_iter = False

            # Recurrent nets (RNN/GRU)

            if run_recurrent:
                hidden_state = model.init_hidden(int(labels.shape[0]))

                for i in range(num_words):
                    output, hidden_state = model(reviews[:,i,:], hidden_state)  # HIDE

                if test_iter:
                    hidden_state_train = model.init_hidden(int(labels.shape[0]))

                    for i in range(num_words):
                        output_train, hidden_state_train = model(reviews_train[:, i, :], hidden_state_train)  # HIDE

            else:
            # Token-wise networks (MLP / MLP + Atten.)
                sub_score = []
                if atten_size > 0:
                    # MLP + atten
                    sub_score, atten_weights = model(reviews)
                else:
                    # MLP
                    sub_score = model(reviews)

                output = torch.mean(sub_score, 1)

                if test_iter:
                    sub_score_train = []
                    if atten_size > 0:
                        # MLP + atten
                        sub_score_train, atten_weights_train = model(reviews_train)
                    else:
                        # MLP
                        sub_score_train = model(reviews_train)

                    output_train = torch.mean(sub_score_train, 1)


            # cross-entropy loss

            loss = criterion(output, labels)

            # optimize in training iterations

            if not test_iter:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # averaged losses
            if test_iter:
                test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
            else:
                train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss

            if test_iter:
                rounded_output = (output == output.max(dim=1, keepdim=True).values).float()
                comparison = (rounded_output == labels)
                rows_agree = comparison.all(dim=1)
                num_rows_agree = rows_agree.sum().item()
                test_accuracy = num_rows_agree / len(labels)
                test_accuracies.append(test_accuracy)

                epoch_test_accuracies.append(test_accuracy)

                rounded_output = (output_train == output_train.max(dim=1, keepdim=True).values).float()
                comparison = (rounded_output == labels_train)
                rows_agree = comparison.all(dim=1)
                num_rows_agree = rows_agree.sum().item()
                train_accuracy = num_rows_agree / len(labels_train)
                train_accuracies.append(train_accuracy)

                epoch_train_accuracies.append(train_accuracy)


                epoch_train_losses.append(train_loss)
                epoch_test_losses.append(test_loss)

                train_losses.append(train_loss)
                test_losses.append(test_loss)

                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{itr + 1}/{len(train_dataset)}], "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Test Loss: {test_loss:.4f} "
                    f"Test Accuracy: {test_accuracy:.4f} "
                    f"Train Accuracy: {train_accuracy:.4f}"
                )

                if not run_recurrent:
                    nump_subs = sub_score.detach().numpy()
                    labels = labels.detach().numpy()
                    print_review(reviews_text[0], nump_subs[0,:,0], nump_subs[0,:,1], labels[0,0], labels[0,1])

                # saving the model
                torch.save(model.state_dict(), model.name() + ".pth")

        total_epoch_test_losses.append(np.mean(epoch_test_losses))
        total_epoch_train_losses.append(np.mean(epoch_train_losses))
        total_epoch_test_accuracies.append(np.mean(epoch_test_accuracies))
        total_epoch_train_accuracies.append(np.mean(epoch_train_accuracies))



    if plot_graphs:
        epochs = list(range(1, len(train_losses) + 1))
        plt.figure(figsize=(10, 6))  # specify figure size
        plt.plot(epochs, test_accuracies, label='Test Accuracy')
        plt.plot(epochs, train_accuracies, label='Train Accuracy')
        plt.title('Accuracies over Steps')
        plt.xlabel('Step / 50')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



        epochs = list(range(1, len(train_losses) + 1))
        plt.figure(figsize=(10, 6))  # specify figure size
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, test_losses, label='Test Loss')
        plt.title('Losses over Steps')
        plt.xlabel('Step / 50')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



        epochs = list(range(1, len(total_epoch_test_accuracies) + 1))
        plt.figure(figsize=(10, 6))  # specify figure size
        plt.plot(epochs, total_epoch_test_accuracies, label='Test accuracy')
        plt.plot(epochs, total_epoch_train_accuracies, label='Train accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



        epochs = list(range(1, len(total_epoch_test_losses) + 1))
        plt.figure(figsize=(10, 6))  # specify figure size
        plt.plot(epochs, total_epoch_test_losses, label='Test loss')
        plt.plot(epochs, total_epoch_train_losses, label='Train loss')
        plt.title('Losses over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
