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

run_recurrent = False    # else run Token-wise MLP
use_RNN = True          # otherwise GRU
atten_size = 2          # atten > 0 means using restricted self atten

reload_model = False
num_epochs = 10
learning_rate = 0.001
test_interval = 50

# Loading sataset, use toy = True for obtaining a smaller dataset

train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size, toy=True)


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
        hidden = (1 - z_t) * hidden_state + z_t * h_tilde_t
        output = torch.sigmoid(self.fc(hidden))

        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size).to(device)

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

        padded = pad(x, (0, 0, atten_size, atten_size, 0, 0))

        x_nei = []
        for k in range(-atten_size, atten_size + 1):
            x_nei.append(torch.roll(padded, k, 1))

        x_nei = torch.stack(x_nei, 2)
        x_nei = x_nei[:, atten_size:-atten_size, :]

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


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()

        self.ReLU = torch.nn.ReLU()

        self.intermediate_size = 32

        # Token-wise MLP network weights
        self.layer1 = MatMul(input_size, hidden_size)
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

def print_review_words_MLP(reviews, reviews_text):
    model2 = ExMLP(input_size, output_size, hidden_size)
    model2.load_state_dict(torch.load("MLP.pth"))
    y = model2(reviews).tolist()
    list = []
    for i in range(len(reviews_text[0])):
        list.append((y[0][i], reviews_text[0][i]))

    print()
    print("Words with corresponding sentiments:")
    print(list)
    print("Total sentiment of review (under softmax):")
    print(torch.softmax(torch.mean(model2(reviews), 1)[0], dim=0))
    print()

def print_review_words_MLP_Atten(reviews, reviews_text):
    model2 = ExRestSelfAtten(input_size, output_size, hidden_size)
    model2.load_state_dict(torch.load("MLP_atten.pth"))
    sub_score_, atten_weights_ = model2(reviews)
    y = sub_score_.tolist()
    z = atten_weights_.tolist()
    list = []
    for i in range(len(reviews_text[0])):
        list.append((reviews_text[0][i], y[0][i], z[0][i]))

    print()
    print("Words with corresponding sentiments:")
    print(list)
    print("Total sentiment of review (under softmax):")
    print(torch.softmax(torch.mean(sub_score_, 1)[0], dim=0))
    print()

def print_review_words_GRU(reviews, reviews_text):
    model3 = ExGRU(input_size, output_size, hidden_size)
    model3.load_state_dict(torch.load("GRU.pth"))
    hidden_state = model3.init_hidden(int(labels.shape[0]))
    for i in range(num_words):
        output, hidden_state = model3(reviews[:, i, :], hidden_state)  # HIDE

    print("GRU:")
    print(torch.softmax(output, dim=1)[0])

    rounded_output = (output == output.max(dim=1, keepdim=True).values).float()
    comparison = (rounded_output == labels)
    rows_agree = comparison.all(dim=1)
    num_rows_agree = rows_agree.sum().item()
    test_accuracy = num_rows_agree / len(labels)
    print("Accuracy: " + str(test_accuracy))

def print_review_words_RNN(reviews, reviews_text):
    model4 = ExRNN(input_size, output_size, hidden_size)
    model4.load_state_dict(torch.load("RNN.pth"))
    hidden_state = model4.init_hidden(int(labels.shape[0]))
    for i in range(num_words):
        output, hidden_state = model4(reviews[:, i, :], hidden_state)  # HIDE

    print("RNN:")
    print(torch.softmax(output, dim=1)[0])

    rounded_output = (output == output.max(dim=1, keepdim=True).values).float()
    comparison = (rounded_output == labels)
    rows_agree = comparison.all(dim=1)
    num_rows_agree = rows_agree.sum().item()
    test_accuracy = num_rows_agree / len(labels)
    print("Accuracy: " + str(test_accuracy))

if __name__ == '__main__':
    iter = 0

    # for labels, reviews, reviews_text in test_dataset:
    #     print_review_words_MLP(reviews, reviews_text)
    #
    # for labels, reviews, reviews_text in test_dataset:
    #     print_review_words_MLP_Atten(reviews, reviews_text)

    for labels, reviews, reviews_text in test_dataset:
        print(" ".join(reviews_text[0]))
        print(labels[0])
        # print_review_words_MLP(reviews, reviews_text)
        print_review_words_MLP_Atten(reviews, reviews_text)
        print()
        print()

