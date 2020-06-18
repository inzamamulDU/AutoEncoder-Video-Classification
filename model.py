import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

class CNNEncoder(nn.Module):
    def __init__(self, cnn_out_dim=256, drop_prob=0.3, bn_momentum=0.01):
        '''
        Use the pre-trained model provided by pytorch as the encoder
        '''
        super(CNNEncoder, self).__init__()

        self.cnn_out_dim = cnn_out_dim
        self.drop_prob = drop_prob
        self.bn_momentum = bn_momentum

        # Use the resnet pre-trained model to extract features and remove the last classifier
        pretrained_cnn = models.resnet152(pretrained=True)
        cnn_layers = list(pretrained_cnn.children())[:-1]

        # Remove the last fc layer of resnet to extract features
        self.cnn = nn.Sequential(*cnn_layers)
        # Embed features into cnn_out_dim dimension vector
        self.fc = nn.Sequential(
            *[
                self._build_fc(pretrained_cnn.fc.in_features, 512, True),
                nn.ReLU(),
                self._build_fc(512, 512, True),
                nn.ReLU(),
                nn.Dropout(p=self.drop_prob),
                self._build_fc(512, self.cnn_out_dim, False)
            ]
        )

    def _build_fc(self, in_features, out_features, with_bn=True):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features, momentum=self.bn_momentum)
        ) if with_bn else nn.Linear(in_features, out_features)

    def forward(self, x_3d):
        '''
        The input is a T frame image，shape = (batch_size, t, h, w, 3)
        '''
        cnn_embedding_out = []
        for t in range(x_3d.size(1)):
            # Use cnn to extract features
            with torch.no_grad():
                x = self.cnn(x_3d[:, t, :, :, :])
                x = torch.flatten(x, start_dim=1)

            # Handle the fc layer
            x = self.fc(x)

            cnn_embedding_out.append(x)

        cnn_embedding_out = torch.stack(cnn_embedding_out, dim=0).transpose(0, 1)

        return cnn_embedding_out

class RNNDecoder(nn.Module):
    def __init__(self, use_gru=True, cnn_out_dim=256, rnn_hidden_layers=3, rnn_hidden_nodes=256,
            num_classes=10, drop_prob=0.3):
        super(RNNDecoder, self).__init__()

        self.rnn_input_features = cnn_out_dim
        self.rnn_hidden_layers = rnn_hidden_layers
        self.rnn_hidden_nodes = rnn_hidden_nodes

        self.drop_prob = drop_prob
        self.num_classes = num_classes # Adjust the number of categories here

        # rnn configuration parameter
        rnn_params = {
            'input_size': self.rnn_input_features,
            'hidden_size': self.rnn_hidden_nodes,
            'num_layers': self.rnn_hidden_layers,
            'batch_first': True
        }

        # Use lstm or gru as the rnn layer
        self.rnn = (nn.GRU if use_gru else nn.LSTM)(**rnn_params)

        # rnn layer output to linear classifier
        self.fc = nn.Sequential(
            nn.Linear(self.rnn_hidden_nodes, 128),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x_rnn):
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(x_rnn, None)
        # batch_first=True guarantees the following structure：
        # rnn_out shape: (batch, timestep, output_size)
        # h_n and h_c shape: (n_layers, batch, hidden_size)

        x = self.fc(rnn_out[:, -1, :]) # Only extract the last layer for output

        return x
