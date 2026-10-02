import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim
    ):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_size
        )

    def forward(self, X):
        output, _ = self.rnn(X)

        return self.fc(output)


class DeepRNN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim
    ):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_size
        )

    def forward(self, X):
        output, _ = self.rnn(X)

        return self.fc(output)


class SimpleGRU(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim
    ):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_size
        )

    def forward(self, X):
        output, _ = self.rnn(X)

        return self.fc(output)


class SimpleLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim 
    ):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_size
        )

    def forward(self, X):
        output, _ = self.rnn(X)

        return self.fc(output)


class BiGRU(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim
    ):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            hidden_dim * 2,
            output_size
        )

    def forward(self, X):
        output, _ = self.rnn(X)

        return self.fc(output)


class BiLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim
    ):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            hidden_dim * 2,
            output_size
        )

    def forward(self, X):
        output, _ = self.rnn(X)

        return self.fc(output)

class DeepBiLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim
    ):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.fc = nn.Linear(
            hidden_dim * 2,
            output_size
        )

    def forward(self, X):
        output, _ = self.rnn(X)

        return self.fc(output)


class BiLSTMProjection(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim,
        projection_dim
    ):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                projection_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                projection_dim,
                output_size
            )
        )

    def forward(self, X):
        output, _ = self.rnn(X)

        return self.classifier(output)


class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim,
        conv_dim=64
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=conv_dim,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )

        self.rnn = nn.LSTM(
            input_size=conv_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            hidden_dim * 2,
            output_size
        )

    def forward(self, X):
        X = X.transpose(1, 2)
        X = self.conv(X)
        X = X.transpose(1, 2)

        output, _ = self.rnn(X)

        return self.fc(output)



models = {
    "simple_rnn": SimpleRNN,
    "deep_rnn": DeepRNN,
    "simple_gru": SimpleGRU,
    "simple_lstm": SimpleLSTM,
    "bigru": BiGRU,
    "bilstm": BiLSTM,
    "deep_bilstm": DeepBiLSTM,
    "bilstm_projection": BiLSTMProjection,
    "cnn_bilstm": CNNBiLSTM,
}