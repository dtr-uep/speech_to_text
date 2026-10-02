
import torch
import torch.nn as nn


# ============================================================
# Base Encoder-Decoder RNN
# ============================================================

class EncoderDecoderRNN(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim
    ):
        super().__init__()

        self.encoder = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.decoder = nn.RNN(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.embedding = nn.Embedding(
            output_size,
            hidden_dim
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_size
        )

    def forward(self, X, decoder_input):

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        _, hidden = self.encoder(X)

        # hidden:
        # [1, B, H]

        # ----------------------------------------------------
        # Decoder input tokens -> embeddings
        # ----------------------------------------------------

        decoder_emb = self.embedding(decoder_input)

        # [B, T_text, H]

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        output, _ = self.decoder(
            decoder_emb,
            hidden
        )

        # [B, T_text, H]

        return self.fc(output)


# ============================================================
# Deep RNN Encoder-Decoder
# ============================================================

class DeepEncoderDecoderRNN(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim
    ):
        super().__init__()

        self.encoder = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        self.decoder = nn.RNN(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        self.embedding = nn.Embedding(
            output_size,
            hidden_dim
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_size
        )

    def forward(self, X, decoder_input):

        _, hidden = self.encoder(X)

        decoder_emb = self.embedding(
            decoder_input
        )

        output, _ = self.decoder(
            decoder_emb,
            hidden
        )

        return self.fc(output)


# ============================================================
# GRU Encoder-Decoder
# ============================================================

class EncoderDecoderGRU(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim
    ):
        super().__init__()

        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.embedding = nn.Embedding(
            output_size,
            hidden_dim
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_size
        )

    def forward(self, X, decoder_input):

        _, hidden = self.encoder(X)

        decoder_emb = self.embedding(
            decoder_input
        )

        output, _ = self.decoder(
            decoder_emb,
            hidden
        )

        return self.fc(output)


# ============================================================
# LSTM Encoder-Decoder
# ============================================================

class EncoderDecoderLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.embedding = nn.Embedding(
            output_size,
            hidden_dim
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_size
        )

    def forward(self, X, decoder_input):

        _, (hidden, cell) = self.encoder(X)

        decoder_emb = self.embedding(
            decoder_input
        )

        output, _ = self.decoder(
            decoder_emb,
            (hidden, cell)
        )

        return self.fc(output)


# ============================================================
# Bidirectional GRU Encoder + GRU Decoder
# ============================================================

class BiEncoderGRUDecoder(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim
    ):
        super().__init__()

        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.decoder = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.embedding = nn.Embedding(
            output_size,
            hidden_dim * 2
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_size
        )

    def forward(self, X, decoder_input):

        _, encoder_hidden = self.encoder(X)

        # [2, B, H] -> [B, 2H]
        encoder_context = torch.cat(
            [
                encoder_hidden[0],
                encoder_hidden[1]
            ],
            dim=1
        )

        # Repeat context for every decoder timestep
        decoder_context = self.embedding(
            decoder_input
        )

        # [B, T, 2H]
        context = encoder_context.unsqueeze(1)

        context = context.expand(
            -1,
            decoder_context.size(1),
            -1
        )

        decoder_input_emb = decoder_context

        output, _ = self.decoder(
            decoder_input_emb,
            encoder_hidden[0:1]
        )

        return self.fc(output)


# ============================================================
# Bidirectional LSTM Encoder + LSTM Decoder
# ============================================================

class BiEncoderLSTMDecoder(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.embedding = nn.Embedding(
            output_size,
            hidden_dim
        )

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.hidden_projection = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        self.cell_projection = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_size
        )

    def forward(self, X, decoder_input):

        _, (hidden, cell) = self.encoder(X)

        # Combine forward/backward states
        hidden = torch.cat(
            [hidden[0], hidden[1]],
            dim=1
        )

        cell = torch.cat(
            [cell[0], cell[1]],
            dim=1
        )

        hidden = self.hidden_projection(
            hidden
        ).unsqueeze(0)

        cell = self.cell_projection(
            cell
        ).unsqueeze(0)

        decoder_emb = self.embedding(
            decoder_input
        )

        output, _ = self.decoder(
            decoder_emb,
            (hidden, cell)
        )

        return self.fc(output)


# ============================================================
# Deep BiLSTM Encoder + LSTM Decoder
# ============================================================

class DeepBiEncoderLSTMDecoder(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.embedding = nn.Embedding(
            output_size,
            hidden_dim
        )

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.hidden_projection = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        self.cell_projection = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_size
        )

    def forward(self, X, decoder_input):

        _, (hidden, cell) = self.encoder(X)

        # Use the final encoder layer
        hidden = torch.cat(
            [hidden[-2], hidden[-1]],
            dim=1
        )

        cell = torch.cat(
            [cell[-2], cell[-1]],
            dim=1
        )

        hidden = self.hidden_projection(
            hidden
        ).unsqueeze(0)

        cell = self.cell_projection(
            cell
        ).unsqueeze(0)

        decoder_emb = self.embedding(
            decoder_input
        )

        output, _ = self.decoder(
            decoder_emb,
            (hidden, cell)
        )

        return self.fc(output)


# ============================================================
# BiLSTM + Projection
# ============================================================

class BiEncoderLSTMProjection(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim,
        projection_dim
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.embedding = nn.Embedding(
            output_size,
            hidden_dim
        )

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.hidden_projection = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        self.cell_projection = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim,
                projection_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                projection_dim,
                output_size
            )
        )

    def forward(self, X, decoder_input):

        _, (hidden, cell) = self.encoder(X)

        hidden = torch.cat(
            [hidden[0], hidden[1]],
            dim=1
        )

        cell = torch.cat(
            [cell[0], cell[1]],
            dim=1
        )

        hidden = self.hidden_projection(
            hidden
        ).unsqueeze(0)

        cell = self.cell_projection(
            cell
        ).unsqueeze(0)

        decoder_emb = self.embedding(
            decoder_input
        )

        output, _ = self.decoder(
            decoder_emb,
            (hidden, cell)
        )

        return self.classifier(output)


# ============================================================
# CNN + BiLSTM Encoder + LSTM Decoder
# ============================================================

class CNNBiEncoderLSTMDecoder(nn.Module):

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
                input_size,
                conv_dim,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )

        self.encoder = nn.LSTM(
            input_size=conv_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.embedding = nn.Embedding(
            output_size,
            hidden_dim
        )

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.hidden_projection = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        self.cell_projection = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_size
        )

    def forward(self, X, decoder_input):

        # [B, T, F]
        X = X.transpose(1, 2)

        # [B, F, T]
        X = self.conv(X)

        # [B, conv_dim, T]
        X = X.transpose(1, 2)

        # [B, T, conv_dim]
        _, (hidden, cell) = self.encoder(X)

        hidden = torch.cat(
            [hidden[0], hidden[1]],
            dim=1
        )

        cell = torch.cat(
            [cell[0], cell[1]],
            dim=1
        )

        hidden = self.hidden_projection(
            hidden
        ).unsqueeze(0)

        cell = self.cell_projection(
            cell
        ).unsqueeze(0)

        decoder_emb = self.embedding(
            decoder_input
        )

        output, _ = self.decoder(
            decoder_emb,
            (hidden, cell)
        )

        return self.fc(output)


# ============================================================
# Transformer Encoder-Decoder
# ============================================================

class TransformerSeq2Seq(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        ff_dim,
        dropout=0.1
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_size,
            hidden_dim
        )

        self.embedding = nn.Embedding(
            output_size,
            hidden_dim
        )

        self.positional_encoding = nn.Parameter(
            torch.randn(
                1,
                2000,
                hidden_dim
            )
        )

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_size
        )

    def forward(self, X, decoder_input):

        # ----------------------------------------------------
        # Encoder input
        # ----------------------------------------------------
        X = self.input_projection(X)
        X = X + self.positional_encoding[
            :, :X.size(1)
        ]
        
        # ----------------------------------------------------
        # Decoder input
        # ----------------------------------------------------

        decoder = self.embedding(
            decoder_input
        )

        decoder = decoder + self.positional_encoding[
            :, :decoder.size(1)
        ]

        # ----------------------------------------------------
        # Causal decoder mask
        # ----------------------------------------------------

        T = decoder.size(1)

        causal_mask = torch.triu(
            torch.ones(
                T,
                T,
                device=X.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        output = self.transformer(
            src=X,
            tgt=decoder,
            tgt_mask=causal_mask
        )

        return self.fc(output)


# ============================================================
# Model registry
# ============================================================

models = {

    "encoder_decoder_rnn":
        EncoderDecoderRNN,

    "deep_encoder_decoder_rnn":
        DeepEncoderDecoderRNN,

    "encoder_decoder_gru":
        EncoderDecoderGRU,

    "encoder_decoder_lstm":
        EncoderDecoderLSTM,

    "bi_encoder_gru_decoder":
        BiEncoderGRUDecoder,

    "bi_encoder_lstm_decoder":
        BiEncoderLSTMDecoder,

    "deep_bi_encoder_lstm_decoder":
        DeepBiEncoderLSTMDecoder,

    "bi_encoder_lstm_projection":
        BiEncoderLSTMProjection,

    "cnn_bi_encoder_lstm_decoder":
        CNNBiEncoderLSTMDecoder,

    "transformer_seq2seq":
        TransformerSeq2Seq,
}
