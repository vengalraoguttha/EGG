# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
import torchvision

from egg.core import LoggingStrategy
from egg.core.continous_communication import SenderReceiverContinuousCommunication
from egg.core.gs_wrappers import gumbel_softmax_sample


def get_vision_module(name: str = "resnet50", pretrained: bool = False):
    modules = {
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "resnet152": torchvision.models.resnet152(pretrained=pretrained),
    }
    if name not in modules:
        raise KeyError(f"{name} is not currently supported.")

    model = modules[name]

    n_features = model.fc.in_features
    model.fc = nn.Identity()

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
        model = model.eval()

    return model, n_features


def get_vision_modules(
    encoder_arch: str, shared: bool = False, pretrain_vision: bool = False
):
    if pretrain_vision:
        assert (
            shared
        ), "A pretrained not shared vision_module is a waste of memory. Please run with --shared set"

    encoder, features_dim = get_vision_module(encoder_arch, pretrain_vision)
    encoder_recv = None
    if not shared:
        encoder_recv, _ = get_vision_module(encoder_arch)

    return encoder, encoder_recv, features_dim


class VisionModule(nn.Module):
    def __init__(
        self,
        sender_vision_module: nn.Module,
        receiver_vision_module: Optional[nn.Module] = None,
    ):
        super(VisionModule, self).__init__()

        self.encoder = sender_vision_module

        self.shared = receiver_vision_module is None
        if not self.shared:
            self.encoder_recv = receiver_vision_module

    def forward(self, x_i, x_j):
        encoded_input_sender = self.encoder(x_i)
        if self.shared:
            encoded_input_recv = self.encoder(x_j)
        else:
            encoded_input_recv = self.encoder_recv(x_j)
        return encoded_input_sender, encoded_input_recv


class VisionGameWrapper(nn.Module):
    def __init__(
        self,
        game: nn.Module,
        vision_module: nn.Module,
    ):
        super(VisionGameWrapper, self).__init__()
        self.game = game
        self.vision_module = vision_module

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        x_i, x_j = sender_input
        sender_encoded_input, receiver_encoded_input = self.vision_module(x_i, x_j)

        return self.game(
            sender_input=sender_encoded_input,
            labels=labels,
            receiver_input=receiver_encoded_input,
        )


class SimCLRSender(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        discrete_evaluation: bool = False,
    ):
        super(SimCLRSender, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=False)

        self.discrete_evaluation = discrete_evaluation

    def forward(self, resnet_output, sender=False):
        first_projection = self.fc(resnet_output)

        if self.discrete_evaluation and (not self.training) and sender:
            logits = first_projection
            size = logits.size()
            indexes = logits.argmax(dim=-1)
            one_hot = torch.zeros_like(logits).view(-1, size[-1])
            one_hot.scatter_(1, indexes.view(-1, 1), 1)
            one_hot = one_hot.view(*size)
            first_projection = one_hot

        out = self.fc_out(first_projection)
        return out, first_projection.detach(), resnet_output.detach()


class EmSSLSender(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        temperature: float = 1.0,
        trainable_temperature: bool = False,
        straight_through: bool = False,
    ):
        super(EmSSLSender, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )
        self.straight_through = straight_through

        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, resnet_output):
        first_projection = self.fc(resnet_output)
        message = gumbel_softmax_sample(
            first_projection, self.temperature, self.training, self.straight_through
        )
        out = self.fc_out(message)
        return out, message.detach(), resnet_output.detach()


class Receiver(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 2048):
        super(Receiver, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, _x, resnet_output):
        return self.fc(resnet_output), resnet_output.detach()


class EmComSSLSymbolGame(SenderReceiverContinuousCommunication):
    def __init__(self, *args, **kwargs):
        super(EmComSSLSymbolGame, self).__init__(*args, **kwargs)

    def forward(self, sender_input, labels, receiver_input):
        if isinstance(self.sender, SimCLRSender):
            message, message_like, resnet_output_sender = self.sender(
                sender_input, sender=True
            )
            receiver_output, _, resnet_output_recv = self.receiver(receiver_input)
        else:
            message, message_like, resnet_output_sender = self.sender(sender_input)
            receiver_output, resnet_output_recv = self.receiver(message, receiver_input)

        loss, aux_info = self.loss(
            sender_input, message, receiver_input, receiver_output, labels, None
        )

        if hasattr(self.sender, "temperature"):
            if isinstance(self.sender.temperature, torch.nn.Parameter):
                temperature = self.sender.temperature.detach()
            else:
                temperature = torch.Tensor([self.sender.temperature])
            aux_info["temperature"] = temperature

        if not self.training:
            aux_info["message_like"] = message_like
            aux_info["resnet_output_sender"] = resnet_output_sender
            aux_info["resnet_output_recv"] = resnet_output_recv

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=None,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=torch.ones(message.size(0)),
            aux=aux_info,
        )

        return loss.mean(), interaction


class EmComFixedLengthSenderGS(nn.Module):

    def __init__(
        self,
        input_dim,
        vocab_size,
        embed_dim,
        hidden_size,
        temperature,
        cell="rnn",
        nos=1,
        trainable_temperature=False,
        straight_through=False,
    ):
        super(EmComFixedLengthSenderGS, self).__init__()
        self.input_dim = input_dim

        assert nos >= 1, "Cannot have a max_len below 1"
        self.nos = nos

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Linear(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )

        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )

        self.straight_through = straight_through
        self.cell = None

        cell = cell.lower()

        if cell == "rnn":
            self.cell = nn.RNNCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "gru":
            self.cell = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "lstm":
            self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, resnet_output, aux_input=None):
        first_projection = self.fc(resnet_output)
        prev_c = torch.zeros_like(first_projection)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * first_projection.size(0))
        sequence = []

        for step in range(self.nos):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (first_projection, prev_c))
            else:
                h_t = self.cell(e_t, first_projection)

            step_logits = self.hidden_to_output(h_t)
            x = gumbel_softmax_sample(
                step_logits, self.temperature, self.training, self.straight_through
            )

            first_projection = h_t
            e_t = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0, 2)

        return sequence


class EmComFixedLengthReceiverGS(nn.Module):

    def __init__(self, input_dim, vocab_size, embed_dim, hidden_size, cell="rnn", nos=1):
        super(EmComFixedLengthReceiverGS, self).__init__()
        self.input_dim = input_dim
        self.nos = nos  # Number of Symbols that the Receiver gets from Sender.
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.cell = None
        cell = cell.lower()
        if cell == "rnn":
            self.cell = nn.RNNCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "gru":
            self.cell = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "lstm":
            self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.embedding = nn.Linear(vocab_size, embed_dim)

    def forward(self, message, resnet_output=None, aux_input=None):

        emb = self.embedding(message)

        prev_hidden = None
        prev_c = None
        h_t = None

        # to get an access to the hidden states, we have to unroll the cell ourselves
        for step in range(message.size(1)):
            e_t = emb[:, step, ...]
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = (
                    self.cell(e_t, (prev_hidden, prev_c))
                    if prev_hidden is not None
                    else self.cell(e_t)
                )
            else:
                h_t = self.cell(e_t, prev_hidden)

            prev_hidden = h_t

        return h_t, self.fc.forward(resnet_output)


class EmComFixedLengthSenderReceiverGS(nn.Module):

    def __init__(
        self,
        sender,
        receiver,
        loss,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        super(EmComFixedLengthSenderReceiverGS, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        messages = self.sender(sender_input, aux_input)
        message, receiver_output = self.receiver(messages, receiver_input, aux_input)

        loss, aux_info = self.loss(
            sender_input,
            message,
            receiver_input,
            receiver_output,
            labels,
            aux_input,
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=torch.ones(message[0].size(0)),
            aux=aux_info,
        )

        return loss.mean(), interaction
