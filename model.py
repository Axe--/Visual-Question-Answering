import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VQABaselineNet(nn.Module):
    """Baseline VQA Architecture"""

    def __init__(self, lstm_params, vgg_pretrained_path=None, is_vgg_pretrained=False):
        super(VQABaselineNet, self).__init__()

        self.image_encoder = ImageEncoder(is_vgg_pretrained, vgg_pretrained_path)
        self.question_encoder = QuestionEncoder(**lstm_params)

        # MLP combining the image & question embeddings
        self.mlp = nn.Sequential(nn.Linear(in_features=1024, out_features=1000),
                                 nn.Linear(in_features=1000, out_features=1000))
        # Final Softmax layer
        self.cls_predict = nn.Softmax(dim=1)

    def forward(self, x_img, x_ques):
        x_img_embedding = self.image_encoder(x_img)
        x_ques_embedding = self.question_encoder(x_ques)

        # Combine the two using element-wise multiplication
        x_embedding = x_img_embedding * x_ques_embedding

        x_embedding = self.mlp(x_embedding)

        label_predict = self.cls_predict(x_embedding)

        return label_predict


class ImageEncoder(nn.Module):
    """Image Encoder - CNN"""

    def __init__(self, is_pretrained=False, pretrained_path=None):
        super(ImageEncoder, self).__init__()

        vgg11 = models.vgg11_bn(pretrained=is_pretrained)

        # Load Pre-Trained VGG_11 from disk if weights file (.pt) is specified
        if is_pretrained and pretrained_path:
            vgg11.load_state_dict(torch.load(pretrained_path))

        # Select all VGG layers (excluding the final FC-1000)
        vgg_layers = list(vgg11.features.children())
        vgg_layers.append(vgg11.avgpool)
        vgg_layers += list(vgg11.classifier.children())[0:4]

        # VGG Encoder: 224 x 224 ---pool 5x---> 7 x 7
        self.vgg11_encoder = nn.Sequential(*vgg_layers)

        # Image Embedding layer
        self.embedding_layer = nn.Linear(in_features=4096, out_features=1024)

    def forward(self, x_img):
        """
        Encodes image of size 224 x 224 to one-dimensional vector of size 1024.

        :param x_img: image tensor (batch, 3, 224, 224)
        :return: embedding tensor (batch, 1024)
        """
        x_img = self.vgg11_encoder(x_img)
        x_img = self.embedding_layer(x_img)
        x_img = F.normalize(x_img, dim=1, p=2)

        return x_img


class QuestionEncoder(nn.Module):
    """Question Encoder - LSTM"""

    def __init__(self, vocab_size, emb_dim, enc_units):
        super(QuestionEncoder, self).__init__()

        self.encoder_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = emb_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # RNN Layer: (input_dim, hidden_units)
        self.gru = nn.GRU(self.embedding_dim, self.encoder_units)
        self.hidden = None  # hidden state of the encoder

    def forward(self, x, seq_lengths, device):
        """
        Performs forward pass on question

        :param x: input tensor (batch)
        :param seq_lengths: corresponding sequence lengths (batch)
        :param device: computation device
        :return: network output tensor
        """
        x = self.embedding(x)
        x = pack_padded_sequence(x, seq_lengths)  # Un-Pad

        self.hidden = self.init_hidden_state(device)

        # x.shape : (seq_len, batch_size, emb_dim)
        output, self.hidden = self.gru(x, self.hidden)
        output, _ = pad_packed_sequence(output)
        # ^^ batch elements will be ordered decreasingly by their length

        # Return the last hidden state (for softmax)
        # For this use either hidden or out[-1]
        return self.hidden
