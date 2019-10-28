import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VQABaselineNet(nn.Module):
    """Baseline VQA Architecture"""

    def __init__(self, ques_enc_params, img_enc_params, K):
        super(VQABaselineNet, self).__init__()

        self.image_encoder = ImageEncoder(**img_enc_params)
        self.question_encoder = QuestionEncoder(**ques_enc_params)

        # MLP combining the image & question embeddings
        self.mlp = nn.Sequential(nn.Linear(1024, 1000),
                                 nn.Dropout(0.5),
                                 nn.Tanh())

        # Final classification layer (computes logits)
        self.fc_final = nn.Linear(1000, K)

    def forward(self, x_img, x_ques, x_ques_len, device):
        x_img_embedding = self.image_encoder(x_img)                             # after: (batch_size, 1024)
        x_ques_embedding = self.question_encoder(x_ques, x_ques_len, device)    # after: (batch_size, 1024)

        # Combine the two using element-wise multiplication
        x_embedding = x_img_embedding * x_ques_embedding

        x_embedding = self.mlp(x_embedding)

        x_logits = self.fc_final(x_embedding)

        # We don't require nn.Softmax(), i.e. label_softmax = self.cls_predict(x_embedding)
        # Since, nn.CrossEntropyLoss() internally computes softmax.

        return x_logits


class ImageEncoder(nn.Module):
    """Image Encoder - CNN"""
    def __init__(self, is_trainable, weights_path):
        super(ImageEncoder, self).__init__()

        # If VGG weights file is given, set pretrained=False (to avoid duplicate download to .cache)
        vgg11 = models.vgg11_bn(pretrained=not weights_path)

        # Load Pre-Trained VGG_11 from disk, if weights file (.pth) is specified
        if weights_path:
            vgg11.load_state_dict(torch.load(weights_path))

        # Select all VGG layers (excluding the final FC-1000)
        vgg_layers = list(vgg11.features.children())
        vgg_layers += list([vgg11.avgpool, nn.Flatten()])
        vgg_layers += list(vgg11.classifier.children())[0:4]

        # VGG Encoder: 224 x 224 ---[pool 5x]---> 7 x 7  ---[FC + L2_norm]---> 4096-dim
        self.vgg11_encoder = nn.Sequential(*vgg_layers)

        # Image embedding layer (1024-dim)
        self.embedding_layer = nn.Sequential(nn.Linear(4096, 1024),
                                             nn.Tanh())

        # Freeze the VGG Encoder layers (is_trainable == False)
        if not is_trainable:
            for param in self.vgg11_encoder.parameters():
                param.requires_grad = False

    def forward(self, x_img):
        """
        Encodes image of size 224 x 224 to one-dimensional vector of size 1024.

        :param x_img: image tensor (batch, 3, 224, 224)
        :return: embedding tensor (batch, 1024)
        """
        # Encode Image: 224 x 224 --> 4096
        x_img = self.vgg11_encoder(x_img)
        x_img = F.normalize(x_img, dim=1, p=2)

        # Compute Embedding
        x_emb = self.embedding_layer(x_img)

        return x_emb


class QuestionEncoder(nn.Module):
    """Question Encoder - GRU"""

    def __init__(self, vocab_size, inp_emb_dim, enc_units, batch_size):
        super(QuestionEncoder, self).__init__()

        self.batch_size = batch_size
        self.encoder_hidden_units = enc_units
        self.vocab_size = vocab_size
        self.input_embedding_dim = inp_emb_dim

        # Input word embedding lookup matrix
        self.word_embedding_matrix = nn.Sequential(nn.Embedding(self.vocab_size, self.input_embedding_dim),
                                                   nn.Tanh())

        # RNN Layer: (input_dim, hidden_units)
        self.gru = nn.GRU(self.input_embedding_dim, self.encoder_hidden_units)
        # hidden state of the encoder
        self.hidden = None

        # Question embedding layer (1024-dim)
        self.embedding_layer = nn.Linear(in_features=enc_units, out_features=1024)

    def forward(self, x, seq_lengths, device):
        """
        Performs forward pass on question sequence

        :param x: input tensor (seq_len, batch)
        :param seq_lengths: corresponding sequence lengths (batch)
        :param device: computation device
        :return: output embedding tensor (batch_size, encoder_hidden_units)
                | `after torch.squeeze(dim=0)`
        """
        x = self.word_embedding_matrix(x)
        # x.shape : (seq_len, batch_size, emb_dim)

        # Without packed padded sequences, hidden and cell are tensors from the last element in the sequence,
        # which will most probably be a pad token, however when using packed padded sequences
        # they are both from the last 'non-padded' element in the sequence.
        x = pack_padded_sequence(x, seq_lengths)  # Un-Pad

        self.hidden = self.initialize_hidden_state(device)

        output = None
        try:
            # Fwd Pass
            output, self.hidden = self.gru(x, self.hidden)

        except RuntimeError:
            print(x.shape)
            print(self.hidden.shape)
            print()

        # Pad back (but to max-length of the mini-batch)
        output, _ = pad_packed_sequence(output)
        # ^^ batch elements will be ordered decreasingly by their length

        # *** Note on pack_padded_sequence() & pad_packed_sequence():
        # given a batch of sequences padded to global_max_len, after applying above functions,
        # we get a batch of sequences of local_max_len (i.e. max length of sequence in the mini-batch & NOT the dataset)

        # We can use either hidden or out[-1]
        # Squeeze: [1, batch_size, enc_hidden_units] --> [batch_size, enc_hidden_units]
        output_final_state = torch.squeeze(self.hidden)

        # Map the final hidden state output to 1024-dim (image-text joint space)
        x_emb = self.embedding_layer(output_final_state)

        return x_emb

    def initialize_hidden_state(self, device):
        return torch.zeros((1, self.batch_size, self.encoder_hidden_units)).to(device)
