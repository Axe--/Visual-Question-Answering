import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import OrderedDict


class VQABaselineNet(nn.Module):
    """Baseline VQA Architecture"""

    def __init__(self, ques_enc_params, img_enc_params, K):
        super(VQABaselineNet, self).__init__()

        self.image_encoder = ImageBaselineEncoder(**img_enc_params)
        self.question_encoder = QuestionBaselineEncoder(**ques_enc_params)

        # MLP combining the image & question embeddings
        self.mlp = nn.Sequential(nn.Linear(1024, 1000),
                                 nn.Dropout(0.5),
                                 nn.Tanh())

        # Final classification layer (computes logits)
        self.fc_final = nn.Linear(1000, K)

    def forward(self, x_img, x_ques, x_ques_len):
        x_img_embedding = self.image_encoder(x_img)                             # x_img_emb: [batch_size, 1024]
        x_ques_embedding = self.question_encoder(x_ques, x_ques_len)            # x_ques_emb: [batch_size, 1024]

        # Combine the two using element-wise multiplication
        x_embedding = x_img_embedding * x_ques_embedding

        x_embedding = self.mlp(x_embedding)

        x_logits = self.fc_final(x_embedding)

        return x_logits


class ImageBaselineEncoder(nn.Module):
    """VGG Image Encoder with 4096-dim embedding"""
    def __init__(self, is_trainable, weights_path):
        super(ImageBaselineEncoder, self).__init__()

        self.is_trainable = is_trainable
        self.weights_path = weights_path

        # VGG Encoder: 224 x 224 ---[pool 5x]---> 7 x 7  ---[FC + L2_norm]---> 4096-dim
        self.vgg11_encoder = self.build_vgg_encoder()

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

    def build_vgg_encoder(self):
        """
        Given VGG model, builds the encoder network from all the VGG layers \n
        except for the final classification layer

        :return: model (nn.Module)
        """
        # If VGG weights file is given, set pretrained=False (to avoid duplicate download to .cache)
        vgg11 = models.vgg11_bn(pretrained=not self.weights_path)

        # Load Pre-Trained VGG_11 from disk, if weights file (.pth) is specified
        if self.weights_path:
            vgg11.load_state_dict(torch.load(self.weights_path))

        # Select all VGG layers (excluding the final FC-1000)
        fc_layers = nn.Sequential(nn.Flatten(), *list(vgg11.classifier)[:-1])

        # VGG Encoder: 224x224 ---[pool 5x]---> 7x7  ---[FC + L2_norm]---> 4096-dim
        vgg_encoder = nn.Sequential(OrderedDict([('conv_layers', vgg11.features),
                                                 ('avgpool', vgg11.avgpool),
                                                 ('fc_layers', fc_layers)]))

        # Freeze the VGG Encoder layers (is_trainable == False)
        if not self.is_trainable:
            for param in vgg_encoder.parameters():
                param.requires_grad = False

        return vgg_encoder


class QuestionBaselineEncoder(nn.Module):
    """Question Encoder - GRU"""

    def __init__(self, vocab_size, inp_emb_dim, enc_units, batch_size):
        super(QuestionBaselineEncoder, self).__init__()

        self.batch_size = batch_size
        self.encoder_hidden_units = enc_units
        self.vocab_size = vocab_size
        self.input_embedding_dim = inp_emb_dim

        # Input word embedding lookup matrix
        self.word_embedding_matrix = nn.Sequential(nn.Embedding(self.vocab_size, self.input_embedding_dim),
                                                   nn.Tanh())

        # RNN Layer: (input_dim, hidden_units)
        self.gru = nn.GRU(self.input_embedding_dim, self.encoder_hidden_units)

        # Question embedding layer (1024-dim)
        self.embedding_layer = nn.Sequential(nn.Linear(enc_units, 1024),
                                             nn.Tanh())

    def forward(self, x, seq_lengths):
        """
        Performs forward pass on question sequence

        :param x: input tensor (seq_len, batch)
        :param seq_lengths: corresponding sequence lengths (batch)
        :return: output embedding tensor (batch_size, encoder_hidden_units)
                | `after torch.squeeze(dim=0)`
        """
        x = self.word_embedding_matrix(x)                   # [seq_len, batch_size, emb_dim]

        # By default, hidden (& cell) are the final state in the sequence, viz. mostly pad token.
        # `PackedSequence` selects the last 'non-pad' element in the sequence.
        x = pack_padded_sequence(x, seq_lengths, batch_first=True)  # Un-Pad

        # Fwd Pass  (use either hidden or out[-1])
        outputs, hidden = self.gru(x)

        # Squeeze: [1, batch_size, enc_hidden_units] --> [batch_size, enc_hidden_units]
        hidden = torch.squeeze(hidden, dim=0)

        # Map the final hidden state output to 1024-dim (image-text joint space)
        x_emb = self.embedding_layer(hidden)

        return x_emb


# **************************************************************
class HierarchicalCoAttentionNet(nn.Module):
    """Hierarchical Co-Attention Architecture"""

    def __init__(self, ques_enc_params, img_enc_params, K):
        super().__init__()

        self.image_encoder = ImageCoAttentionEncoder(**img_enc_params)
        self.question_encoder = QuestionCoAttentionEncoder(**ques_enc_params)
        self.parallel_co_attention = ParallelCoAttention()

    def forward(self):
        pass


class ImageCoAttentionEncoder(nn.Module):
    """VGG Image Encoder with 512 x 14 x 14 feature map"""

    def __init__(self, is_trainable, weights_path):
        super(ImageCoAttentionEncoder, self).__init__()

        self.is_trainable = is_trainable
        self.weights_path = weights_path

        # VGG Encoder: 448 x 448 x 3 ---[pool 5x]---> 512 x 14 x 14
        self.vgg11_encoder = self.build_vgg_encoder()

        # Flatten the feature map grid [B, D, H, W] --> [B, D, H*W]
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x_img):
        """
        Encodes image of size 448 x 448 to feature map of size 512 x 14 x 14.

        :param x_img: image tensor (batch, 3, 224, 224)
        :return: embedding tensor (batch, 1024)
        """
        x_feat_map = self.vgg11_encoder(x_img)

        # Flatten (14 x 14 x 512) --> (14*14, 512)
        x_feat = self.flatten(x_feat_map)

        return x_feat

    def build_vgg_encoder(self):
        """
        Given VGG model, builds the encoder network from all the VGG layers \n
        except for the final classification layer

        :return: model (nn.Module)
        """
        # If VGG weights file is given, set pretrained=False (to avoid duplicate download to .cache)
        vgg11 = models.vgg11_bn(pretrained=not self.weights_path)

        # Load Pre-Trained VGG_11 from disk, if weights file (.pth) is specified
        if self.weights_path:
            vgg11.load_state_dict(torch.load(self.weights_path))

        # conv_1 --- [Conv - BatchNorm - MaxPool] (5x) ---> max_pool_5
        vgg_encoder = vgg11.features

        # Freeze the VGG Encoder layers (is_trainable == False)
        if not self.is_trainable:
            for param in vgg_encoder.parameters():
                param.requires_grad = False

        return vgg_encoder


class QuestionCoAttentionEncoder(nn.Module):
    """
    Encode question phrases using 1D Convolution for
    filter sizes:- 1: unigram, 2:bigram, 3:trigram. \n

    Max-pool the sequence (across the n-grams dim) \n

    Finally, apply an LSTM to encode the question.
    """
    def __init__(self, vocab_size, inp_emb_dim, enc_units, batch_size):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = inp_emb_dim
        self.lstm_units = enc_units

        # Word Embedding matrix
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Phrase Convolution + MaxPool
        self.phrase_conv = PhraseConvPool(self.embedding_dim)

        # Sentence LSTM
        self.question_lstm = nn.LSTM(self.embedding_dim, self.lstm_units)

    def forward(self, x, seq_lens):

        pass


class PhraseConvPool(nn.Module):
    """Implements Conv + Max-pool for Question phrases"""

    def __init__(self, emb_dim):
        super().__init__()
        self.conv_unigram = nn.Sequential(nn.ConstantPad1d((0, 0), 0), nn.Conv1d(emb_dim, emb_dim, 1, 1), nn.Tanh())
        self.conv_bigram = nn.Sequential(nn.ConstantPad1d((1, 0), 0), nn.Conv1d(emb_dim, emb_dim, 2, 1), nn.Tanh())
        self.conv_trigram = nn.Sequential(nn.ConstantPad1d((1, 1), 0), nn.Conv1d(emb_dim, emb_dim, 3, 1), nn.Tanh())

        # Max-Pool (kernel = 1x3) - subsample from n-gram representations of tokens)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 3))

    def forward(self, x_question):
        batch_size, emb_dim, seq_len = x_question.shape                 # [batch_size, emb_dim, max_seq_len]

        # Compute the n-gram phrase embeddings (n=1,2,3)
        x_uni = self.conv_unigram(x_question)
        x_bi = self.conv_bigram(x_question)
        x_tri = self.conv_trigram(x_question)

        # Concat
        x = torch.cat([x_uni, x_bi, x_tri], dim=1)                      # [batch_size, 3*emb_dim, max_seq_len]

        # Position the three n-gram representations along a new axis (for pooling)
        x = x.permute(0, 2, 1)                                          # [batch_size, max_seq_len, 3*emb_dim]
        x = x.unsqueeze(dim=3)                                          # [batch_size, max_seq_len, 3*emb_dim, 1]
        x = x.reshape([batch_size, seq_len, emb_dim, 3])                # [batch_size, max_seq_len, emb_dim, 3]

        # Max-pool across n-gram features
        x = self.max_pool(x).squeeze(dim=3)                             # [batch_size, max_seq_len, emb_dim]

        return x


class ParallelCoAttention(nn.Module):
    """
    Implements Parallel Co-Attention mechanism
    given image & question features.
    """
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass
