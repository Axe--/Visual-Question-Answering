import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
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

    def __init__(self, vocab_size, word_emb_dim, hidden_dim):
        super(QuestionBaselineEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.word_emb_dim = word_emb_dim

        # Input word embedding lookup matrix
        self.word_embedding = nn.Sequential(nn.Embedding(self.vocab_size, self.word_emb_dim), nn.Tanh())

        # RNN Layer: (input_dim, hidden_units)
        self.gru = nn.GRU(self.word_emb_dim, self.hidden_dim)

        # Question embedding layer (1024-dim)
        self.embedding_layer = nn.Sequential(nn.Linear(self.hidden_dim, 1024),
                                             nn.Tanh())

    def forward(self, x, seq_lengths):
        """
        Performs forward pass on question sequence

        :param x: input tensor [batch_size, seq_len]
        :param seq_lengths: corresponding sequence lengths [batch_size]
        :return: output embedding tensor [batch_size, hidden_dim]
                | `after torch.squeeze(dim=0)`
        """
        x = self.word_embedding(x)                  # [batch_size, seq_len, word_emb_dim]

        # By default, hidden (& cell) are the final state in the sequence, viz. mostly pad token.
        # `PackedSequence` selects the last 'non-pad' element in the sequence.
        x = pack_padded_sequence(x, seq_lengths, batch_first=True)

        # outputs: PackedSequence, hidden: tensor
        outputs, hidden = self.gru(x)               # outputs: [sum_{i=0}^batch (seq_lengths[i]), hidden_dim]

        hidden = torch.squeeze(hidden, dim=0)       # hidden: [1, batch_size, hidden_dim] --> [batch_size, hidden_dim]

        # Map the final hidden state to 1024-dim (image-question joint space)
        x_emb = self.embedding_layer(hidden)        # [batch_size, 1024]

        return x_emb


# ************************************************************************************************


class HierarchicalCoAttentionNet(nn.Module):
    """Hierarchical Co-Attention Architecture"""

    def __init__(self, ques_enc_params, img_enc_params, K, mlp_dim=1024):
        super().__init__()
        self.hidden_dim = ques_enc_params['hidden_dim']

        self.image_encoder = ImageCoAttentionEncoder(**img_enc_params)
        self.question_encoder = QuestionCoAttentionEncoder(**ques_enc_params)

        self.co_attention = ParallelCoAttention(self.hidden_dim)

        self.mlp_classify = MLPClassifier(self.hidden_dim, mlp_dim, K)

    def forward(self, x_img, x_ques, x_ques_lens):
        # Word, Phrase & Sentence features
        x_word, x_phrase, x_sentence = self.question_encoder(x_ques, x_ques_lens)   # [batch, max_seq_len, hidden_dim]

        # Question Features ([word, phrase, sentence])
        x_ques_features = [x_word, x_phrase, x_sentence]                            # 3*[batch, max_seq_len, hidden_dim]

        # Image Features
        x_img_features = self.image_encoder(x_img)                                  # [batch, spatial_locs, hidden_dim]

        # Co-Attention - Attention weighted image & question features (at all 3 levels)
        x_img_attn, x_ques_attn = self.co_attention(x_img_features, x_ques_features)    # 3*[B, hid_dim], 3*[B, hid_dim]

        # Predict Answer (logits)
        x_logits = self.mlp_classify(x_img_attn, x_ques_attn)                       # [batch_size, K]

        return x_logits


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

        x_feat = x_feat.permute(0, 2, 1)                            # [batch_size, spatial_locs, 512]

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
    def __init__(self, vocab_size, word_emb_dim, hidden_dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = word_emb_dim
        self.hidden_dim = hidden_dim

        # Word Embedding matrix
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)     # {<PAD> : 0}

        # Phrase Convolution + MaxPool
        self.phrase_conv_pool = PhraseConvPool(self.embedding_dim)

        # Sentence LSTM
        self.sentence_lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)

    def forward(self, x, x_lens):
        """
        Forward pass to compute the word, phrase & sentence level representations.

        :param x: question token sequence [batch_size, max_seq_len]
        :param x_lens: actual sequence length of the corresponding question [batch_size]
        :returns: word, phrase & sentence vectors;  3 * [batch_size, max_seq_len, hidden_dim]
        """
        # max sequence length (in the dataset); for padding PackedSequence
        max_seq_len = x.shape[1]

        x_word_emb = self.word_embedding(x)                             # [batch_size, max_seq_len, emb_dim]

        x_phrase_emb = self.phrase_conv_pool(x_word_emb)                # [batch_size, max_seq_len, emb_dim]

        # Pack the padded input
        x_phrase_emb = pack_padded_sequence(x_phrase_emb, x_lens, batch_first=True)

        x_sentence_emb, last_state = self.sentence_lstm(x_phrase_emb)

        # Un-pack (Pad) the packed phrase & sentence feature sequences
        x_phrase_emb = pad_packed_sequence(x_phrase_emb, batch_first=True,
                                           total_length=max_seq_len)[0]

        x_sentence_emb = pad_packed_sequence(x_sentence_emb, batch_first=True,
                                             total_length=max_seq_len)[0]

        return x_word_emb, x_phrase_emb, x_sentence_emb                 # 3*[batch_size, max_seq_len, hidden_dim]


class PhraseConvPool(nn.Module):
    """Implements Conv + Max-pool for Question phrases"""

    def __init__(self, emb_dim):
        super().__init__()
        self.conv_unigram = nn.Sequential(nn.ConstantPad1d((0, 0), 0), nn.Conv1d(emb_dim, emb_dim, 1, 1), nn.Tanh())
        self.conv_bigram  = nn.Sequential(nn.ConstantPad1d((1, 0), 0), nn.Conv1d(emb_dim, emb_dim, 2, 1), nn.Tanh())
        self.conv_trigram = nn.Sequential(nn.ConstantPad1d((1, 1), 0), nn.Conv1d(emb_dim, emb_dim, 3, 1), nn.Tanh())

        # Max-Pool (kernel = 1x3) - subsample from n-gram representations of tokens)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 3))

    def forward(self, x_question):
        batch_size, max_seq_len, emb_dim = x_question.shape             # [batch_size, max_seq_len, emb_dim]

        x_question = x_question.permute(0, 2, 1)                        # [batch_size, emb_dim, max_seq_len]

        # Compute the n-gram phrase embeddings (n=1,2,3)
        x_uni = self.conv_unigram(x_question)
        x_bi = self.conv_bigram(x_question)
        x_tri = self.conv_trigram(x_question)

        # Concat
        x = torch.cat([x_uni, x_bi, x_tri], dim=1)                      # [batch_size, 3*emb_dim, max_seq_len]

        # Position the three n-gram representations along a new axis (for pooling)
        x = x.permute(0, 2, 1)                                          # [batch_size, max_seq_len, 3*emb_dim]
        x = x.unsqueeze(dim=3)                                          # [batch_size, max_seq_len, 3*emb_dim, 1]
        x = x.reshape([batch_size, max_seq_len, emb_dim, 3])            # [batch_size, max_seq_len, emb_dim, 3]

        # Max-pool across n-gram features
        x = self.max_pool(x).squeeze(dim=3)                             # [batch_size, max_seq_len, emb_dim]

        return x


class ParallelCoAttention(nn.Module):
    """
    Implements Parallel Co-Attention mechanism
    given image & question features.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Affinity layer
        self.W_b = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Attention layers
        self.W_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_q = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.w_v = nn.Linear(self.hidden_dim, 1)
        self.w_q = nn.Linear(self.hidden_dim, 1)

    def forward(self, x_img, x_ques_hierarchy):
        """
        Given image & question features, for all three levels in the question hierarchy,
        computes the attention-weighted image & question features.

        :param Tensor x_img: image features (flattened map)  [batch_size, spatial_locs, 512]
        :param list x_ques_hierarchy: question features list(word, phrase, sentence)  3*[batch_size, max_seq_len, 512]

        :returns: image-attention                   3*[batch_size, 512],
                  question-attention features       3*[batch_size, 512]
        :rtype: (list, list)
        """
        # For all feature levels of the question hierarchy, compute the image & question features
        img_feats = []
        quest_feats = []

        for x_ques in x_ques_hierarchy:
            Q = x_ques                                              # [batch_size, max_seq_len, hidden_dim]
            V = x_img.permute(0, 2, 1)                              # [batch_size,  hidden_dim, spatial_locs]

            # Affinity matrix
            C = F.tanh(torch.bmm(Q, V))                             # [batch_size, max_seq_len, spatial_locs]
            V = V.permute(0, 2, 1)                                  # [batch_size, spatial_locs, hidden_dim]

            H_v = F.tanh(self.W_v(V) +                              # [batch_size, spatial_locs, hidden_dim]
                         torch.bmm(C.transpose(2, 1), self.W_q(Q)))

            H_q = F.tanh(self.W_q(Q) +                              # [batch_size, max_seq_len, hidden_dim]
                         torch.bmm(C, self.W_v(V)))

            # Attention weights
            a_v = F.softmax(self.w_v(H_v), dim=1)                   # [batch_size, spatial_locs, 1]
            a_q = F.softmax(self.w_q(H_q), dim=1)                   # [batch_size, max_seq_len, 1]

            # Compute attention-weighted features
            v = torch.sum(a_v * V, dim=1)                           # [batch_size, hidden_dim]
            q = torch.sum(a_q * Q, dim=1)                           # [batch_size, hidden_dim]

            img_feats.append(v)
            quest_feats.append(q)

        return img_feats, quest_feats                               # 3*[batch, hidden_dim], 3*[batch, hidden_dim]


class MLPClassifier(nn.Module):
    """
    Implements the MLP module for classifying
    answers, given the hierarchical attention-weighted
    image & question features.
    """
    def __init__(self, hidden_dim, mlp_dim, K):
        super().__init__()

        self.W_w = nn.Linear(hidden_dim, hidden_dim)
        self.W_p = nn.Linear(2*hidden_dim, hidden_dim)
        self.W_s = nn.Linear(2*hidden_dim, mlp_dim)
        self.W_h = nn.Linear(mlp_dim, K)

    def forward(self, x_img_feats, x_ques_feats):
        """
        Recursively encode the image & question features
        across the three levels.

        :param list x_img_feats: attention-weighted image representation         # 3*[B, hidden_dim]
        :param list x_ques_feats: attention-weighted question representation     # 3*[B, hidden_dim]

        :return: logit (class prediction)  [batch_size, K]
        """
        q_w, q_p, q_s = x_ques_feats                                    # [batch_size, hidden_dim]
        v_w, v_p, v_s = x_img_feats                                     # [batch_size, hidden_dim]

        h_w = F.tanh(self.W_w(q_w + v_w))                               # [batch_size, hidden_dim]
        h_p = F.tanh(self.W_p(torch.cat([q_p + v_p, h_w], dim=1)))      # [batch_size, hidden_dim]
        h_s = F.tanh(self.W_s(torch.cat([q_s + v_s, h_p], dim=1)))      # [batch_size, mlp_dim]

        # Final answer (classification logit)
        logit = self.W_h(h_s)                                           # [batch_size, K]

        return logit
