import torch.nn as nn
import torch.nn.functional as F

class SModel(nn.Module):
  def __init__(self, encoder, num_labels):
    super().__init__()
    self.num_labels = num_labels
    self.encoder = encoder
    self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
    self.classifer = nn.Linear(self.encoder.config.hidden_size, self.num_labels)
    #self.encoder.init_weights()

  def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, e_span=None):
    output = self.encoder(input_ids, token_type_ids, attention_mask)
    sequence_output = output[0]
    batchsize, _, _ = sequence_output.size()
    batch_index = [i for i in range(batchsize)]
    repr = sequence_output[batch_index, e_span]

    cls_token = self.dropout(repr)
    logits = self.classifer(cls_token)
    avg_loss = F.cross_entropy(logits, labels)
    return avg_loss, logits
