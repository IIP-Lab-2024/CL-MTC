from transformers import BertModel, AutoModel
import torch.nn.functional as F
import torch.nn as nn
import torch
import transformers 

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # revise to translate contrastive loss, where the current sentence itself and its translation is 1.
            mask = torch.eye(int(batch_size / 2), dtype=torch.float32)
            mask = torch.cat([mask, mask], dim=1)
            mask = torch.cat([mask, mask], dim=0)
            mask = mask.to(device)

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            # print(mask)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask + 1e-20
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print(exp_logits.shape)
        # print(log_prob.shape)
        # compute mean of log-likelihood over positive
        # DONE: I modified here to prevent nan
        mean_log_prob_pos = ((mask * log_prob).sum(1) + 1e-20) / (mask.sum(1) + 1e-20)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # this would occur nan, I think we can divide then sum
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class SupConLossMulti(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLossMulti, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, weights=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # revise to translate contrastive loss, where the current sentence itself and its translation is 1.
            mask = torch.eye(int(batch_size / 2), dtype=torch.float32)
            mask = torch.cat([mask, mask], dim=1)
            mask = torch.cat([mask, mask], dim=0)
            mask = mask.to(device)

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            # print(mask)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask + 1e-20
        exp_logits = torch.mul(exp_logits,weights)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        # DONE: I modified here to prevent nan
        mean_log_prob_pos = ((mask * log_prob).sum(1) + 1e-20) / (mask.sum(1) + 1e-20)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # this would occur nan, I think we can divide then sum
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class BertEncoder(nn.Module):
    def __init__(self, lang='English'):
        """
        :param lang: str, train bert encoder for a given language
        """
        super(BertEncoder, self).__init__()
        if lang == 'English':
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        elif lang == 'Arabic':
            self.bert = AutoModel.from_pretrained("asafaya/bert-base-arabic")
        elif lang == 'Spanish':
            self.bert = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
        elif lang == 'Indonesian':
            self.bert = AutoModel.from_pretrained("cahya/bert-base-indonesian-1.5G")
        self.feature_size = self.bert.config.hidden_size

    def forward(self, input_ids):
        """
        :param input_ids: list[str], list of tokenised sentences
        :return: last hidden representation, torch.tensor of shape (batch_size, seq_length, hidden_dim)
        """
        if int((transformers.__version__)[0]) == 4:
            last_hidden_state = self.bert(input_ids=input_ids).last_hidden_state
        else: #transformers version should be as indicated in the requirements.txt file
            last_hidden_state, pooler_output = self.bert(input_ids=input_ids)
        return last_hidden_state


class SpanEmo(nn.Module):
    def __init__(self, output_dropout=0.1, lang='English', joint_loss='joint', alpha=0.2, beta=0.0, gamma=0.0, temperature1=0.2, temperature2=0.2):
        """ casting multi-label emotion classification as span-extraction
        :param output_dropout: The dropout probability for output layer
        :param lang: encoder language
        :param joint_loss: which loss to use cel|corr|cel+corr
        :param alpha: control contribution of each loss function in case of joint training
        """
        super(SpanEmo, self).__init__()
        self.bert = BertEncoder(lang=lang)
        self.joint_loss = joint_loss
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature1 = temperature1
        self.temperature2 = temperature2
        self.ffn = nn.Sequential(
            nn.Linear(self.bert.feature_size, self.bert.feature_size),
            nn.Tanh(),
            nn.Dropout(p=output_dropout),
            nn.Linear(self.bert.feature_size, 1)
        )

    def forward(self, batch, device):
        """
        :param batch: tuple of (input_ids, labels, length, label_indices)
        :param device: device to run calculations on
        :return: loss, num_rows, y_pred, targets
        """
        #prepare inputs and targets
        inputs, targets, lengths, label_idxs = batch
        inputs, num_rows = inputs.to(device), inputs.size(0)
        label_idxs, targets = label_idxs[0].long().to(device), targets.float().to(device)

        #Bert encoder
        last_hidden_state = self.bert(inputs)

        # FFN---> 2 linear layers---> linear layer + tanh---> linear layer
        # select span of labels to compare them with ground truth ones
        # [32, 11]
        logits = self.ffn(last_hidden_state).squeeze(-1).index_select(dim=1, index=label_idxs)
        # [32, 11, 768]
        labels_embeddings = last_hidden_state.index_select(dim=1, index=label_idxs)
        # [32, 768]

        sentences_embeddings = last_hidden_state[:, 0, :]
        sentences_embeddings = sentences_embeddings.unsqueeze(1)


        #Loss Function
        if self.joint_loss == 'joint':
            # print(targets.shape)
            cel = F.binary_cross_entropy_with_logits(logits, targets).cuda()
            cl = self.corr_loss(logits, targets)
            loss = ((1 - self.alpha) * cel) + (self.alpha * cl)
        elif self.joint_loss == 'cross-entropy':
            loss = F.binary_cross_entropy_with_logits(logits, targets).cuda()
        elif self.joint_loss == 'corr_loss':
            loss = self.corr_loss(logits, targets)
        elif self.joint_loss == 'SLCL':
            loss_sum = 0
            for i in range(logits.shape[1]):
                sloss = SupConLoss(contrast_mode='all', temperature=self.temperature).to('cuda')
                one_sentiment_labels = targets[:, i]
                sloss = sloss(sentences_embeddings, one_sentiment_labels)
                sloss = sloss / len(sentences_embeddings)
                loss_sum += sloss
            loss_sum = loss_sum / logits.shape[1]
            loss = F.binary_cross_entropy_with_logits(logits, targets).cuda()
            loss = (1-self.alpha) * loss + self.alpha * loss_sum
        elif self.joint_loss == 'JSCL':
            weights = []
            for num_1, target1 in enumerate(targets):
                weight_tmp = []
                for num_2, target2 in enumerate(targets):
                    tmp = torch.add(target1, target2).cpu().numpy().tolist()
                    if (tmp.count(2) + tmp.count(1)) == 0:
                        weight_tmp.append(0)
                    else:
                        weight_tmp.append(
                            1.0 * tmp.count(2) / (tmp.count(2) + tmp.count(1))
                        )
                weights.append(weight_tmp)
            weights = torch.tensor(weights).to('cuda')
            mask = torch.ones(targets.shape[0],targets.shape[0]).to('cuda')
            sloss = SupConLossMulti(contrast_mode='all', temperature=self.temperature).to('cuda')
            sloss = sloss(sentences_embeddings,mask=mask, weights=weights)
            loss = F.binary_cross_entropy_with_logits(logits, targets).cuda()
            loss = (1-self.alpha) * loss + self.alpha * sloss
        elif self.joint_loss == 'SCL':
            mask = []
            for num_1, target1 in enumerate(targets):
                mask_tmp = []
                for num_2, target2 in enumerate(targets):
                    tmp = torch.add(target1, target2).cpu().numpy().tolist()
                    if tmp.count(1) == 0:
                        mask_tmp.append(1)
                    else:
                        mask_tmp.append(0)
                mask.append(mask_tmp)
            mask = torch.tensor(mask).to('cuda')
            sloss = SupConLoss(contrast_mode='all', temperature=self.temperature).to('cuda')
            sloss = sloss(sentences_embeddings,mask=mask)
            loss = F.binary_cross_entropy_with_logits(logits, targets).cuda()
            loss = (1-self.alpha) * loss + self.alpha * sloss

        elif self.joint_loss == 'JSPCL':
            weights = []
            for num_1, target1 in enumerate(targets):
                weight_tmp = []
                for num_2, target2 in enumerate(targets):
                    tmp = torch.add(target1, target2).cpu().numpy().tolist()
                    if (tmp.count(2) + tmp.count(1)) == 0:
                        weight_tmp.append(0)
                    else:
                        weight_tmp.append(
                            1.0 * tmp.count(2) / (tmp.count(2) + tmp.count(1))
                        )
                weights.append(weight_tmp)
            weights = torch.tensor(weights).to('cuda')
            mask = torch.ones(targets.shape[0],targets.shape[0]).to('cuda')
            sloss = SupConLossMulti(contrast_mode='all', temperature=self.temperature).to('cuda')
            sloss = sloss(logits.unsqueeze(1),mask=mask, weights=weights)
            loss = F.binary_cross_entropy_with_logits(logits, targets).cuda()
            loss = (1-self.alpha) * loss + self.alpha * sloss

        elif self.joint_loss == 'ICL':
            loss_sum = 0
            for i in range(labels_embeddings.shape[0]):
                sloss = SupConLoss(contrast_mode='all', temperature=self.temperature).to('cuda')
                sample_embeddings = labels_embeddings[i, :, :]
                sample_embeddings = sample_embeddings.unsqueeze(1)
                one_sentiment_labels = targets[i,:]
                sloss = sloss(sample_embeddings, one_sentiment_labels)
                sloss = sloss / len(sample_embeddings)
                loss_sum += sloss
            loss_sum = loss_sum / labels_embeddings.shape[0]
            loss = F.binary_cross_entropy_with_logits(logits, targets).cuda()
            loss = (1-self.alpha) * loss + self.alpha * loss_sum

        y_pred = self.compute_pred(logits)
        return loss, num_rows, y_pred, targets.cpu().numpy(),sentences_embeddings

    @staticmethod
    def corr_loss(y_hat, y_true, reduction='mean'):
        """
        :param y_hat: model predictions, shape(batch, classes)
        :param y_true: target labels (batch, classes)
        :param reduction: whether to avg or sum loss
        :return: loss
        """
        loss = torch.zeros(y_true.size(0)).cuda()
        for idx, (y, y_h) in enumerate(zip(y_true, y_hat.sigmoid())):
            y_z, y_o = (y == 0).nonzero(), y.nonzero()
            if y_o.nelement() != 0:
                output = torch.exp(torch.sub(y_h[y_z], y_h[y_o][:, None]).squeeze(-1)).sum()
                num_comparisons = y_z.size(0) * y_o.size(0)
                loss[idx] = output.div(num_comparisons)
        return loss.mean() if reduction == 'mean' else loss.sum()
        
    @staticmethod
    def compute_pred(logits, threshold=0.5):
        """
        :param logits: model predictions
        :param threshold: threshold value
        :return:
        """
        y_pred = torch.sigmoid(logits) > threshold
        return y_pred.float().cpu().numpy()
