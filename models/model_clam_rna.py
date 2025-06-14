import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes
"""
class Attn_Net_RNA(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_RNA, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes
"""
class Attn_Net_Gated_RNA(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated_RNA, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    n_classes: number of classes
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
    embed_dim: dimension of the input features (concatenated: original_patch_feature_dim + master_rna_dim)
"""
class CLAM_SB_RNA(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=20968): # Default embed_dim updated
        super(CLAM_SB_RNA, self).__init__()
        # embed_dim is the concatenated feature size: original_patch_feature_dim + master_rna_dim
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        
        # The first linear layer (size[0]) uses the new embed_dim
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        
        if gate:
            attention_net = Attn_Net_Gated_RNA(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net_RNA(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.embed_dim = embed_dim


    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()

    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample, dim=1)[1][-1] # Ensure dim is correct for A (KxN or 1xN) -> dim=1
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1] # Ensure dim is correct
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample, dim=1)[1][-1] # Ensure dim is correct
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        # h is now the concatenated features [N, embed_dim]
        A, h_transformed = self.attention_net(h)  # A: NxK, h_transformed: NxL' (output of first Linear layer)
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                # Pass h_transformed (features after first linear layer) to inst_eval
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h_transformed, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h_transformed, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping and len(self.instance_classifiers) > 0 : # Avoid division by zero if no instance classifiers used
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h_transformed) # M: KxL'
        logits = self.classifiers(M) # Input to classifier is L'
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

class CLAM_MB_RNA(CLAM_SB_RNA): # Inherits from CLAM_SB_RNA to get most functionality
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=20968): # Default embed_dim updated
        # Call CLAM_SB_RNA's __init__ but Attn_Net will have n_classes outputs
        # We will then override parts of it.
        # The CLAM_SB_RNA init sets up self.attention_net, instance_classifiers etc.
        # The key difference for CLAM_MB is how attention_net and classifiers are defined.
        
        # Explicitly call nn.Module's init
        nn.Module.__init__(self)

        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        self.embed_dim = embed_dim # Store embed_dim

        # fc part is the same, first Linear layer uses the new embed_dim
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        
        if gate:
            # For MB, Attn_Net_Gated_RNA has n_classes output channels for attention scores
            attention_net = Attn_Net_Gated_RNA(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            # For MB, Attn_Net_RNA has n_classes output channels
            attention_net = Attn_Net_RNA(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        # CLAM_MB uses independent linear layers for each class for bag classification
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)
        
        # Instance classifiers are the same as CLAM_SB
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping


    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        # h is now the concatenated features [N, embed_dim]
        A, h_transformed = self.attention_net(h)  # A: NxK (K is n_classes here), h_transformed: NxL'
        A = torch.transpose(A, 1, 0)  # KxN (K is n_classes)
        
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N for each of the K attention heads

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)): # Loop over n_classes
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                # For CLAM_MB, A[i] is the attention score for the i-th class (1xN)
                # Pass h_transformed (features after first linear layer) to inst_eval
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i].unsqueeze(0), h_transformed, classifier) # A[i] is 1D, unsqueeze
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i].unsqueeze(0), h_transformed, classifier) # A[i] is 1D, unsqueeze
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping and len(self.instance_classifiers) > 0: # Avoid division by zero
                total_inst_loss /= len(self.instance_classifiers)

        # M = torch.mm(A, h) # Original CLAM_SB was A (1xN) x h (NxL') -> M (1xL')
        # For CLAM_MB, A is KxN (K=n_classes), h_transformed is NxL'. So M is KxL'
        M = torch.mm(A, h_transformed) # M: KxL' (K=n_classes)

        # Logits are calculated per class using respective classifier
        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c].unsqueeze(0)) # M[c] is L', unsqueeze to 1xL'

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M}) # M is KxL'
        return logits, Y_prob, Y_hat, A_raw, results_dict