from torch.nn import functional as F
import torch


class InfoNceLoss(torch.nn.Module):

    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau
    
    def calc_similarity_batch(self, a, b):
        """_summary_

        Args:
            a (_type_): _description_
            b (_type_): _description_

        Returns:
            _type_: _description_
        """
        features = torch.cat([a, b], dim=0)
        return F.cosine_similarity(
            features.unsqueeze(1), features.unsqueeze(0), dim=2
        )

    def forward(self, p1, p2):
        """_summary_

        Args:
            p1 (_type_): _description_
            p2 (_type_): _description_

        Returns:
            _type_: _description_
        """

        z_i, z_j = F.normalize(p1, dim=1), F.normalize(p2, dim=1)
        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, len(p1))
        sim_ji = torch.diag(similarity_matrix, -len(p1))

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.tau)
        mask = (
            ~torch.eye(len(p1) * 2, len(p1) * 2, dtype=bool)
        ).float().to(p1.device)

        denominator = mask* torch.exp(similarity_matrix / self.tau)
        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))

        return torch.sum(all_losses) / (2 * len(p1))
    
    def __str__(self):
        return 'InfoNCELoss/SimCLR'


class IsoLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.criteria = torch.nn.CosineSimilarity(dim=1)
    
    def forward(self, hi, hj, pi):
        return self.criteria(hi, (hj + hi - pi).detach()).mean()
    
    def __str__(self):
        return 'IsoLoss'
    
class BarlowTwins(torch.nn.Module):

    def __init__(self, lda=0.1, noise = False):
        super().__init__()
        self._lambda = lda
        self.noise = noise
    @staticmethod
    def off_diagonal(x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        n, m = x.shape
        assert n == m

        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, p1, p2):
        """_summary_

        Args:
            p1 (_type_): _description_
            p2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        z_i, z_j = F.normalize(p1, dim=1), F.normalize(p2, dim=1)
        
        corr = (z_i.T @ z_j).div_(len(p1))

        on_diag = torch.diagonal(corr).add_(-1).pow_(2).sum()
        
        if self.noise:
            noise = torch.randn(1)
            off_diag = self.off_diagonal(corr).add_(noise).pow_(2).sum()
        else:
            off_diag = self.off_diagonal(corr).pow_(2).sum()

        loss = on_diag + self._lambda * off_diag
        
        return loss


    def __str__(self):
        return 'BarlowTwins'


class VICReg(torch.nn.Module):

    def __init__(self, _conf):
        super().__init__()

        self.variance_loss_epsilon = 1e-04
        
        self.invariance_loss_weight = _conf['invariance_loss_weight']
        self.variance_loss_weight = _conf['variance_loss_weight']
        self.covariance_loss_weight = _conf['covariance_loss_weight']

    def forward(self, z_a, z_b):
        """_summary_

        Args:
            z_a (_type_): _description_
            z_b (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        loss_inv = F.mse_loss(z_a, z_b)

        std_z_a = torch.sqrt(
            z_a.var(dim=0) + self.variance_loss_epsilon
        )
        std_z_b = torch.sqrt(
            z_b.var(dim=0) + self.variance_loss_epsilon
        )
        loss_v_a = torch.mean(F.relu(1 - std_z_a))
        loss_v_b = torch.mean(F.relu(1 - std_z_b))
        loss_var = loss_v_a + loss_v_b

        N, D = z_a.shape

        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)

        cov_z_a = ((z_a.T @ z_a) / (N - 1)).square()  # DxD
        cov_z_b = ((z_b.T @ z_b) / (N - 1)).square()  # DxD
        loss_c_a = (cov_z_a.sum() - cov_z_a.diagonal().sum()) / D
        loss_c_b = (cov_z_b.sum() - cov_z_b.diagonal().sum()) / D
        loss_cov = loss_c_a + loss_c_b

        weighted_inv = loss_inv * self.invariance_loss_weight
        weighted_var = loss_var * self.variance_loss_weight
        weighted_cov = loss_cov * self.covariance_loss_weight

        loss = weighted_inv + weighted_var + weighted_cov

        return loss.mean()