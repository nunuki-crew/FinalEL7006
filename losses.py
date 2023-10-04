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