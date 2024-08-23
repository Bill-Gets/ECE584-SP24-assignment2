import torch
import torch.nn as nn


class BoundHardTanh(nn.Hardtanh):
    def __init__(self):
        super(BoundHardTanh, self).__init__()
        self.upper_u:torch.Tensor = None
        self.lower_l:torch.Tensor = None

    @staticmethod
    def convert(act_layer):
        r"""Convert a HardTanh layer to BoundHardTanh layer

        Args:
            act_layer (nn.HardTanh): The HardTanh layer object to be converted.

        Returns:
            l (BoundHardTanh): The converted layer object.
        """
        # TODO: Return the converted HardTanH
        l = BoundHardTanh()
        return l

    def boundpropogate(self, last_uA, last_lA, start_node=None):
        """
        Propagate upper and lower linear bounds through the HardTannh activation function
        based on pre-activation bounds.

        Args:
            last_uA (tensor): A (the coefficient matrix) that is bound-propagated to this layer
            (from the layers after this layer). It's exclusive for computing the upper bound.

            last_lA (tensor): A that is bound-propagated to this layer. It's exclusive for computing the lower bound.

            start_node (int): An integer indicating the start node of this bound propagation

        Returns:
            uA (tensor): The new A for computing the upper bound after taking this layer into account.

            ubias (tensor): The bias (for upper bound) produced by this layer.

            lA( tensor): The new A for computing the lower bound after taking this layer into account.

            lbias (tensor): The bias (for lower bound) produced by this layer.

        """
        # These are preactivation bounds that will be used for form the linear relaxation.
        preact_lb = self.lower_l
        preact_ub = self.upper_u

        """
         Hints: 
         1. Have a look at the section 3.2 of the CROWN paper [1] (Case Studies) as to how segments are made for multiple activation functions
         2. Look at the HardTanH graph, and see multiple places where the pre activation bounds could be located
         3. Refer the ReLu example in the class and the diagonals to compute the slopes/intercepts
         4. The paper talks about 3 segments S+, S- and S+- for sigmoid and tanh. You should figure your own segments based on preactivation bounds for hardtanh.
         [1] https://arxiv.org/pdf/1811.00866.pdf
        """

        # You should return the linear lower and upper bounds after propagating through this layer.
        # Upper bound: uA is the coefficients, ubias is the bias.
        # Lower bound: lA is the coefficients, lbias is the bias.

        hatz_ub, hatz_lb = preact_ub.clamp(min = -1, max = 1), preact_lb.clamp(min = -1, max = 1)
        upper_d_max = (hatz_ub - hatz_lb) / (hatz_ub - preact_lb).clamp(min = 1e-8)  # max slope; avoid division by 0
        lower_d_max = (hatz_ub - hatz_lb) / (preact_ub - hatz_lb).clamp(min = 1e-8)  # max slope; avoid division by 0
        upper_d = torch.where(preact_ub + preact_lb <= 2, upper_d_max, 0)  # slope
        lower_d = torch.where(preact_ub + preact_lb >= -2, lower_d_max, 0)   # slope
        upper_b = hatz_ub - upper_d * hatz_ub  # intercept
        lower_b = hatz_lb - lower_d * hatz_lb  # intercept

        uA = lA = None
        ubias = lbias = 0

        pos_uA, neg_uA = last_uA.clamp(min=0), last_uA.clamp(max=0)
        pos_lA, neg_lA = last_lA.clamp(min=0), last_lA.clamp(max=0)
        mult_uA = pos_uA.view(last_uA.size(0), last_uA.size(1), -1)
        mult_lA = neg_lA.view(last_lA.size(0), last_lA.size(1), -1)
        
        if last_uA is not None:
            uA = upper_d * pos_uA + lower_d * neg_uA
            ubias1 = mult_uA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
            ubias2 = mult_lA.matmul(lower_b.view(lower_b.size(0), -1, 1)).squeeze(-1)
            ubias = ubias1 + ubias2

        if last_lA is not None:
            lA = upper_d * neg_lA + lower_d * pos_lA
            lbias1 = mult_uA.matmul(lower_b.view(lower_b.size(0), -1, 1)).squeeze(-1)
            lbias2 = mult_lA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
            lbias = lbias1 + lbias2

        return uA, ubias, lA, lbias