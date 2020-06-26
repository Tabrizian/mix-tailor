import torch

from torch.distributions.normal import Normal


class Attack:

    def __init__(self, num_byz, grad_size):
        self.grad_size = grad_size
        self.num_byz = num_byz

    def grad(self, grads):
        # assuming an omniscient attacker you have access to all the gradients
        # of the non-byzantine workers. attackers can collude with each other
        # by using dist.get_rank
        raise NotImplementedError


class Gaussian(Attack):

    def __init__(self, num_byz, grad_size, mean=0, cov=200):
        super().__init__(num_byz, grad_size)
        # specify means and covariance matrix
        self.dist = Normal(mean, cov)

    def grad(self, grads):
        return self.dist.sample([self.grad_size])


class Reverse(Attack):
    # This attack reverses the gradient direction

    def __init__(self, num_byz, grad_size):
        super().__init__(num_byz, grad_size)

    def grad(self, grads):
        num_byz = self.num_byz

        return -torch.sum(grads, dim=0) / num_byz


class EpsilonReverse(Attack):
    # This attack reverses the gradient direction with an epsilon value

    def __init__(self, num_byz, grad_size, epsilon):
        super().__init__(num_byz, grad_size)
        self._epsilon = epsilon

    def grad(self, grads):
        num_byz = self.num_byz
        epsilon = self._epsilon

        return -torch.sum(grads, dim=0) / num_byz * epsilon


class Mimic(Attack):
    # implementation of mimic attack as mentioned in section 3.3
    # https://arxiv.org/pdf/2006.09365.pdf

    def __init__(self, num_byz, grad_size):
        super().__init__(num_byz, grad_size)

    def grad(self, grads):
        return grads[0]


class BulyanAttack(Attack):
    # implementation of Bulyan attack as mentioned in section 3.2
    # http://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a.pdf

    def __init__(self, num_byz, grad_size):
        super().__init__(num_byz, grad_size)

    def grad(self, grads):
        agg = torch.sum(grads, dim=0) / len(grads)
        shape = agg.shape
        rand_index = torch.randint(0, shape[0], (1,))
        attack_grad = torch.zeros(shape)
        attack_grad[rand_index] = 1


class ALittle(Attack):

    def __init__(self, num_byz, grad_size, num_std):
        super().__init__(num_byz, grad_size)
        self.num_std = num_std

    def grad(self, grads):
        agg = torch.sum(grads, dim=0) / len(grads)
        var = torch.var(grads, axis=0) ** 0.5
        agg -= self.num_std * var
        return agg


class Adaptive(Attack):

    def __init__(self, num_byz, grad_size, attack_list, aggregator_list):
        super().__init__(num_byz, grad_size)
        self._attack_list = attack_list
        self._aggregator_list = aggregator_list

    def grad(self, grads):
        num_byz = self.num_byz
        true_grad = torch.sum(grads, dim=0) / len(grads)
        attack_list = []
        for aggregator in self._aggregator_list:
            dot_product_list = []
            attacks = []
            for attack in self._attack_list:
                grad_attack = attack.grad(grads)
                attacks.append(grad_attack)
                my_grads = [grads]
                for _ in range(num_byz):
                    my_grads.append(grad_attack[None, :])

                agg_grad = aggregator.agg_g(torch.cat(my_grads))

                # Calculate the dot product betwen the aggregated gradient and
                # true gradient. The attack that is able to fool the aggregator
                # most is the chosen attack.
                dot_product_list.append(torch.dot(agg_grad, true_grad))

            # Choose the vector with smallest dot product (i.e. the attack
            # vector that works best against the aggregator)
            index = torch.argmin(torch.tensor(dot_product_list))
            attack_list.append(attacks[index])

        return -torch.sum(torch.stack(attack_list), dim=0) / num_byz
