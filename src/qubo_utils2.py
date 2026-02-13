from typing import List, Callable
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from neal import SimulatedAnnealingSampler
import dimod


class QuboXFeatureSelection():

    def __init__(self):
        self.features = None
        self.gradients = None
        self.input_tensor = None
        self.model = None
        self.target_layer = None
        self.target_module_id = None

    def set_hyperparameters(self, alpha, cosine_similarity, penalty_lambda=5, num_reads=100, n_features_constraint=3,
                            beta=0.5, only_positive=False, similarity_condition=True):
        """
        Set the hyperparameters for the QUBO problem.
        :param alpha: List of weights for each node.
        :param cosine_similarity: Function to compute cosine similarity between features.
        :param penalty_lambda: Penalty term for the QUBO problem.
        :param num_reads: Number of reads for the QUBO solver.
        """

        if only_positive:
            self.n_features = alpha[alpha > 0].shape[0]  # only positive gradients
            if self.n_features == 0:
                raise ValueError("No positive features found in alpha.")

            alpha[alpha <= 0] = 0.
            self.idxs_alpha_nozero = np.nonzero(alpha)[0]
            print('Non zero indices in alpha=', self.idxs_alpha_nozero)
            self.alpha = alpha[self.idxs_alpha_nozero]

        else:
            self.n_features = alpha.shape[0]  # only positive gradients
            if self.n_features == 0:
                raise ValueError("No positive features found in alpha.")

            self.idxs_alpha_nozero = np.nonzero(alpha)[0]
            self.alpha = alpha[self.idxs_alpha_nozero]

        #### let's normalize the alpha
        self.alpha = (self.alpha - self.alpha.min()) / (self.alpha.max() - self.alpha.min())

        # hyperparameters for the qubo
        self.penalty_lambda = penalty_lambda
        self.num_reads = num_reads
        self.beta = beta
        self.n_features_constraint = n_features_constraint

        self.cosine_similarity = cosine_similarity[self.idxs_alpha_nozero[:, None], self.idxs_alpha_nozero[None, :]]
        if similarity_condition:
            self.sign = -1
        else:
            self.sign = +1

    def __energy_functional_without_qubo__(self, x, alpha, cov, beta, lambd, n_tot):
        x = np.array(x)
        sum_x = np.sum(x)
        linear_obj = -beta * np.dot(alpha, x)
        quad_obj = (1 - beta) * np.dot(x, np.dot(cov, x))  # Avoid division by zero

        penalty = lambd * ((sum_x) ** 2 - 2 * n_tot * sum_x + n_tot ** 2)

        return linear_obj + quad_obj + penalty

    def __brute_force_algorithm__(self, plot_energy_spectrum=False):

        # Brute-force all 2^n binary combinations
        best_x = None
        best_energy = float('inf')

        energy_spectrum = []
        for x in product([0, 1], repeat=self.n_features):
            energy = self.__energy_functional_without_qubo__(x, self.alpha, self.cosine_similarity, self.beta,
                                                             self.penalty_lambda, self.n_features_constraint)
            energy_spectrum.append(energy)
            if energy <= best_energy:
                best_energy = energy
                best_x = x

        # plt.plot(energy_spectrum)
        if plot_energy_spectrum:
            plt.hist(energy_spectrum, bins=300, alpha=0.7)
            plt.title("Energy Spectrum")
            plt.xlabel("Energy")
            plt.ylabel("Frequency")
            plt.show()
            print("Best solution:", best_x)
            print("Minimum energy:", best_energy)

        best_idxs = np.nonzero(best_x)[0]

        print("Best Indices:", self.idxs_alpha_nozero[best_idxs], '\n')

        return best_idxs

    def __simulated_annealing__(self):
        Q = self.__build_qubo_with_cardinality__(self.alpha, self.cosine_similarity, self.beta, self.penalty_lambda,
                                                 self.n_features_constraint, self.n_features)
        # Use Simulated Annealing
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample_qubo(Q, num_reads=self.num_reads)

        best_x = sampleset.first.sample
        best_x = np.array([best_x[i] for i in range(self.n_features)])
        print("Energy:", sampleset.first.energy)

        best_idxs = np.nonzero(best_x)[0]

        print("Best Indices:", self.idxs_alpha_nozero[best_idxs], '\n')

        return best_idxs

    def __build_qubo_with_cardinality__(self, alpha, cov, beta, lambd, n_tot, n_features=64):

        Q = {}

        # Linear terms (diagonal) with penalty
        for r in range(n_features):
            linear_penalty = lambd * (1 - 2 * n_tot)

            Q[(r, r)] = -beta * alpha[r] + linear_penalty

        # Quadratic terms with penalty (off-diagonal), symmetrically added
        for r in range(n_features):
            for l in range(r + 1, n_features):
                quad_val = self.sign * (1 - beta) * cov[r, l] + lambd  # penalty term
                Q[(r, l)] = quad_val
                Q[(l, r)] = quad_val  # symmetric entry

        return Q

    def solve(self, method: str = 'simulated_annealing', plot_energy_spectrum: bool = False) -> List[int]:
        """
        Solve the QUBO problem using the specified method.
        :param method: Method to use for solving the QUBO problem ('brute_force' or 'simulated_annealing').
        :param plot_energy_spectrum: Whether to plot the energy spectrum.
        :return: List of selected feature indices.
        """
        # if self.n_features <20:
        #     print("Using brute force algorithm for small number of features.")
        #     method = 'brute_force'

        if method == 'brute_force':
            idxs_best = self.__brute_force_algorithm__(plot_energy_spectrum)

            self.optimal_features = self.idxs_alpha_nozero[idxs_best]
            print("Optimal features found using brute force:", self.optimal_features)
        elif method == 'simulated_annealing':
            idxs_best = self.__simulated_annealing__()
            self.optimal_features = self.idxs_alpha_nozero[idxs_best]
            print("Optimal features found using simulated annealing:", self.optimal_features)

        else:
            raise ValueError("Invalid method. Use 'brute_force' or 'simulated_annealing'.")