import numpy as np
import pomegranate.distributions as distributions
import pomegranate.hmm as hmm
import torch

from Config.Config import *
from Pipelines.pipeline import pipeline
import sys


def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

    # Test a matrix


class pome_pipeline(pipeline):
    def __init__(self, reader, omitter, config: pome_pipeline_config):
        """
        :param reader: the initialized reader
        :param omitter: the initialized reader
        :param config: the config
        """
        super().__init__(reader, omitter, config)

    def generate_initial_random_model(self, n_iter):
        """
        Generates random parameters for the pomegranate model
        :param n_iter: the number of iterations the model should perform
        """
        dists = self.create_random_distributions()

        tensor_type = torch.float32
        edges = torch.rand(self._config.n_components, self._config.n_components, dtype=tensor_type)
        starts = torch.rand(self._config.n_components, dtype=tensor_type)
        starts = starts / torch.sum(starts)
        ends = torch.rand(self._config.n_components, dtype=tensor_type)
        ends, edges = self.normalize_edges_and_ends(ends, edges)
        self._model = hmm.DenseHMM(dists, edges=edges, starts=starts, ends=ends, max_iter=n_iter)

        # save first random iter
        means = [dist.means.item() for dist in dists]
        self._means_list.append(means)
        self._transmat_list.append(edges.numpy())
        self._startprob_list.append(starts.numpy())

    def generate_initial_model(self, n_iter):
        """
        Generates random parameters for the pomegranate model, except for the distributions which are taken from the reader
        :param n_iter: the number of iterations the model should perform
        """
        dists = self.create_distributions()

        tensor_type = torch.float32
        edges = torch.rand(self._config.n_components, self._config.n_components, dtype=tensor_type)
        starts = torch.rand(self._config.n_components, dtype=tensor_type)
        starts = starts / torch.sum(starts)
        ends = torch.rand(self._config.n_components, dtype=tensor_type)
        ends, edges = self.normalize_edges_and_ends(ends, edges)
        self._model = hmm.DenseHMM(dists, edges=edges, starts=starts, ends=ends, max_iter=n_iter)

        # save first random iter
        means = [dist.means.item() for dist in dists]
        self._means_list.append(means)
        self._transmat_list.append(edges.numpy())
        self._startprob_list.append(starts.numpy())

    def create_masked_tensor(self, sentence, ws):
        """
        Creates a torch.masked.MaskedTensor using the data in sentence and the mask in ws after pre-processing
        :param sentence: the seen emissions data
        :param ws: the original locations of the seen emissions
        :return: a maskedTensor of the data
        """
        sentence_with_zeros = torch.full((self.reader._config.sentence_length,), 0, dtype=torch.double)
        sentence_with_zeros[ws] = sentence
        mask = torch.full((self.reader._config.sentence_length,), False, dtype=torch.bool)
        mask[ws] = True
        return torch.masked.MaskedTensor(sentence_with_zeros, mask)

    def _iter_fit_alt(self, data, n_iter, ws):
        """
        Fits the data one iteration at a time. After each iteration, saves the transmat, startprob and means
        :param data: the data to fit
        :param n_iter: the number of iteration
        :param ws: the original location of the seen emissions
        """

        self.generate_initial_model(n_iter)

        # Prepares data
        if ws is not None:
            masked_data = []
            for sentence, ws in zip(data, ws):
                masked_data.append(self.create_masked_tensor(sentence, ws))
            data = masked_data

        if data[0].ndim == 1:
            data = [arr.reshape(-1, 1) for arr in data]  # adding dimension to 1 dimensional array
        data = self.partition_sequences(
            data)  # We need to do this ourselves in order to bypass a bug in pomegranate 1.0.3

        # perform fit saves the results after each iter
        for i in range(n_iter):
            self._model.fit(X=data)
            means = [dist.means.item() for dist in self._model.distributions]
            self._transmat_list.append(torch.exp(self._model.edges).numpy())
            self._startprob_list.append(torch.exp(self._model.starts).numpy())
            self._means_list.append(means)

    def fit(self):
        """
        Creates the omitted datam and fits the model to the data
        """
        data, ws = self.omitter.omit(self.reader.get_obs())
        ## TODO: Take care of omission_idx later.
        data = [torch.from_numpy(sentence) for sentence in data]

        self._iter_fit_alt(data, self._config.n_iter, ws)


    @property
    def emissionprob(self):
        return self.create_emissionprob()

    def create_emissionprob(self):
        """
        Creates the emission probability matrix from the fitted distributions.
        This would only be relevant to call if freeze_distributions=false
        :return: the emission probability matrix
        """
        emission_prob = []
        for prob in self._model.distributions:
            emission_prob.append(np.array(prob.probs))

        return np.vstack(emission_prob)

    def partition_sequences(self, data):
        """
        groups sentences of the same length together.
        :param data: list of tensors or masked tensors of shape (n_obs,n_features)
        :return: values - list of tensors or masked tensors of shape (n_sentences,length,n_features)
        """
        lengths_dict = {}
        for tensor in data:
            if tensor.shape[0] not in lengths_dict:
                lengths_dict[tensor.shape[0]] = torch.unsqueeze(tensor, dim=0)
            else:
                lengths_dict[tensor.shape[0]] = torch.cat(
                    (lengths_dict[tensor.shape[0]], torch.unsqueeze(tensor, dim=0)))
        values = list(lengths_dict.values())
        values = sorted(values, key=lambda x: x.shape[1])
        return values

    def create_distributions(self):
        """
        Creates the distributions for the pomegranate model based on the distribution type.
        The distributions are generated with params extracted from the reader
        TAKE NOTICE: We have to initialize the distributions with data
        (e.g. emissions_prob in Categorical, means and covs in Gaussian) in order to bypass a bug in pomegranate 1.0.3
        :return: the initialized distributions
        """
        if self._config.distribution == 'Categorical':
            return [distributions.Categorical(torch.from_numpy(np.atleast_2d(dist)),
                                              frozen=self._config.freeze_distributions)
                    for dist in self.reader.get_emission_prob()]
        elif self._config.distribution == 'Gaussian':
            means = self.reader.means
            sigmas = self.reader.covs
            return [
                distributions.Normal(np.atleast_1d(mu).astype(float), np.atleast_1d(sigmas[i]).astype(float),
                                     frozen=self._config.freeze_distributions, covariance_type='diag')
                for
                i, mu in enumerate(means)]
        else:
            raise NotImplementedError(f"No model implemented for {self._config.distribution} distribution")

    def create_random_distributions(self):
        """
        Creates the distributions for the pomegranate model based on the distribution type.
        The distributions are generated with random params
        TAKE NOTICE: We have to initialize the distributions with data
        (e.g. emissions_prob in Categorical, means and covs in Gaussian) in order to bypass a bug in pomegranate 1.0.3
        :return: the initialized distributions
        """
        if self._config.distribution == 'Categorical':
            return [distributions.Categorical(torch.from_numpy(np.atleast_2d(dist)),
                                              frozen=self._config.freeze_distributions)
                    for dist in self.reader.get_emission_prob()]
        elif self._config.distribution == 'Gaussian':
            means, covs = self.generate_initial_normal_params()
            return [distributions.Normal(means=means[i], covs=covs[i], frozen=self._config.freeze_distributions) for i
                    in
                    range(self._config.n_components)]
        else:
            raise NotImplementedError(f"No model implemented for {self._config.distribution} distribution")

    def generate_initial_normal_params(self):
        """
        Generates the initial params for the normal dist.
        :return: the means and the cov matrices
        """
        tensor_type = torch.float32
        means = [torch.rand(self._config.n_features, dtype=tensor_type) for i in range(self._config.n_components)]

        random_matrix = torch.randn(self._config.n_features, self._config.n_features)

        # This ensures that the cov matrix is a legal cov matrix
        covariance_matrices = [torch.mm(random_matrix, random_matrix.t()) for i in range(self._config.n_components)]

        # This is just because I wanted to check if the cov is actually positive definite
        test = [is_positive_definite(covariance_matrix) for
                covariance_matrix in covariance_matrices]
        if not np.any(test):
            print(self.__str__())
        return means, covariance_matrices

    def normalize_edges_and_ends(self, ends, edges):
        """
        Normalizes the edges and ends probabilities so that each row sums to 1
        :param ends: the end probabilities
        :param edges: the edges probabilities
        :return: the normalized ends and edges probabilities
        """
        for i in range(ends.shape[0]):
            sum_row = ends[i] + torch.sum(edges[i])
            ends[i] = ends[i] / sum_row
            edges[i] = edges[i] / sum_row
        return ends, edges
