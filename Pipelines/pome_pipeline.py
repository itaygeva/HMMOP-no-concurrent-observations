import numpy as np
import pomegranate.distributions as distributions
import pomegranate.hmm as hmm
import torch

from Config.Config import *
from Pipelines.pipeline import pipeline


def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

    # Test a matrix


class pome_pipeline(pipeline):
    # TODO: Test support of multivariate data
    # TODO: Fix emission Prob (now I dont think it supports getting it in the framework. It should still work in pome)
    def __init__(self, reader, omitter, config: pome_pipeline_config):
        # TODO: get rid of the need for n_features. Seems like it is needed in order to initialize the means and covs.
        super().__init__(reader, omitter, config)

    def generate_initial_random_model(self, n_iter):
        """
                Generates random parameters for the pomegranate model
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
                Generates random parameters for the pomegranate model
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
        sentence_with_zeros = torch.full((max(ws) + 1,), 0)
        sentence_with_zeros[ws] = sentence
        mask = torch.full((max(ws) + 1,), False)
        mask[ws] = True
        return torch.masked.MaskedTensor(sentence_with_zeros, mask)

    def _iter_fit(self, data, n_iter, ws=None):

        self.generate_initial_model(n_iter)
        #  TODO: Support multivariate data (n_features>1)
        if ws is not None:
            #  TODO: Export this to a method
            masked_data = []
            for sentence, ws in zip(data, ws):
                masked_data.append(self.create_masked_tensor(sentence, ws))
            data = masked_data

        if data[0].ndim == 1:
            data = [arr.reshape(-1, 1) for arr in data]  # adding dimension to 1 dimensional array

        data = self.partition_sequences(
            data)  # We need to do this ourselves in order to bypass a bug in pomegranate 1.0.3
        self._model.fit(X=data)
        means = [dist.means.item() for dist in self._model.distributions]
        return torch.exp(self._model.edges), torch.exp(self._model.starts), means

    def _iter_fit_alt(self, data, n_iter, ws):
        self.generate_initial_model(n_iter)
        #  TODO: Support multivariate data (n_features>1)
        if ws is not None:
            #  TODO: Export this to a method
            masked_data = []
            for sentence, ws in zip(data, ws):
                masked_data.append(self.create_masked_tensor(sentence, ws))
            data = masked_data

        if data[0].ndim == 1:
            data = [arr.reshape(-1, 1) for arr in data]  # adding dimension to 1 dimensional array

        data = self.partition_sequences(
            data)  # We need to do this ourselves in order to bypass a bug in pomegranate 1.0.3
        for i in range(n_iter):
            self._model.fit(X=data)
            means = [dist.means.item() for dist in self._model.distributions]
            self._transmat_list.append(torch.exp(self._model.edges).numpy())
            self._startprob_list.append(torch.exp(self._model.starts).numpy())
            self._means_list.append(means)

    def fit(self):
        data, ws = self.omitter.omit(self.reader.get_obs())
        ## TODO: Take care of omission_idx later.
        ws = None
        data = [torch.from_numpy(sentence) for sentence in data]

        self._iter_fit_alt(data, self._config.n_iter, ws)

        """for i in range(1, self._config.n_iter + 1):
            #transmat, start_prob, means = self._iter_fit(data, i, ws)
            transmat, start_prob, means = self._iter_fit_alt(data, i, ws)
            self._transmat_list.append(transmat.numpy())
            self._startprob_list.append(start_prob.numpy())
            self._means_list.append(means)"""

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
        # TODO: test of this supports multivariate data
        # TODO: check if maybe we can use itertools.groupby instead
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
        Currently, supports - Gaussian, Categorical.
        TAKE NOTICE: We have to initialize the distributions with data
        (e.g. emissions_prob in Categorical, means and covs in Gaussian) in order to bypass a bug in pomegranate 1.0.3
        :return: the initialized distributions
        """
        if self._config.distribution == 'Categorical':
            ## TODO: support categorical and emission_prob
            return [distributions.Categorical(torch.from_numpy(np.atleast_2d(dist)),
                                              frozen=self._config.freeze_distributions)
                    for dist in self.reader.get_emission_prob()]
        elif self._config.distribution == 'Gaussian':
            means = self.reader.means
            sigmas = self.reader.covs
            return [
                distributions.Normal([float(mu)], [float(sigmas[i])], 'diag', frozen=self._config.freeze_distributions)
                for
                i, mu in enumerate(means)]
        else:
            raise NotImplementedError(f"No model implemented for {self._config.distribution} distribution")

    def create_random_distributions(self):
        """
        Creates the distributions for the pomegranate model based on the distribution type.
        Currently, supports - Gaussian, Categorical.
        TAKE NOTICE: We have to initialize the distributions with data
        (e.g. emissions_prob in Categorical, means and covs in Gaussian) in order to bypass a bug in pomegranate 1.0.3
        :return: the initialized distributions
        """
        if self._config.distribution == 'Categorical':
            ## TODO: support categorical and emission_prob
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
        Generates the initial params for the normal distributions
        :return:
        """
        tensor_type = torch.float32
        means = [torch.rand(self._config.n_features, dtype=tensor_type) for i in range(self._config.n_components)]

        random_matrix = torch.randn(self._config.n_features, self._config.n_features)

        # This ensures that the cov matrix is a legal cov matrix
        covariance_matrices = [torch.mm(random_matrix, random_matrix.t()) for i in range(self._config.n_components)]
        """covariance_matrices = [covariance_matrix / torch.diag(covariance_matrix).sqrt().view(-1, 1) for
                               covariance_matrix in covariance_matrices]
        covariance_matrices = [covariance_matrix / torch.diag(covariance_matrix).sqrt().view(1, -1) for
                               covariance_matrix in covariance_matrices]"""

        # This is just because i wanted to check if the cov is actually positive definite
        test = [is_positive_definite(covariance_matrix) for
                covariance_matrix in covariance_matrices]
        if not np.any(test):
            print(self.__str__())
        return means, covariance_matrices

    def normalize_edges_and_ends(self, ends, edges):
        for i in range(ends.shape[0]):
            sum_row = ends[i] + torch.sum(edges[i])
            ends[i] = ends[i] / sum_row
            edges[i] = edges[i] / sum_row
        return ends, edges
