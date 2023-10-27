import numpy as np
import pomegranate.distributions as distributions
import pomegranate.hmm as hmm
import torch

from Config.Config import *
from Pipelines.pipeline import pipeline


class pome_pipeline(pipeline):
    # TODO: Test support of multivariate data
    # TODO: Fix emission Prob (now I dont think it supports getting it in the framework. It should still work in pome)
    def __init__(self, reader, omitter, config: pome_pipeline_config):
        # TODO: get rid of the need for n_features. Seems like it is needed in order to initialize the means and covs.
        super().__init__(reader, omitter, config)

    def generate_initial_model_pytorch(self, n_iter):
        """
        Generates random parameters for the pomegranate model
        """
        dists = self.create_distributions()
        tensor_type = torch.float32
        edges = torch.rand(self._config.n_components, self._config.n_components, dtype=tensor_type)
        starts = torch.rand(self._config.n_components, dtype=tensor_type)
        starts = starts / torch.sum(starts)
        ends = torch.rand(self._config.n_components, dtype=tensor_type)
        self._model = hmm.DenseHMM(dists, edges=edges, starts=starts, ends=ends, max_iter=n_iter)

    def create_masked_tensor(self, sentence, ws):
        sentence_with_zeros = torch.full((max(ws) + 1,), 0)
        sentence_with_zeros[ws] = sentence
        mask = torch.full((max(ws) + 1,), False)
        mask[ws] = True
        return torch.masked.MaskedTensor(sentence_with_zeros,mask)

    def _iter_fit(self, data, n_iter, ws=None):

        self.generate_initial_model_pytorch(n_iter)
        #  TODO: Support multivariate data (n_features>1)
        if ws is not None:
            #  TODO: Export this to a method
            masked_data = []
            for sentence, ws in zip(data, ws):
                masked_data.append(self.create_masked_tensor(sentence, ws))
            data = masked_data

        data = [arr.reshape(-1, 1) for arr in data]  # assuming that the data is tensors
        data = self.partition_sequences(data)  # We need to do this ourselves in order to bypass a bug in pomegranate 1.0.3
        self._model.fit(X=data)
        return torch.exp(self._model.edges), torch.exp(self._model.starts)

    def fit(self):
        data, ws = self.omitter.omit(self.reader.get_obs())
        ## TODO: Take care of omission_idx later.
        ws = None
        data = [torch.from_numpy(sentence) for sentence in data]
        for i in range(1, self._config.n_iter + 1):
            transmat, start_prob = self._iter_fit(data, i, ws)
            self._transmat_list.append(transmat)
            self._startprob_list.append(start_prob)

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
            return [distributions.Categorical(torch.from_numpy(np.atleast_2d(dist)), frozen=self._config.freeze_distributions)
                    for dist in self._config.emission_prob]
        elif self._config.distribution == 'Gaussian':
            means, covs = self.generate_initial_normal_params_pytorch()
            return [distributions.Normal(means=means[i], covs=covs[i], frozen=self._config.freeze_distributions) for i in
                    range(self._config.n_components)]
        else:
            raise NotImplementedError(f"No model implemented for {self._config.distribution} distribution")

    def generate_initial_normal_params_pytorch(self):
        """
        Generates the initial params for the normal distributions
        :return:
        """
        tensor_type = torch.float32
        means = [torch.rand(self._config.n_features, dtype=tensor_type) for i in range(self._config.n_components)]

        random_matrix = torch.randn(self._config.n_features, self._config.n_features)

        # This ensures that the cov matrix is a legal cov matrix
        covariance_matrices = [torch.mm(random_matrix, random_matrix.t()) for i in range(self._config.n_components)]
        covariance_matrices = [covariance_matrix / torch.diag(covariance_matrix).sqrt().view(-1, 1) for
                               covariance_matrix in covariance_matrices]
        covariance_matrices = [covariance_matrix / torch.diag(covariance_matrix).sqrt().view(1, -1) for
                               covariance_matrix in covariance_matrices]

        return means, covariance_matrices

