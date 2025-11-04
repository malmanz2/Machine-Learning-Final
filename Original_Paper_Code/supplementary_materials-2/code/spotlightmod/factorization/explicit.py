import numpy as np
import time
import torch

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from spotlight.helpers import _repr_model
from spotlight.factorization._components import _predict_process_ids
from spotlight.factorization.representations import BilinearNet
from spotlight.losses import (poisson_loss,
                              regression_loss,
                              logistic_loss)
from spotlight.evaluation import *

from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle


class ExplicitFactorizationModel(object):
    """
    An explicit feedback matrix factorization model. Uses a classic
    matrix factorization [1]_ approach, with latent vectors used
    to represent both users and items. Their dot product gives the
    predicted score for a user-item pair.

    The latent representation is given by
    :class:`spotlight.factorization.representations.BilinearNet`.

    .. [1] Koren, Yehuda, Robert Bell, and Chris Volinsky.
       "Matrix factorization techniques for recommender systems."
       Computer 42.8 (2009).

    Parameters
    ----------

    loss: string, optional
        One of 'regression', 'poisson', 'logistic'
        corresponding to losses from :class:`spotlight.losses`.
    embedding_dim: int, optional
        Number of embedding dimensions to use for users and items.
    n_iter: int, optional
        Number of iterations to run.
    batch_size: int, optional
        Minibatch size.
    l2: float, optional
        L2 loss penalty.
    learning_rate: float, optional
        Initial learning rate.
    optimizer_func: function, optional
        Function that takes in module parameters as the first argument and
        returns an instance of a PyTorch optimizer. Overrides l2 and learning
        rate if supplied. If no optimizer supplied, then use ADAM by default.
    use_cuda: boolean, optional
        Run the model on a GPU.
    representation: a representation module, optional
        If supplied, will override default settings and be used as the
        main network module in the model. Intended to be used as an escape
        hatch when you want to reuse the model's training functions but
        want full freedom to specify your network topology.
    sparse: boolean, optional
        Use sparse gradients for embedding layers.
    random_state: instance of numpy.random.RandomState, optional
        Random state to use when fitting.
    """

    def __init__(self,
                 loss='regression',
                 embedding_dim=32,
                 n_iter=10,
                 batch_size=256,
                 l2=0.0,
                 learning_rate=1e-2,
                 sep_penalty=False,                 
                 item_penlam=0,
                 user_penlam=0,
                 jnt_penalty=False,                 
                 all_penlam=0,
                 lr_sched=False,
                 optimizer_func=None,
                 use_cuda=False,
                 representation=None,
                 sparse=False,
                 random_state=None):

        assert loss in ('regression',
                        'poisson',
                        'logistic')

        self._loss = loss
        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2 = l2
        self._sep_penalty = sep_penalty
        self._item_penlam = item_penlam
        self._user_penlam = user_penlam        
        self._jnt_penalty = jnt_penalty
        self._all_penlam = all_penlam
        
        self._lr_sched = lr_sched
        self._use_cuda = use_cuda
        self._representation = representation
        self._sparse = sparse
        self._optimizer_func = optimizer_func
        self._random_state = random_state or np.random.RandomState()

        self._num_users = None
        self._num_items = None
        self._net = None
        self._optimizer = None
        self._loss_func = None

        set_seed(self._random_state.randint(-10**8, 10**8),
                 cuda=self._use_cuda)

    def __repr__(self):

        return _repr_model(self)

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):

        (self._num_users,
         self._num_items) = (interactions.num_users,
                             interactions.num_items)

        if self._representation is not None:
            self._net = gpu(self._representation,
                            self._use_cuda)
        else:
            self._net = gpu(
                BilinearNet(self._num_users,
                            self._num_items,
                            self._embedding_dim,
                            sparse=self._sparse),
                self._use_cuda
            )

        if self._optimizer_func is None:
            self._optimizer = optim.Adam(
                self._net.parameters(),
                weight_decay=self._l2,
                lr=self._learning_rate
            )
        else:
            self._optimizer = self._optimizer_func(self._net.parameters(), lr=self._learning_rate, weight_decay=self._l2)
            
        if self._loss == 'regression':
            self._loss_func = regression_loss
        elif self._loss == 'poisson':
            self._loss_func = poisson_loss
        elif self._loss == 'logistic':
            self._loss_func = logistic_loss
        else:
            raise ValueError('Unknown loss: {}'.format(self._loss))

    def _check_input(self, user_ids, item_ids, allow_items_none=False):

        if isinstance(user_ids, int):
            user_id_max = user_ids
        else:
            user_id_max = user_ids.max()

        if user_id_max >= self._num_users:
            raise ValueError('Maximum user id greater '
                             'than number of users in model.')

        if allow_items_none and item_ids is None:
            return

        if isinstance(item_ids, int):
            item_id_max = item_ids
        else:
            item_id_max = item_ids.max()

        if item_id_max >= self._num_items:
            raise ValueError('Maximum item id greater '
                             'than number of items in model.')

    def _get_state_dict(self):
        return self._net.state_dict()             
            
    def effective_rank(self, mat):
        """calculate effective rank (expects a torch matrix)"""
        _U, sigmas, _V = mat.svd(compute_uv=False)
        x = sigmas / torch.sum(sigmas)
        return torch.exp(-torch.sum(x * torch.log(x + 1e-10)))
    
    def test_postbatch(self, test_uid, test_itemid, test_ratings):
        """
        Evaluate test performance at end of every mini-batch
        """                       
        self._net.eval()

        runloss = 0.

        for (minibatch_num,
             (batch_user,
              batch_item,
              batch_ratings)) in enumerate(minibatch(test_uid,
                                                     test_itemid,
                                                     test_ratings,
                                                     batch_size=self._batch_size)):
            with torch.no_grad():
                pred = self._net(batch_user, batch_item)        
                tstloss = self._loss_func(batch_ratings, pred)

            if np.isnan(tstloss.item()):
                raise ValueError('Degenerate tstloss: {}'
                                 .format(tstloss))

            runloss += np.sqrt(tstloss.item())

        tstRMSE = (runloss/(minibatch_num+1))
        
        self._net.train()

        return tstRMSE
    
    
    def test_all(self, interactions_test):
        """
        Evaluate test performance
        """                       
        self._net.eval()
        
        user_ids_tst = interactions_test.user_ids.astype(np.int64)
        item_ids_tst = interactions_test.item_ids.astype(np.int64)        
        ratings_tst = interactions_test.item_ids.astype(np.float64)                

        user_ids_tsttensor = gpu(torch.from_numpy(user_ids_tst),
                              self._use_cuda)
        item_ids_tsttensor = gpu(torch.from_numpy(item_ids_tst),
                              self._use_cuda)
        ratings_tsttensor = gpu(torch.from_numpy(ratings_tst),
                             self._use_cuda)
        
        preds = self._net(user_ids_tsttensor, item_ids_tsttensor)

        tstRMSE = (((ratings_tsttensor - preds) ** 2).mean()) ** 0.5
        
        self._net.train()

        return tstRMSE.item()    
                    


    def fit(self, interactions, interactions_test, prop=None, seed_run=None, rootdir=None, verbose=False, seed=None):
        """
        Fit the model.

        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit
        call.

        Parameters
        ----------

        interactions: :class:`spotlight.interactions.Interactions`
            The input dataset. Must have ratings.

        verbose: bool
            Output additional information about current epoch and loss.
        """
            

        user_ids = interactions.user_ids.astype(np.int64)
        item_ids = interactions.item_ids.astype(np.int64)

        user_ids_tst = interactions_test.user_ids.astype(np.int64)
        item_ids_tst = interactions_test.item_ids.astype(np.int64)
        
        if not self._initialized:
            self._initialize(interactions)
            self._initialize(interactions_test)            

        self._check_input(user_ids, item_ids)
        self._check_input(user_ids_tst, item_ids_tst)
        
        if self._lr_sched:
            scheduler = ReduceLROnPlateau(self._optimizer, factor=0.5, patience=3, threshold=0.01)
        
        
        for epoch_num in range(self._n_iter):

            users, items, ratings = shuffle(user_ids,
                                            item_ids,
                                            interactions.ratings,
                                            random_state=self._random_state)

            users_tst, items_tst, ratings_tst = shuffle(user_ids_tst,
                                            item_ids_tst,
                                            interactions_test.ratings,
                                            random_state=np.random.seed(3))
            
            user_ids_tensor = gpu(torch.from_numpy(users),
                                  self._use_cuda)
            item_ids_tensor = gpu(torch.from_numpy(items),
                                  self._use_cuda)
            ratings_tensor = gpu(torch.from_numpy(ratings),
                                 self._use_cuda)

            
            user_ids_tsttensor = gpu(torch.from_numpy(users_tst),
                                  self._use_cuda)
            item_ids_tsttensor = gpu(torch.from_numpy(items_tst),
                                  self._use_cuda)
            ratings_tsttensor = gpu(torch.from_numpy(ratings_tst),
                                 self._use_cuda)
                       
            epoch_loss = 0.0
            for (minibatch_num,
                 (batch_user,
                  batch_item,
                  batch_ratings)) in enumerate(minibatch(user_ids_tensor,
                                                         item_ids_tensor,
                                                         ratings_tensor,
                                                         batch_size=self._batch_size)):

                predictions = self._net(batch_user, batch_item)

                if self._loss == 'poisson':
                    predictions = torch.exp(predictions)

                self._optimizer.zero_grad()

                loss = self._loss_func(batch_ratings, predictions)

                if self._sep_penalty:
                    if self._item_penlam > 0 and self._user_penlam > 0: 
                        loss = loss + self._item_penlam*(self._net.item_embeddings.weight.norm("nuc")/ self._net.item_embeddings.weight.norm('fro'))+\
                        self._user_penlam*(self._net.user_embeddings.weight.norm("nuc")/ self._net.user_embeddings.weight.norm('fro'))
                    elif self._item_penlam > 0 and self._user_penlam <= 0:
                        loss = loss + self._item_penlam*(self._net.item_embeddings.weight.norm("nuc")/ self._net.item_embeddings.weight.norm('fro'))
                    elif self._item_penlam <= 0 and self._user_penlam > 0:                 
                        loss = loss + self._user_penlam*(self._net.user_embeddings.weight.norm("nuc")/ self._net.user_embeddings.weight.norm('fro'))                   

                if self._jnt_penalty:
                    e2e_mat = torch.matmul(self._net.user_embeddings.weight, self._net.item_embeddings.weight.t())
                    loss = loss + self._all_penlam * (e2e_mat.norm("nuc")/e2e_mat.norm('fro'))
                                                
                epoch_loss += loss.item()

                loss.backward()
                self._optimizer.step()
                            
            ## avg training loss per minibatch
            epoch_loss /= (minibatch_num + 1)
            
            ## avg test loss per minibatch
            test_loss = self.test_postbatch(user_ids_tsttensor, item_ids_tsttensor, ratings_tsttensor)
            
            test_loss_all = self.test_all(interactions_test)
            
            if self._lr_sched: 
                scheduler.step(epoch_loss)            
            
            if verbose:
                print(f"Epoch: {epoch_num+1}, trn_batch RMSE: {epoch_loss ** 2}, tst_batch RMSE: {test_loss}")
                        
            if np.isnan(epoch_loss):
                raise ValueError('Degenerate epoch loss: {}'
                                 .format(epoch_loss))
    
            
    def get_eranks(self):
        """
        Calculate ranks of user and item embedding matrices, used to calculate ranks at beginning and end of optimization
        """
        usrnet_erank = self.effective_rank(self._net.user_embeddings.weight)
        itemnet_erank = self.effective_rank(self._net.item_embeddings.weight)
        
        usrnet_rank = np.linalg.matrix_rank(self._net.user_embeddings.weight.cpu().detach().numpy())
        itemnet_rank = np.linalg.matrix_rank(self._net.item_embeddings.weight.cpu().detach().numpy())
        
        if self._jnt_penalty: 
            all_mat = torch.matmul(self._net.user_embeddings.weight, 
                                   self._net.item_embeddings.weight.t())
            all_erank = self.effective_rank(all_mat)
            all_rank = np.linalg.matrix_rank(all_mat.cpu().detach().numpy())
        
            ranks = {"user_weight_erank":usrnet_erank, 
                     "user_weight_rank":usrnet_rank,
                     "item_weight_erank":itemnet_erank,
                     "item_weight_rank":itemnet_rank,
                     "all_erank":all_erank,
                     "all_rank":all_rank
                    }
        else:
            ranks = {"user_weight_erank":usrnet_erank, 
                     "user_weight_rank":usrnet_rank,
                     "item_weight_erank":itemnet_erank,
                     "item_weight_rank":itemnet_rank,
                     "all_erank":"na",
                     "all_rank":"na"
                    }
        
        return ranks
        
    
    def predict(self, user_ids, item_ids=None):
        """
        Make predictions: given a user id, compute the recommendation
        scores for items.

        Parameters
        ----------

        user_ids: int or array
           If int, will predict the recommendation scores for this
           user for all items in item_ids. If an array, will predict
           scores for all (user, item) pairs defined by user_ids and
           item_ids.
        item_ids: array, optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.

        Returns
        -------

        predictions: np.array
            Predicted scores for all items in item_ids.
        """

        self._check_input(user_ids, item_ids, allow_items_none=True)
        self._net.train(False)

        user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                                                  self._num_items,
                                                  self._use_cuda)

        out = self._net(user_ids, item_ids)

        if self._loss == 'poisson':
            out = torch.exp(out)
        elif self._loss == 'logistic':
            out = torch.sigmoid(out)

        return cpu(out).detach().numpy().flatten()
