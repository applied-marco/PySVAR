import numpy as np
import datetime
from typing import Literal, List, Union, Optional
import random
import matplotlib.pyplot as plt
import palettable.tableau as pt

from utils.plot_params import *
from Estimation import Estimation


class ReducedModel(Estimation):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 constant: bool = True,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic',
                 data_frequency: Literal['D', 'W', 'M', 'Q', 'SA', 'A'] = 'Q',
                 date_range: List[datetime.date] = None):
        super().__init__(data, constant)
        self.n_obs, self.n_vars = self.data.shape
        self.var_names = var_names
        if self.n_vars != len(self.var_names):
            raise ValueError('Names are not consistent with data dimension!')
        self.date_range = date_range
        self.data_frequency = data_frequency
        self.lag_order = self.optim_lag(y=self.data, criterion=info_criterion)
        self.identity = np.eye(self.n_vars)
        self.shock_names = [f'orth_shock_{i + 1}' for i in range(self.n_vars)]

    def fit(self) -> None:
        self.comp_mat, self._cov_mat, self._resids, self._intercepts, self._x = self._Estimation__estimate(self.data,
                                                                                                           self.lag_order)
        self.coeff_mat = self.comp_mat[:self.n_vars]
        self.cov_mat = self._cov_mat[:self.n_vars, :self.n_vars]
        self.intercepts = self._intercepts[:self.n_vars]
        zs = np.zeros((self.lag_order, self.n_vars))
        self.resids = np.concatenate((zs, self._resids[:self.n_vars, :].T), axis=0)  # this is the true residuals
        self.ar_coeff = dict()
        for i in range(0, self.lag_order):
            self.ar_coeff[str(i + 1)] = self.comp_mat[:self.n_vars, i * self.n_vars:(i + 1) * self.n_vars]
        self.__prepare_bootstrap()
        self.__pack_likelihood_info()

    def get_mean(self, whos: Union[str, List]) -> Union[int, List]:
        if isinstance(whos, str):
            return self.data.mean(axis=0)[self.var_names.index(whos)]
        else:
            index_list = self.__get_index(whos)
            return self.data.mean(axis=0)[index_list]

    def get_var(self, whos: Union[str, List]) -> Union[int, List]:
        if isinstance(whos, str):
            return self.data.var(axis=0, ddof=1)[self.var_names.index(whos)]
        else:
            index_list = self.__get_index(whos)
            return self.data.var(axis=0, ddof=1)[index_list]

    def get_corr(self,
                 var1: str,
                 var2: str) -> float:
        idx1 = self.var_names.index(var1)
        idx2 = self.var_names.index(var2)
        calced = np.corrcoef(self.data, rowvar=False)

        return calced[idx1, idx2]

    # methods not allowed to be called in the derived class
    def __pack_likelihood_info(self) -> None:
        Bhat = np.column_stack((self.intercepts, self.comp_mat[:self.n_vars, :]))
        Bhat = Bhat.T
        self.likelihood_info = {'Y': self.data, 'X': self._x.T, 'Bhat': Bhat, 'sigma': self.cov_mat,
                                'n': self.n_vars, 't': self.n_obs, 'p': self.lag_order}

    def __prepare_bootstrap(self) -> None:
        self._data_T = self.data.T
        self._yy = self._data_T[:, self.lag_order - 1:self.n_obs]
        for i in range(1, self.lag_order):
            self._yy = np.concatenate((self._yy, self._data_T[:, self.lag_order - i - 1:self.n_obs - i]), axis=0)
        self._yyr = np.zeros((self.lag_order * self.n_vars, self.n_obs - self.lag_order + 1))
        self._index_set = range(self.n_obs - self.lag_order)

    def __make_bootstrap_sample(self) -> np.ndarray:
        pos = random.randint(0, self.n_obs - self.lag_order)
        self._yyr[:, 0] = self._yy[:, pos]
        idx = np.random.choice(self._index_set, size=self.n_obs - self.lag_order)
        ur = np.concatenate((np.zeros((self.lag_order * self.n_vars, 1)), self._resids[:, idx]), axis=1)

        for i in range(1, self.n_obs - self.lag_order + 1):
            self._yyr[:, i] = self._intercepts.T + np.dot(self.comp_mat, self._yyr[:, i - 1]) + ur[:, i]
        yr = self._yyr[:self.n_vars, :]

        for i in range(1, self.lag_order):
            temp = self._yyr[i * self.n_vars:(i + 1) * self.n_vars, 0].reshape((-1, 1))
            yr = np.concatenate((temp, yr), axis=1)
        yr = yr.T

        return yr

    def __make_confid_intvl(self,
                            mat: np.ndarray,
                            sigs: Union[int, List]) -> dict:
        confid_intvl = dict()
        if not isinstance(sigs, list):
            sigs = [sigs]
        for sig in sigs:
            confid_intvl[sig] = dict()
            confid_intvl[sig]['lower'] = np.percentile(mat, (100 - sig) / 2, axis=0)
            confid_intvl[sig]['upper'] = np.percentile(mat, 100 - (100 - sig) / 2, axis=0)

        return confid_intvl

    def __get_index(self, var_list: List) -> List:
        index_list = []
        for var in var_list:
            index_list.append(self.var_names.index(var))
        return index_list

    def __get_irf(self,
                  h: int,
                  comp_mat: np.ndarray,
                  cov_mat: np.ndarray,
                  rotation: Optional[np.ndarray] = None) -> np.ndarray:

        if rotation is None:
            rotation = np.eye(self.n_vars)
        j = np.concatenate((np.eye(self.n_vars), np.zeros((self.n_vars, self.n_vars * (self.lag_order - 1)))), axis=1)
        aa = np.eye(self.n_vars * self.lag_order)
        chol = np.linalg.cholesky(cov_mat)  # cholesky gives you the lower triangle in numpy
        irf = np.dot(np.dot(np.dot(np.dot(j, aa), j.T), chol), rotation)
        irf = irf.reshape((self.n_vars ** 2, -1), order='F')

        for i in range(1, h + 1):
            aa = np.dot(aa, comp_mat)
            temp = np.dot(np.dot(np.dot(np.dot(j, aa), j.T), chol), rotation)
            temp = temp.reshape((self.n_vars ** 2, -1), order='F')
            irf = np.concatenate((irf, temp), axis=1)

        return irf

    def __get_vd(self, irfs: np.ndarray) -> np.ndarray:

        irf_mat = np.transpose(irfs)
        irf_mat_sq = irf_mat ** 2
        irf_mat_sq = irf_mat_sq.reshape((-1, self.n_vars, self.n_vars), order='F')
        irf_sq_sum_h = np.cumsum(irf_mat_sq, axis=0)
        total_fev = np.sum(irf_sq_sum_h, axis=2)
        total_fev_expand = np.expand_dims(total_fev, axis=2)
        vd = irf_sq_sum_h / total_fev_expand
        vd = vd.T.reshape((self.n_vars ** 2, -1))

        return vd

    def __get_hd(self):
        pass

    def __make_irf_graph(self,
                         h: int,
                         var_list: List[str],
                         shock_list: List[str],
                         sigs: Union[List[int], int],
                         max_cols: int,
                         with_ci: bool,
                         save_path: str) -> None:

        # confidence interval
        if with_ci:
            assert sigs is not None
            if not isinstance(sigs, list):
                sigs = [sigs]
            sig_available = list(self.irf_confid_intvl.keys())
            for sig in sigs:
                if sig not in sig_available:
                    raise ValueError(f'{sig} significance not available.')

        # layout
        ns = len(shock_list)
        nv = len(var_list)
        split = nv > max_cols
        n_cols = max_cols if split else nv
        n_rows = nv // max_cols + 1 if split else 1
        x_ticks = range(h)

        # plotting
        for i in range(ns):
            plt.figure(figsize=(n_cols * 10, n_rows * 10))
            plt.subplots_adjust(wspace=0.25, hspace=0.35)
            color = pt.BlueRed_6.mpl_colors[i]
            shock_id = self.shock_names.index(shock_list[i])
            for j in range(nv):
                ax = plt.subplot(n_rows, n_cols, j + 1)
                var_id = self.var_names.index(var_list[j])
                row = var_id + shock_id * self.n_vars
                plt.plot(x_ticks, self.irf_point_estimate[row, :], color=color, linewidth=3)
                plt.axhline(y=0, color='black', linestyle='-', linewidth=3)
                if with_ci:
                    for sig, alpha in zip(sigs, alpha_list[1:]):
                        plt.fill_between(x_ticks,
                                         self.irf_confid_intvl[sig]['lower'][row, :],
                                         self.irf_confid_intvl[sig]['upper'][row, :],
                                         alpha=alpha, edgecolor=color, facecolor=color, linewidth=0)
                plt.xlim(0, h - 1)
                plt.xticks(list(range(0, h, 5)))
                plt.title(var_list[j], font_prop_title, pad=5.)
                plt.tick_params(labelsize=25)
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname('Palatino') for label in labels]
                if j == 0:
                    ax.set_xlabel(date_transfer_dict[self.data_frequency], fontdict=font_prop_xlabels, labelpad=1.)
                plt.grid(linestyle='--', linewidth=1.5, color='black', alpha=0.35)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
            plt.suptitle(shock_list[i], fontproperties=font_prop_suptitle)

            if save_path is not None:
                full_path = save_path + f'/orth_shock{i}.png'
                plt.savefig(full_path, bbox_inches='tight')
            plt.show()

    def __make_vd_graph(self,
                        h: int,
                        var_list: List[str],
                        shock_list: List[str],
                        max_cols: int,
                        save_path: str) -> None:

        nv = len(var_list)
        split = nv > max_cols
        n_cols = max_cols if split else nv
        n_rows = nv // max_cols + 1 if split else 1
        x_ticks = range(h)

        plt.figure(figsize=(n_cols * 10, n_rows * 10))
        plt.subplots_adjust(wspace=0.25, hspace=0.35)

        for idxv, var in enumerate(var_list):
            accum = np.zeros(h)
            ax = plt.subplot(n_rows, n_cols, idxv + 1)
            for idxs, sho in enumerate(shock_list):
                color = pt.BlueRed_6.mpl_colors[idxs]
                shock_id = self.shock_names.index(sho)
                var_id = self.var_names.index(var)
                row = var_id + shock_id * self.n_vars
                plt.plot(x_ticks, self.vd_point_estimate[row, :], color=color, linewidth=3)
                accum += self.vd_point_estimate[row, :]
                plt.axhline(y=0, color='black', linestyle='-', linewidth=3)
            vd_rest = 1 - accum
            if np.sum(vd_rest) > 1e-10:
                plt.plot(x_ticks, vd_rest, color='k', linewidth=3)
            plt.xlim(0, h - 1)
            plt.xticks(list(range(0, h, 5)))
            plt.title(var, font_prop_title, pad=5.)
            plt.tick_params(labelsize=25)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Palatino') for label in labels]
            if idxv == 0:
                ax.set_xlabel(date_transfer_dict[self.data_frequency], fontdict=font_prop_xlabels, labelpad=1.)
            plt.grid(linestyle='--', linewidth=1.5, color='black', alpha=0.35)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

        plt.suptitle('Variance Decomposition', fontproperties=font_prop_suptitle)
        if save_path is not None:
            full_path = save_path + '/variance_decomposition.png'
            plt.savefig(full_path, bbox_inches='tight')
        plt.show()
