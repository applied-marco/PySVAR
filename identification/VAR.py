import datetime
import random
from typing import Union, Literal, List, Optional
import numpy as np

from ReducedModel import ReducedModel


class VAR(ReducedModel):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 constant: bool = True,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic',
                 data_frequency: Literal['D', 'W', 'M', 'Q', 'SA', 'A'] = 'Q',
                 date_range: List[datetime.date] = None):
        super().__init__(data, var_names, constant, info_criterion, data_frequency, date_range)

    def irf(self, h: int) -> np.ndarray:
        self.h_irf = h
        self.irf_point_estimate = self._ReducedModel__get_irf(h=self.h_irf, comp_mat=self.comp_mat,
                                                              cov_mat=self.cov_mat)

        return self.irf_point_estimate

    def vd(self, h: int) -> np.ndarray:
        if 'irf_point_estimate' not in self.__dir__() or h > self.irf_point_estimate.shape[1]:
            _ = self._ReducedModel__get_irf(h=h, comp_mat=self.comp_mat, cov_mat=self.cov_mat)
            self.vd_point_estimate = self._ReducedModel__get_vd(_)
        else:
            self.vd_point_estimate = self._ReducedModel__get_vd(self.irf_point_estimate[:, :h + 1])
        self.h_vd = h

        return self.vd_point_estimate

    def bootstrap(self,
                  h: int,
                  n_path: int = 1000,
                  seed: Union[bool, int] = False) -> None:
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.irf_mat = np.zeros((n_path, self.n_vars ** 2, h + 1))
        self.vd_mat = np.zeros((n_path, self.n_vars ** 2, h + 1))
        for r in range(n_path):
            yr = self._ReducedModel__make_bootstrap_sample()
            comp_mat_r, cov_mat_r, _, _, _ = self._Estimation__estimate(yr, self.lag_order)
            irfr = self._ReducedModel__get_irf(h, comp_mat=comp_mat_r, cov_mat=cov_mat_r[:self.n_vars, :self.n_vars])
            self.irf_mat[r, :, :] = irfr
            vdr = self._ReducedModel__get_vd(irfs=irfr)
            self.vd_mat[r, :, :] = vdr

    def irf_cv(self, sig_irf: Union[List[int], int]) -> None:
        if 'irf_mat' not in self.__dir__():
            raise ValueError("bootstrap first")
        self.irf_confid_intvl = self._ReducedModel__make_confid_intvl(mat=self.irf_mat, sigs=sig_irf)

    def vd_cv(self, sig_vd: Union[List[int], int]) -> None:
        if 'vd_mat' not in self.__dir__():
            raise ValueError("bootstrap first")
        self.vd_confid_intvl = self._ReducedModel__make_confid_intvl(mat=self.vd_mat, sigs=sig_vd)

    def plot_irf(self,
                 var_list: Optional[List[str]] = None,
                 shock_list: Optional[List[int]] = None,
                 sigs: Union[List[int], int, None] = None,
                 max_cols: int = 3,
                 with_ci: bool = True,
                 save_path: Optional[str] = None) -> None:

        if 'irf_point_estimate' not in self.__dir__():
            raise ValueError("IRFs should be estimated.")
        if 'irf_confid_intvl' not in self.__dir__() and with_ci:
            if sigs is None:
                raise ValueError('Not specifying significance levels.')
            self.irf_cv(sigs)
        h = min(self.irf_point_estimate.shape[1], self.irf_mat.shape[2])

        if var_list is None:
            var_list = self.var_names
        if shock_list is None:
            shock_list = self.shock_names
        else:
            _shock_list = []
            for i in shock_list:
                _shock_list.append(f'orth_shock_{i + 1}')
            shock_list = _shock_list

        self._ReducedModel__make_irf_graph(h=h, var_list=var_list, shock_list=shock_list, sigs=sigs,
                                           max_cols=max_cols, with_ci=with_ci, save_path=save_path)

    def plot_vd(self,
                var_list: Optional[List[str]] = None,
                shock_list: Optional[List[int]] = None,
                max_cols: int = 3,
                save_path: Optional[str] = None) -> None:

        if 'vd_point_estimate' not in self.__dir__():
            raise ValueError("IRFs should be estimated.")

        if var_list is None:
            var_list = self.var_names
        if shock_list is None:
            shock_list = self.shock_names
        else:
            _shock_list = []
            for i in shock_list:
                _shock_list.append(f'orth_shock_{i + 1}')
            shock_list = _shock_list
        h = self.vd_point_estimate.shape[1]

        self._ReducedModel__make_vd_graph(h=h, var_list=var_list, shock_list=shock_list, max_cols=max_cols,
                                          save_path=save_path)
