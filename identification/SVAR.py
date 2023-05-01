import datetime
import random
from typing import Union, Literal, List, Optional
import numpy as np

from ReducedModel import ReducedModel


class SVAR(ReducedModel):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 constant: bool = True,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic',
                 data_frequency: Literal['D', 'W', 'M', 'Q', 'SA', 'A'] = 'Q',
                 date_range: List[datetime.date] = None):
        super().__init__(data, var_names, constant, info_criterion, data_frequency, date_range)
        self.shock_names = shock_names
        self.n_shocks = len(shock_names)
        self.n_diff = self.n_vars - self.n_shocks
        self.fit()
        self.chol = np.linalg.cholesky(self.cov_mat)  # cholesky gives you the lower triangle in numpy

    def get_structural_shocks(self, rotation: np.ndarray) -> np.ndarray:
        return np.dot(np.linalg.inv(np.dot(self.chol, rotation)), self.resids[:self.n_vars, :]).T

    def set_params(self):
        pass

    def irf(self) -> np.ndarray:
        if 'irf_point_estimate' in self.__dir__():
            raise ValueError('Estimate first.')
        return self.irf_point_estimate

    def vd(self) -> np.ndarray:
        if 'vd_point_estimate' in self.__dir__():
            raise ValueError('Estimate first.')
        return self.vd_point_estimate

    def irf_cv(self,
               irf_mat: np.ndarray,
               irf_sig: Union[List[int], int],
               median_as_point_estimate: bool) -> None:

        self.irf_confid_intvl = self._ReducedModel__make_confid_intvl(mat=irf_mat, sigs=irf_sig)
        if median_as_point_estimate:
            self.irf_point_estimate = np.percentile(irf_mat, 50, axis=0)

    def vd_cv(self,
              vd_mat: np.ndarray,
              vd_sig: Union[List[int], int],
              median_as_point_estimate: bool) -> None:

        self.vd_confid_intvl = self._ReducedModel__make_confid_intvl(mat=vd_mat, sigs=vd_sig)
        if median_as_point_estimate:
            self.vd_point_estimate = np.percentile(vd_mat, 50, axis=0)

    def plot_irf(self,
                 var_list: List[str],
                 shock_list: List[str],
                 sigs: Union[List[int], int],
                 max_cols: int = 3,
                 with_ci: bool = True,
                 save_path: Optional[str] = None) -> None:

        if 'irf_point_estimate' not in self.__dir__():
            raise ValueError("IRFs should be estimated.")
        if 'irf_confid_intvl' not in self.__dir__() and with_ci:
            if sigs is None:
                raise ValueError('Not specifying significance levels.')
            if 'irf_mat' not in self.__dir__():
                raise ValueError('Bootstrap first.')
        h = self.irf_point_estimate.shape[1]

        if var_list is None:
            var_list = self.var_names
        if shock_list is None:
            shock_list = self.shock_names

        self._ReducedModel__make_irf_graph(h=h, var_list=var_list, shock_list=shock_list, sigs=sigs,
                                           max_cols=max_cols, with_ci=with_ci, save_path=save_path)

    def plot_vd(self,
                var_list: Optional[List[str]] = None,
                shock_list: Optional[List[str]] = None,
                max_cols: int = 3,
                save_path: Optional[str] = None) -> None:

        if 'vd_point_estimate' not in self.__dir__():
            raise ValueError("VD should be estimated.")
        h = self.vd_point_estimate.shape[1]

        if var_list is None:
            var_list = self.var_names
        if shock_list is None:
            shock_list = self.shock_names

        self._ReducedModel__make_vd_graph(h=h, var_list=var_list, shock_list=shock_list, max_cols=max_cols,
                                          save_path=save_path)


class SetIdentifiedSVAR(SVAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 constant: bool = True,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic',
                 data_frequency: Literal['D', 'W', 'M', 'Q', 'SA', 'A'] = 'Q',
                 date_range: List[datetime.date] = None):
        super().__init__(data, var_names, shock_names, constant, info_criterion, data_frequency, date_range)
        self.rotation_mat = None


class PointIdentifiedSVAR(SVAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 constant: bool = True,
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic',
                 data_frequency: Literal['D', 'W', 'M', 'Q', 'SA', 'A'] = 'Q',
                 date_range: List[datetime.date] = None):
        super().__init__(data, var_names, shock_names, constant, info_criterion, data_frequency, date_range)
        self.rotation = None

    def bootstrap(self,
                  h: int,
                  n_path: int,
                  seed: Union[bool, int] = False) -> None:
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.irf_mat = np.zeros((n_path, self.n_vars ** 2, h + 1))
        self.vd_mat = np.zeros((n_path, self.n_vars ** 2, h + 1))
        self.rotation_mat = np.zeros((n_path, self.n_vars, self.n_vars))

        for r in range(n_path):
            yr = self._ReducedModel__make_bootstrap_sample()
            comp_mat_r, cov_mat_r, _, _, _ = self._Estimation__estimate(yr, self.lag_order)
            cov_mat_r = cov_mat_r[:self.n_vars, :self.n_vars]
            rotationr = self.solve(comp_mat=comp_mat_r, cov_mat=cov_mat_r)
            self.rotation_mat[r, :, :] = rotationr
            irfr = self._ReducedModel__get_irf(h, rotation=rotationr, comp_mat=comp_mat_r, cov_mat=cov_mat_r)
            self.irf_mat[r, :, :] = irfr
            vdr = self._ReducedModel__get_vd(irfs=irfr)
            self.vd_mat[r, :, :] = vdr
