import numpy as np
import scipy.optimize as spo
from typing import Union, Literal, List, Optional, Set
import datetime

from SVAR import PointIdentifiedSVAR


class ExclusionRestriction(PointIdentifiedSVAR):
    def __init__(self,
                 data: np.ndarray,
                 var_names: list,
                 shock_names: list,
                 exclusion: Set[tuple],
                 constant: bool = True,
                 data_frequency: Literal['D', 'W', 'M', 'Q', 'SA', 'A'] = 'Q',
                 date_range: List[datetime.date] = None,  # specific to HD
                 info_criterion: Literal['aic', 'bic', 'hqc'] = 'aic'):
        super().__init__(data, var_names, shock_names, constant, info_criterion, data_frequency, date_range)
        self.identification = 'exclusion'
        self.exclusions = exclusion
        self.rotation = None
        self.n_restrictions = len(exclusion)
        if self.n_restrictions != self.n_vars * (self.n_vars - 1) / 2:
            raise ValueError('The model is not exactly identified!')
        all_list = {(i, j) for i in range(self.n_vars) for j in range(self.n_vars)}
        self.unrestricted = all_list.difference(self.exclusions)

    def __assign_non_zero_elements(self, unknown: np.ndarray) -> np.ndarray:

        target_rotation = np.zeros((self.n_vars, self.n_vars))
        for idx, item in enumerate(self.unrestricted):
            target_rotation[item] = unknown[idx]

        return target_rotation

    def __make_target_function(self,
                               x: np.ndarray,
                               cov_mat: np.ndarray,
                               normalization: bool = True):
        B_mat = self.__assign_non_zero_elements(x)
        rotation = np.linalg.inv(B_mat)
        if normalization:
            func = np.dot(rotation, rotation.T) - cov_mat
        else:
            func = 0

        return np.sum(func ** 2)

    def solve(self,
              comp_mat: Optional[np.ndarray] = None,
              cov_mat: Optional[np.ndarray] = None):
        if cov_mat is None:
            cov_mat = self.cov_mat
        if comp_mat is None:
            comp_mat = self.comp_mat

        target_func = lambda gamma: self.__make_target_function(gamma, cov_mat=cov_mat)
        sol = spo.minimize(fun=target_func, x0=np.ones(len(self.unrestricted)))
        rotation = np.linalg.inv(self.__assign_non_zero_elements(sol.x))

        return rotation

    def identify(self, h: int):
        if self.rotation is None:
            self.rotation = self.solve()
        self.irf_point_estimate = self._ReducedModel__get_irf(h=h, comp_mat=self.comp_mat, cov_mat=self.cov_mat,
                                                              rotation=self.rotation)
        self.vd_point_estimate = self._ReducedModel__get_vd(irfs=self.irf_point_estimate)

    def boot_confid_intvl(self,
                          h: int,
                          n_path: int,
                          irf_sig: Union[int, list],
                          vd_sig: Union[int, list, None] = None,
                          seed: Union[bool, int] = False):
        if vd_sig is None:
            vd_sig = irf_sig
        self.bootstrap(h=h, n_path=n_path, seed=seed)
        self.irf_cv(irf_mat=self.irf_mat, irf_sig=irf_sig, median_as_point_estimate=False)
        self.vd_cv(vd_mat=self.vd_mat, vd_sig=vd_sig, median_as_point_estimate=False)
