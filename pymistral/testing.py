import numpy as np
from statsmodels.stats.multitest import multipletests


def xr_multipletest(p, alpha=0.05, method='fdr_bh', **multipletests_kwargs):
    """Apply statsmodels.stats.multitest.multipletests for multi-dimensional
     xr.objects.

    Args:
        p (xr.object): uncorrected p-values.
        alpha (type): FWER, family-wise error rate. Defaults to 0.05.
        method (str): Method used for testing and adjustment of pvalues. Can be
            either the full name or initial letters.  Available methods are:
            - bonferroni : one-step correction
            - sidak : one-step correction
            - holm-sidak : step down method using Sidak adjustments
            - holm : step-down method using Bonferroni adjustments
            - simes-hochberg : step-up method (independent)
            - hommel : closed method based on Simes tests (non-negative)
            - fdr_bh : Benjamini/Hochberg (non-negative)
            - fdr_by : Benjamini/Yekutieli (negative)
            - fdr_tsbh : two stage fdr correction (non-negative)
            - fdr_tsbky : two stage fdr correction (non-negative)
           Defaults to 'fdr_bh'.
        **multipletests_kwargs (optional dict): is_sorted, returnsorted
           see statsmodels.stats.multitest.multitest

    Returns:
        reject (xr.object): true for hypothesis that can be rejected for given
            alpha
        pvals_corrected (xr.object): p-values corrected for multiple tests

    Example:
        reject, xpvals_corrected = xr_multipletest(p)
    """
    # stack all to 1d array
    p_stacked = p.stack(s=p.dims)
    # mask only where not nan:
    # https://github.com/statsmodels/statsmodels/issues/2899
    mask = np.isfinite(p_stacked)
    pvals_corrected = np.full(p_stacked.shape, np.nan)
    reject = np.full(p_stacked.shape, np.nan)
    # apply test where mask
    reject[mask] = multipletests(
        p_stacked[mask], alpha=alpha, method=method, **multipletests_kwargs
    )[0]
    pvals_corrected[mask] = multipletests(
        p_stacked[mask], alpha=alpha, method=method, **multipletests_kwargs
    )[1]

    def unstack(reject, p_stacked):
        """Exchange values from p_stacked w/ reject (1d array) and unstack."""
        xreject = p_stacked.copy()
        xreject.values = reject
        xreject = xreject.unstack()
        return xreject

    reject = unstack(reject, p_stacked)
    pvals_corrected = unstack(pvals_corrected, p_stacked)
    return reject, pvals_corrected
