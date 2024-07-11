"""Math utils for the Nonadditivity Analysis package."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from pandas import DataFrame
from scipy import stats


def is_number(value: Any) -> bool:
    """Check whether value is convertible to a float.

    Args:
        value (Any): value to check.

    Returns:
        bool: true if value is a number and is or is convertible to float.
    """
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def mad_std(values: Sequence[float]) -> float:
    """Calculate median absolute deviation.

    Estimate the standard deviation based on the median absolute deviation
    (MAD) In Gaussian distributions, SD = 1.4826 * MAD

    This is a fast and simple robust SD estimator. However, it may have
    problems for small datasets where the average is not well estimated.

    Args:
        values (Sequence[float]): values to compute the MADSTD

    Returns:
        float: MADSTD for values

    Raises:
        ValueError: if either wrong type, e.g. NoneType or wrong values e.g.
        list[str] are given.
    """
    try:
        return float(1.4826 * np.median(np.abs(values - np.median(values))))
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Values has to be of type np.ndarray[float] or list[float]",
        ) from exc


def sn_medmed_std(values: Sequence[float]) -> float:
    """Calculate std off median pairwise difference.

    Estimate the standard deviation based on the median of median pairwise
    differences as defined by Rousseeuw and Croux [1]:

    Sn = 1.1926 * med_i( med_j(abs(xi - xj)))

    [1] Rousseeuw, Peter J.; Croux, Christophe (December 1993),
    "Alternatives to the Median Absolute Deviation",
    Journal of the American Statistical Association, American Statistical
    Association, 88 (424): 1273-1283, doi:10.2307/2291267, JSTOR 2291267

    Returns 0 if len(values) == 1

    Args:
        values (Sequence[float]): values to compute the sn_medmed_std

    Returns:
        float:  sn_medmed_std for vlaues

    Raises:
        ValueError: if wrong values or types given.
    """
    try:
        iter(values)
    except TypeError as exc:
        if not is_number(values):
            raise ValueError(
                "Values has to be of type np.ndarray[float] or list[float]",
            ) from exc
        return 0
    try:
        pairwise_medians = np.empty(len(values))
        for idx, i_value in enumerate(values):
            # Do I need to remove the case where i==j ? Tests indicate yes
            pairwise_medians[idx] = np.median(
                np.abs(i_value - np.delete(values, [idx])),
            )
        return float(1.1926 * np.median(pairwise_medians))
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Values has to be of type np.ndarray[float] or list[float]",
        ) from exc


def _calculate_series_theo_quantile(
    series_ids: list[str],
    nonadditivities: list[float],
) -> list[list[float]]:
    """Calculate theoquantiles using scipy.stats.boxplot.

    Args:
        series_ids (list[str]): list of the series name per circle
        nonadditivities (list[float]): nonadditivitiy per circle.

    Returns:
        list[list[float]]: list of the theoquantiles per circle
    """
    theo_quantiles = []
    for series in set(series_ids):
        sorted_nonadditivitiies = sorted(
            [
                (nonadditivity, x[0])
                for nonadditivity, x in zip(
                    nonadditivities,
                    enumerate(series_ids),
                )
                if x[1] == series
            ],
        )
        additivity_indices = [x[1] for x in sorted_nonadditivitiies]
        if len(additivity_indices) > 2:
            quantiles = list(stats.probplot(additivity_indices)[0][0])
        elif len(additivity_indices) == 2:
            quantiles = [-0.54495214, 0.54495214]
        else:
            quantiles = [0, 0]
        theo_quantiles += list(zip(additivity_indices, quantiles))
    return [quantile[1] for quantile in sorted(theo_quantiles)]


def _calculate_theo_quantiles(
    series_columns: str | None,
    series_ids: list[str],
    nonadditivities: list[float],
) -> list[list[float]]:
    """Calculate theoquantiles using scipy.stats.probplot.

    Args:
        series_columns (str | None): name of the series column.
        series_ids (list[str]): list of the series name per circle
        nonadditivities (list[float]): nonadditivitiy per circle.

    Returns:
        list[list[float]]: list of the theoquantiles per circle
    """
    if series_columns:
        return _calculate_series_theo_quantile(series_ids, nonadditivities)
    sorted_nonadditivities = sorted(
        zip(nonadditivities, range(len(nonadditivities))),
    )
    theo_quantiles = stats.probplot(sorted(nonadditivities))[0][0]
    theo_quantiles = [
        j[1]
        for j in sorted(
            zip(
                [i[1] for i in sorted_nonadditivities],
                list(theo_quantiles),
            ),
        )
    ]
    return theo_quantiles


def _calculate_mean_and_std_per_compound(
    per_compound_dataframe: DataFrame,
    property_column: str,
    series_column: str | None = None,
) -> list[list[float | None]] | list[list[list[float | None]]]:
    """Calculate mean nonadditivty and std per compound.

    Takes all nonadditivity values per compound and calculates mean,
    std and number of occurence.

    Args:
        per_compound_dataframe (DataFrame): per compound dataframe.
        property_column (str): name of the property column
        series_column (str | None, optional): name of the series column.
        Defaults to None.

    Returns:
        list[float | None] | list[list[float | None]]: list
        of lists containing mean, std, n_occurence for every row in the per
        compound dataframe.
    """

    def calculate_values(values: list[float]) -> list[float | None]:
        """Calculate mean std and num values for a list of values.

        Returns [None, None, 0] if the list is empty.

        Args:
            values (list[float | None]): list of mean, std and num values.

        Returns:
            list[Union[float, None]]: [mean, std, num_values] if len(values)
            > 0 else [None, None, 0]
        """
        num_values = len(values)
        if num_values > 0:
            mean = np.mean(values)
            std = np.std(values)
            return [float(mean), float(std), float(num_values)]
        return [None, None, 0]

    column = f"{property_column}_Nonadditivities"
    if series_column:
        pure_values, mixed_values = [], []
        for nonadditivities in per_compound_dataframe[column].to_numpy():
            pure_values.append(
                calculate_values(
                    values=[
                        value[0] for value in nonadditivities if value[1] == "pure"
                    ],
                ),
            )
            mixed_values.append(
                calculate_values(
                    values=[
                        value[0] for value in nonadditivities if value[1] == "mixed"
                    ],
                ),
            )
        return [pure_values, mixed_values]
    return [
        calculate_values(
            values=[value[0] for value in nonadditivities],
        )
        for nonadditivities in per_compound_dataframe[column].to_numpy()
    ]


def _calculate_nonadditivity(
    values: list[float],
) -> float:
    r"""Calculate the actual nonadditivity value as defined in the paper.

    C1-C2
    |	|
    C4-C3

    Na = p(C3) + p(C1) - p(C2) - p(C4)

    Args:
        values (list[float]): Values for the 4 compounds

    Returns:
        float: Calculated Nonaddictivity
    """
    return values[2] + values[0] - values[1] - values[3]
