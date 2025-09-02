"""
Winsorization Implementation for Academic Research
---------------------------------------------------
This module provides a class for performing winsorizations, including MAD, k-sigma, 
and winsorization at extreme percentiles. The implementation focuses on clarity, 
documentation, and best practices for financial research.
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple

class Winsorizer:
    """
    A class to perform winsorizations on cross-sectional characteristic matrices.
    Supports input as either pandas DataFrame or numpy array.
    """

    @staticmethod
    def __to_dataframe(features: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Convert the input features to a pandas DataFrame if it is a numpy array.

        Args:
            features: (Time x Stock) DataFrame of features.

        Returns:
            pd.DataFrame: The input converted to a DataFrame.
        """
        if isinstance(features, pd.DataFrame):
            return features.copy()
        elif isinstance(features, np.ndarray):
            return pd.DataFrame(features)
        else:
            raise TypeError("Input features must be a pandas DataFrame or a numpy array.")

    @staticmethod
    def __to_original_type(result: pd.DataFrame, original: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Convert the DataFrame result back to the type of the original input.

        Args:
            result (pd.DataFrame): The processed DataFrame.
            original (Union[pd.DataFrame, np.ndarray]): The original input features.

        Returns:
            Union[pd.DataFrame, np.ndarray]: The result in the same type as the original input.
        """
        if isinstance(original, np.ndarray):
            return result.values
        return result

    @classmethod
    def MAD_winsorization(
        cls,
        features: Union[pd.DataFrame, np.ndarray],
        scaled: bool = False,
        k: int = 3
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply winsorization on features using the Median Absolute Deviation (MAD) method.

        Args:
            features: (Time x Stock) DataFrame of features.
            scaled (bool, optional): Whether to scale the MAD value (MAD * 1.4826). Default is False.
            k (int, optional): Scaling factor to determine limits. Default is 3.

        Returns:
            Union[pd.DataFrame, np.ndarray]: Winsorized features using MAD.
        """
        original = features
        df = cls.__to_dataframe(features)

        # Calculate the median for each row
        median = df.median(axis=1)
        # Compute the absolute deviation from the median, then calculate the median of these deviations for each row
        mad = (df.sub(median, axis=0)).abs().median(axis=1)

        # Scale the MAD if required
        if scaled:
            mad *= 1.4826

        # Calculate the lower and upper limits for winsorization
        lower = median - k * mad
        upper = median + k * mad

        # Apply winsorization using the DataFrame.clip() method for each row
        result = df.clip(lower=lower, upper=upper, axis=0)
        return cls.__to_original_type(result, original)

    @classmethod
    def sigma_winsorization(
        cls,
        features: Union[pd.DataFrame, np.ndarray],
        k: int = 3
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply winsorization on features using the k-sigma rule.

        Args:
            features: (Time x Stock) DataFrame of features.
            k (int, optional): Scaling factor to determine limits. Default is 3.

        Returns:
            Union[pd.DataFrame, np.ndarray]: Winsorized features using the k-sigma rule.
        """
        original = features
        df = cls.__to_dataframe(features)

        # Calculate the mean and standard deviation for each row
        mean = df.mean(axis=1)
        std = df.std(axis=1)

        # Calculate the lower and upper limits for winsorization
        lower = mean - k * std
        upper = mean + k * std

        # Apply winsorization using the DataFrame.clip() method for each row
        result = df.clip(lower=lower, upper=upper, axis=0)
        return cls.__to_original_type(result, original)

    @classmethod
    def percentile_winsorization(
        cls,
        features: Union[pd.DataFrame, np.ndarray],
        percentile: Tuple[float, float] = (0.01, 0.99),
        set_outlier_nan: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply winsorization on features using the percentile rule.

        Args:
            features: (Time x Stock) DataFrame of features.
            percentile (Tuple[float, float], optional): The lower and upper percentiles to winsorize. 
                Default is (0.01, 0.99).
            set_outlier_nan (bool, optional): Whether to set outliers to be NaN instead of clipping them. Default is False.

        Returns:
            Union[pd.DataFrame, np.ndarray]: Winsorized features using the percentile rule.
        """
        original = features
        df = cls.__to_dataframe(features)

        # Calculate lower and upper bounds based on the given percentiles
        lower_bound = df.quantile(percentile[0], axis=1)
        upper_bound = df.quantile(percentile[1], axis=1)

        if set_outlier_nan:
            # set the outlier values to be NaN
            mask = df.lt(lower_bound, axis=0) | df.gt(upper_bound, axis=0)
            result = df.mask(mask)
        else:
            # Clip values to the specified bounds
            result = df.clip(lower=lower_bound, upper=upper_bound, axis=0)

        return cls.__to_original_type(result, original)