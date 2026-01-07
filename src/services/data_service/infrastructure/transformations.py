"""
Data transformation engine with configurable transformation steps.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
from uuid import UUID, uuid4
import re
from abc import ABC, abstractmethod

from ..domain.entities import (
    DataTransformation, TransformationType, Dataset, DataSchema,
    DataQualityReport, DataDomainService
)


class TransformationStep(ABC):
    """Abstract base class for transformation steps."""
    
    def __init__(self, step_id: str, parameters: Dict[str, Any] = None):
        self.step_id = step_id
        self.parameters = parameters or {}
        self.execution_time: Optional[float] = None
        self.rows_before: Optional[int] = None
        self.rows_after: Optional[int] = None
        self.columns_before: Optional[int] = None
        self.columns_after: Optional[int] = None
    
    @abstractmethod
    async def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the transformation step."""
        pass
    
    @abstractmethod
    def validate_parameters(self) -> Tuple[bool, Optional[str]]:
        """Validate transformation parameters."""
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get transformation step summary."""
        return {
            "step_id": self.step_id,
            "parameters": self.parameters,
            "execution_time": self.execution_time,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "columns_before": self.columns_before,
            "columns_after": self.columns_after,
            "rows_changed": (self.rows_after - self.rows_before) if self.rows_before and self.rows_after else None,
            "columns_changed": (self.columns_after - self.columns_before) if self.columns_before and self.columns_after else None
        }


class CleaningTransformations:
    """Data cleaning transformation steps."""
    
    class RemoveDuplicates(TransformationStep):
        """Remove duplicate rows."""
        
        def __init__(self, parameters: Dict[str, Any] = None):
            super().__init__("remove_duplicates", parameters)
        
        def validate_parameters(self) -> Tuple[bool, Optional[str]]:
            subset = self.parameters.get("subset")
            if subset and not isinstance(subset, list):
                return False, "subset parameter must be a list of column names"
            return True, None
        
        async def execute(self, df: pd.DataFrame) -> pd.DataFrame:
            start_time = datetime.now()
            self.rows_before = len(df)
            self.columns_before = len(df.columns)
            
            subset = self.parameters.get("subset")
            keep = self.parameters.get("keep", "first")
            
            result_df = df.drop_duplicates(subset=subset, keep=keep)
            
            self.rows_after = len(result_df)
            self.columns_after = len(result_df.columns)
            self.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result_df
    
    class RemoveMissingValues(TransformationStep):
        """Remove rows or columns with missing values."""
        
        def __init__(self, parameters: Dict[str, Any] = None):
            super().__init__("remove_missing_values", parameters)
        
        def validate_parameters(self) -> Tuple[bool, Optional[str]]:
            axis = self.parameters.get("axis", 0)
            if axis not in [0, 1, "rows", "columns"]:
                return False, "axis must be 0 (rows), 1 (columns), 'rows', or 'columns'"
            
            how = self.parameters.get("how", "any")
            if how not in ["any", "all"]:
                return False, "how must be 'any' or 'all'"
            
            return True, None
        
        async def execute(self, df: pd.DataFrame) -> pd.DataFrame:
            start_time = datetime.now()
            self.rows_before = len(df)
            self.columns_before = len(df.columns)
            
            axis = self.parameters.get("axis", 0)
            how = self.parameters.get("how", "any")
            subset = self.parameters.get("subset")
            
            # Convert string axis to numeric
            if axis == "rows":
                axis = 0
            elif axis == "columns":
                axis = 1
            
            result_df = df.dropna(axis=axis, how=how, subset=subset)
            
            self.rows_after = len(result_df)
            self.columns_after = len(result_df.columns)
            self.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result_df
    
    class FillMissingValues(TransformationStep):
        """Fill missing values with specified strategy."""
        
        def __init__(self, parameters: Dict[str, Any] = None):
            super().__init__("fill_missing_values", parameters)
        
        def validate_parameters(self) -> Tuple[bool, Optional[str]]:
            strategy = self.parameters.get("strategy", "constant")
            valid_strategies = ["constant", "mean", "median", "mode", "forward_fill", "backward_fill"]
            
            if strategy not in valid_strategies:
                return False, f"strategy must be one of {valid_strategies}"
            
            if strategy == "constant" and "value" not in self.parameters:
                return False, "value parameter required for constant strategy"
            
            return True, None
        
        async def execute(self, df: pd.DataFrame) -> pd.DataFrame:
            start_time = datetime.now()
            self.rows_before = len(df)
            self.columns_before = len(df.columns)
            
            strategy = self.parameters.get("strategy", "constant")
            columns = self.parameters.get("columns")
            
            result_df = df.copy()
            
            # Select columns to process
            if columns:
                target_columns = [col for col in columns if col in df.columns]
            else:
                target_columns = df.columns.tolist()
            
            for col in target_columns:
                if strategy == "constant":
                    value = self.parameters.get("value", 0)
                    result_df[col] = result_df[col].fillna(value)
                elif strategy == "mean":
                    if pd.api.types.is_numeric_dtype(result_df[col]):
                        result_df[col] = result_df[col].fillna(result_df[col].mean())
                elif strategy == "median":
                    if pd.api.types.is_numeric_dtype(result_df[col]):
                        result_df[col] = result_df[col].fillna(result_df[col].median())
                elif strategy == "mode":
                    mode_value = result_df[col].mode()
                    if len(mode_value) > 0:
                        result_df[col] = result_df[col].fillna(mode_value.iloc[0])
                elif strategy == "forward_fill":
                    result_df[col] = result_df[col].fillna(method='ffill')
                elif strategy == "backward_fill":
                    result_df[col] = result_df[col].fillna(method='bfill')
            
            self.rows_after = len(result_df)
            self.columns_after = len(result_df.columns)
            self.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result_df
    
    class RemoveOutliers(TransformationStep):
        """Remove outliers using statistical methods."""
        
        def __init__(self, parameters: Dict[str, Any] = None):
            super().__init__("remove_outliers", parameters)
        
        def validate_parameters(self) -> Tuple[bool, Optional[str]]:
            method = self.parameters.get("method", "iqr")
            valid_methods = ["iqr", "zscore", "modified_zscore"]
            
            if method not in valid_methods:
                return False, f"method must be one of {valid_methods}"
            
            return True, None
        
        async def execute(self, df: pd.DataFrame) -> pd.DataFrame:
            start_time = datetime.now()
            self.rows_before = len(df)
            self.columns_before = len(df.columns)
            
            method = self.parameters.get("method", "iqr")
            columns = self.parameters.get("columns")
            threshold = self.parameters.get("threshold", 3.0)
            
            result_df = df.copy()
            
            # Select numeric columns
            if columns:
                numeric_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            else:
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            outlier_mask = pd.Series([False] * len(df), index=df.index)
            
            for col in numeric_columns:
                if method == "iqr":
                    Q1 = result_df[col].quantile(0.25)
                    Q3 = result_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    col_outliers = (result_df[col] < lower_bound) | (result_df[col] > upper_bound)
                
                elif method == "zscore":
                    z_scores = np.abs((result_df[col] - result_df[col].mean()) / result_df[col].std())
                    col_outliers = z_scores > threshold
                
                elif method == "modified_zscore":
                    median = result_df[col].median()
                    mad = np.median(np.abs(result_df[col] - median))
                    modified_z_scores = 0.6745 * (result_df[col] - median) / mad
                    col_outliers = np.abs(modified_z_scores) > threshold
                
                outlier_mask = outlier_mask | col_outliers
            
            result_df = result_df[~outlier_mask]
            
            self.rows_after = len(result_df)
            self.columns_after = len(result_df.columns)
            self.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result_df
    
    class StandardizeText(TransformationStep):
        """Standardize text data."""
        
        def __init__(self, parameters: Dict[str, Any] = None):
            super().__init__("standardize_text", parameters)
        
        def validate_parameters(self) -> Tuple[bool, Optional[str]]:
            operations = self.parameters.get("operations", [])
            valid_operations = ["lowercase", "uppercase", "trim", "remove_special_chars", "remove_extra_spaces"]
            
            for op in operations:
                if op not in valid_operations:
                    return False, f"operation '{op}' not valid. Valid operations: {valid_operations}"
            
            return True, None
        
        async def execute(self, df: pd.DataFrame) -> pd.DataFrame:
            start_time = datetime.now()
            self.rows_before = len(df)
            self.columns_before = len(df.columns)
            
            operations = self.parameters.get("operations", ["lowercase", "trim"])
            columns = self.parameters.get("columns")
            
            result_df = df.copy()
            
            # Select text columns
            if columns:
                text_columns = [col for col in columns if col in df.columns]
            else:
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            for col in text_columns:
                for operation in operations:
                    if operation == "lowercase":
                        result_df[col] = result_df[col].astype(str).str.lower()
                    elif operation == "uppercase":
                        result_df[col] = result_df[col].astype(str).str.upper()
                    elif operation == "trim":
                        result_df[col] = result_df[col].astype(str).str.strip()
                    elif operation == "remove_special_chars":
                        result_df[col] = result_df[col].astype(str).str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                    elif operation == "remove_extra_spaces":
                        result_df[col] = result_df[col].astype(str).str.replace(r'\s+', ' ', regex=True)
            
            self.rows_after = len(result_df)
            self.columns_after = len(result_df.columns)
            self.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result_df


class NormalizationTransformations:
    """Data normalization transformation steps."""
    
    class MinMaxScaling(TransformationStep):
        """Min-Max normalization (0-1 scaling)."""
        
        def __init__(self, parameters: Dict[str, Any] = None):
            super().__init__("min_max_scaling", parameters)
        
        def validate_parameters(self) -> Tuple[bool, Optional[str]]:
            feature_range = self.parameters.get("feature_range", (0, 1))
            if not isinstance(feature_range, (list, tuple)) or len(feature_range) != 2:
                return False, "feature_range must be a tuple/list of 2 values"
            return True, None
        
        async def execute(self, df: pd.DataFrame) -> pd.DataFrame:
            start_time = datetime.now()
            self.rows_before = len(df)
            self.columns_before = len(df.columns)
            
            columns = self.parameters.get("columns")
            feature_range = self.parameters.get("feature_range", (0, 1))
            
            result_df = df.copy()
            
            # Select numeric columns
            if columns:
                numeric_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            else:
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numeric_columns:
                min_val = result_df[col].min()
                max_val = result_df[col].max()
                
                if max_val != min_val:  # Avoid division by zero
                    scaled = (result_df[col] - min_val) / (max_val - min_val)
                    result_df[col] = scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
            
            self.rows_after = len(result_df)
            self.columns_after = len(result_df.columns)
            self.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result_df
    
    class ZScoreNormalization(TransformationStep):
        """Z-score normalization (standardization)."""
        
        def __init__(self, parameters: Dict[str, Any] = None):
            super().__init__("zscore_normalization", parameters)
        
        def validate_parameters(self) -> Tuple[bool, Optional[str]]:
            return True, None
        
        async def execute(self, df: pd.DataFrame) -> pd.DataFrame:
            start_time = datetime.now()
            self.rows_before = len(df)
            self.columns_before = len(df.columns)
            
            columns = self.parameters.get("columns")
            
            result_df = df.copy()
            
            # Select numeric columns
            if columns:
                numeric_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            else:
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numeric_columns:
                mean_val = result_df[col].mean()
                std_val = result_df[col].std()
                
                if std_val != 0:  # Avoid division by zero
                    result_df[col] = (result_df[col] - mean_val) / std_val
            
            self.rows_after = len(result_df)
            self.columns_after = len(result_df.columns)
            self.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result_df
    
    class RobustScaling(TransformationStep):
        """Robust scaling using median and IQR."""
        
        def __init__(self, parameters: Dict[str, Any] = None):
            super().__init__("robust_scaling", parameters)
        
        def validate_parameters(self) -> Tuple[bool, Optional[str]]:
            return True, None
        
        async def execute(self, df: pd.DataFrame) -> pd.DataFrame:
            start_time = datetime.now()
            self.rows_before = len(df)
            self.columns_before = len(df.columns)
            
            columns = self.parameters.get("columns")
            
            result_df = df.copy()
            
            # Select numeric columns
            if columns:
                numeric_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            else:
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numeric_columns:
                median_val = result_df[col].median()
                q1 = result_df[col].quantile(0.25)
                q3 = result_df[col].quantile(0.75)
                iqr = q3 - q1
                
                if iqr != 0:  # Avoid division by zero
                    result_df[col] = (result_df[col] - median_val) / iqr
            
            self.rows_after = len(result_df)
            self.columns_after = len(result_df.columns)
            self.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result_df


class AggregationTransformations:
    """Data aggregation transformation steps."""
    
    class GroupByAggregation(TransformationStep):
        """Group by aggregation."""
        
        def __init__(self, parameters: Dict[str, Any] = None):
            super().__init__("groupby_aggregation", parameters)
        
        def validate_parameters(self) -> Tuple[bool, Optional[str]]:
            group_by = self.parameters.get("group_by")
            if not group_by:
                return False, "group_by parameter is required"
            
            aggregations = self.parameters.get("aggregations")
            if not aggregations:
                return False, "aggregations parameter is required"
            
            return True, None
        
        async def execute(self, df: pd.DataFrame) -> pd.DataFrame:
            start_time = datetime.now()
            self.rows_before = len(df)
            self.columns_before = len(df.columns)
            
            group_by = self.parameters.get("group_by")
            aggregations = self.parameters.get("aggregations")
            
            # Ensure group_by columns exist
            if isinstance(group_by, str):
                group_by = [group_by]
            
            missing_cols = [col for col in group_by if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Group by columns not found: {missing_cols}")
            
            result_df = df.groupby(group_by).agg(aggregations).reset_index()
            
            # Flatten column names if multi-level
            if isinstance(result_df.columns, pd.MultiIndex):
                result_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in result_df.columns.values]
            
            self.rows_after = len(result_df)
            self.columns_after = len(result_df.columns)
            self.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result_df
    
    class TimeSeriesResampling(TransformationStep):
        """Time series resampling and aggregation."""
        
        def __init__(self, parameters: Dict[str, Any] = None):
            super().__init__("timeseries_resampling", parameters)
        
        def validate_parameters(self) -> Tuple[bool, Optional[str]]:
            date_column = self.parameters.get("date_column")
            if not date_column:
                return False, "date_column parameter is required"
            
            frequency = self.parameters.get("frequency")
            if not frequency:
                return False, "frequency parameter is required (e.g., 'D', 'W', 'M')"
            
            return True, None
        
        async def execute(self, df: pd.DataFrame) -> pd.DataFrame:
            start_time = datetime.now()
            self.rows_before = len(df)
            self.columns_before = len(df.columns)
            
            date_column = self.parameters.get("date_column")
            frequency = self.parameters.get("frequency")
            aggregation = self.parameters.get("aggregation", "mean")
            
            result_df = df.copy()
            
            # Convert date column to datetime
            result_df[date_column] = pd.to_datetime(result_df[date_column])
            
            # Set date column as index
            result_df = result_df.set_index(date_column)
            
            # Resample
            if isinstance(aggregation, str):
                result_df = result_df.resample(frequency).agg(aggregation)
            else:
                result_df = result_df.resample(frequency).agg(aggregation)
            
            # Reset index
            result_df = result_df.reset_index()
            
            self.rows_after = len(result_df)
            self.columns_after = len(result_df.columns)
            self.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result_df


class FilterTransformations:
    """Data filtering transformation steps."""
    
    class RowFilter(TransformationStep):
        """Filter rows based on conditions."""
        
        def __init__(self, parameters: Dict[str, Any] = None):
            super().__init__("row_filter", parameters)
        
        def validate_parameters(self) -> Tuple[bool, Optional[str]]:
            conditions = self.parameters.get("conditions")
            if not conditions:
                return False, "conditions parameter is required"
            
            return True, None
        
        async def execute(self, df: pd.DataFrame) -> pd.DataFrame:
            start_time = datetime.now()
            self.rows_before = len(df)
            self.columns_before = len(df.columns)
            
            conditions = self.parameters.get("conditions")
            logic = self.parameters.get("logic", "and")  # "and" or "or"
            
            result_df = df.copy()
            
            # Build filter mask
            masks = []
            for condition in conditions:
                column = condition.get("column")
                operator = condition.get("operator")
                value = condition.get("value")
                
                if column not in df.columns:
                    continue
                
                if operator == "==":
                    mask = result_df[column] == value
                elif operator == "!=":
                    mask = result_df[column] != value
                elif operator == ">":
                    mask = result_df[column] > value
                elif operator == ">=":
                    mask = result_df[column] >= value
                elif operator == "<":
                    mask = result_df[column] < value
                elif operator == "<=":
                    mask = result_df[column] <= value
                elif operator == "in":
                    mask = result_df[column].isin(value)
                elif operator == "not_in":
                    mask = ~result_df[column].isin(value)
                elif operator == "contains":
                    mask = result_df[column].astype(str).str.contains(str(value), na=False)
                elif operator == "not_contains":
                    mask = ~result_df[column].astype(str).str.contains(str(value), na=False)
                else:
                    continue
                
                masks.append(mask)
            
            # Combine masks
            if masks:
                if logic == "and":
                    final_mask = masks[0]
                    for mask in masks[1:]:
                        final_mask = final_mask & mask
                else:  # "or"
                    final_mask = masks[0]
                    for mask in masks[1:]:
                        final_mask = final_mask | mask
                
                result_df = result_df[final_mask]
            
            self.rows_after = len(result_df)
            self.columns_after = len(result_df.columns)
            self.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result_df
    
    class ColumnFilter(TransformationStep):
        """Select or drop specific columns."""
        
        def __init__(self, parameters: Dict[str, Any] = None):
            super().__init__("column_filter", parameters)
        
        def validate_parameters(self) -> Tuple[bool, Optional[str]]:
            action = self.parameters.get("action", "select")
            if action not in ["select", "drop"]:
                return False, "action must be 'select' or 'drop'"
            
            columns = self.parameters.get("columns")
            if not columns:
                return False, "columns parameter is required"
            
            return True, None
        
        async def execute(self, df: pd.DataFrame) -> pd.DataFrame:
            start_time = datetime.now()
            self.rows_before = len(df)
            self.columns_before = len(df.columns)
            
            action = self.parameters.get("action", "select")
            columns = self.parameters.get("columns")
            
            # Ensure columns is a list
            if isinstance(columns, str):
                columns = [columns]
            
            # Filter to existing columns
            existing_columns = [col for col in columns if col in df.columns]
            
            if action == "select":
                result_df = df[existing_columns]
            else:  # drop
                result_df = df.drop(columns=existing_columns)
            
            self.rows_after = len(result_df)
            self.columns_after = len(result_df.columns)
            self.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result_df


class DerivedTransformations:
    """Derived column transformation steps."""
    
    class CreateDerivedColumn(TransformationStep):
        """Create new columns based on existing data."""
        
        def __init__(self, parameters: Dict[str, Any] = None):
            super().__init__("create_derived_column", parameters)
        
        def validate_parameters(self) -> Tuple[bool, Optional[str]]:
            new_column = self.parameters.get("new_column")
            if not new_column:
                return False, "new_column parameter is required"
            
            expression_type = self.parameters.get("expression_type")
            if expression_type not in ["arithmetic", "conditional", "string", "date"]:
                return False, "expression_type must be one of: arithmetic, conditional, string, date"
            
            return True, None
        
        async def execute(self, df: pd.DataFrame) -> pd.DataFrame:
            start_time = datetime.now()
            self.rows_before = len(df)
            self.columns_before = len(df.columns)
            
            new_column = self.parameters.get("new_column")
            expression_type = self.parameters.get("expression_type")
            
            result_df = df.copy()
            
            if expression_type == "arithmetic":
                # Arithmetic operations between columns
                operand1 = self.parameters.get("operand1")
                operand2 = self.parameters.get("operand2")
                operation = self.parameters.get("operation", "+")
                
                if operand1 in df.columns and operand2 in df.columns:
                    if operation == "+":
                        result_df[new_column] = result_df[operand1] + result_df[operand2]
                    elif operation == "-":
                        result_df[new_column] = result_df[operand1] - result_df[operand2]
                    elif operation == "*":
                        result_df[new_column] = result_df[operand1] * result_df[operand2]
                    elif operation == "/":
                        result_df[new_column] = result_df[operand1] / result_df[operand2]
            
            elif expression_type == "conditional":
                # Conditional column creation
                condition_column = self.parameters.get("condition_column")
                condition_operator = self.parameters.get("condition_operator", "==")
                condition_value = self.parameters.get("condition_value")
                true_value = self.parameters.get("true_value")
                false_value = self.parameters.get("false_value")
                
                if condition_column in df.columns:
                    if condition_operator == "==":
                        condition = result_df[condition_column] == condition_value
                    elif condition_operator == ">":
                        condition = result_df[condition_column] > condition_value
                    elif condition_operator == "<":
                        condition = result_df[condition_column] < condition_value
                    elif condition_operator == ">=":
                        condition = result_df[condition_column] >= condition_value
                    elif condition_operator == "<=":
                        condition = result_df[condition_column] <= condition_value
                    else:
                        condition = result_df[condition_column] == condition_value
                    
                    result_df[new_column] = np.where(condition, true_value, false_value)
            
            elif expression_type == "string":
                # String operations
                source_columns = self.parameters.get("source_columns", [])
                operation = self.parameters.get("operation", "concatenate")
                separator = self.parameters.get("separator", "")
                
                if operation == "concatenate" and source_columns:
                    existing_cols = [col for col in source_columns if col in df.columns]
                    if existing_cols:
                        result_df[new_column] = result_df[existing_cols].astype(str).agg(separator.join, axis=1)
            
            elif expression_type == "date":
                # Date operations
                source_column = self.parameters.get("source_column")
                operation = self.parameters.get("operation", "extract_year")
                
                if source_column in df.columns:
                    date_series = pd.to_datetime(result_df[source_column], errors='coerce')
                    
                    if operation == "extract_year":
                        result_df[new_column] = date_series.dt.year
                    elif operation == "extract_month":
                        result_df[new_column] = date_series.dt.month
                    elif operation == "extract_day":
                        result_df[new_column] = date_series.dt.day
                    elif operation == "extract_weekday":
                        result_df[new_column] = date_series.dt.dayofweek
                    elif operation == "days_from_today":
                        today = pd.Timestamp.now()
                        result_df[new_column] = (today - date_series).dt.days
            
            self.rows_after = len(result_df)
            self.columns_after = len(result_df.columns)
            self.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result_df


class TransformationEngine:
    """Main transformation engine that orchestrates transformation steps."""
    
    def __init__(self):
        self.transformation_registry = self._build_transformation_registry()
    
    def _build_transformation_registry(self) -> Dict[str, type]:
        """Build registry of available transformation steps."""
        return {
            # Cleaning transformations
            "remove_duplicates": CleaningTransformations.RemoveDuplicates,
            "remove_missing_values": CleaningTransformations.RemoveMissingValues,
            "fill_missing_values": CleaningTransformations.FillMissingValues,
            "remove_outliers": CleaningTransformations.RemoveOutliers,
            "standardize_text": CleaningTransformations.StandardizeText,
            
            # Normalization transformations
            "min_max_scaling": NormalizationTransformations.MinMaxScaling,
            "zscore_normalization": NormalizationTransformations.ZScoreNormalization,
            "robust_scaling": NormalizationTransformations.RobustScaling,
            
            # Aggregation transformations
            "groupby_aggregation": AggregationTransformations.GroupByAggregation,
            "timeseries_resampling": AggregationTransformations.TimeSeriesResampling,
            
            # Filter transformations
            "row_filter": FilterTransformations.RowFilter,
            "column_filter": FilterTransformations.ColumnFilter,
            
            # Derived transformations
            "create_derived_column": DerivedTransformations.CreateDerivedColumn,
        }
    
    async def execute_transformation_pipeline(
        self,
        df: pd.DataFrame,
        transformations: List[Dict[str, Any]],
        validate_steps: bool = True
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Execute a pipeline of transformations."""
        
        result_df = df.copy()
        execution_summary = []
        
        for i, transformation_config in enumerate(transformations):
            step_name = transformation_config.get("step")
            parameters = transformation_config.get("parameters", {})
            
            if step_name not in self.transformation_registry:
                raise ValueError(f"Unknown transformation step: {step_name}")
            
            # Create transformation step
            step_class = self.transformation_registry[step_name]
            step = step_class(parameters)
            
            # Validate parameters if requested
            if validate_steps:
                is_valid, error_message = step.validate_parameters()
                if not is_valid:
                    raise ValueError(f"Step {i+1} ({step_name}) validation failed: {error_message}")
            
            try:
                # Execute transformation
                result_df = await step.execute(result_df)
                
                # Add to execution summary
                step_summary = step.get_summary()
                step_summary["step_number"] = i + 1
                step_summary["status"] = "success"
                execution_summary.append(step_summary)
                
            except Exception as e:
                # Add error to summary
                step_summary = {
                    "step_number": i + 1,
                    "step_id": step_name,
                    "status": "failed",
                    "error": str(e),
                    "parameters": parameters
                }
                execution_summary.append(step_summary)
                raise ValueError(f"Step {i+1} ({step_name}) failed: {str(e)}")
        
        return result_df, execution_summary
    
    def get_available_transformations(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available transformations."""
        
        transformations_info = {
            "cleaning": {
                "remove_duplicates": {
                    "description": "Remove duplicate rows from the dataset",
                    "parameters": {
                        "subset": "List of column names to consider for duplicates (optional)",
                        "keep": "Which duplicates to keep: 'first', 'last', or False (default: 'first')"
                    }
                },
                "remove_missing_values": {
                    "description": "Remove rows or columns with missing values",
                    "parameters": {
                        "axis": "0 for rows, 1 for columns (default: 0)",
                        "how": "'any' or 'all' (default: 'any')",
                        "subset": "List of column names to consider (optional)"
                    }
                },
                "fill_missing_values": {
                    "description": "Fill missing values with specified strategy",
                    "parameters": {
                        "strategy": "constant, mean, median, mode, forward_fill, backward_fill",
                        "value": "Value to use for constant strategy",
                        "columns": "List of columns to process (optional)"
                    }
                },
                "remove_outliers": {
                    "description": "Remove outliers using statistical methods",
                    "parameters": {
                        "method": "iqr, zscore, modified_zscore (default: iqr)",
                        "threshold": "Threshold for zscore methods (default: 3.0)",
                        "columns": "List of columns to process (optional)"
                    }
                },
                "standardize_text": {
                    "description": "Standardize text data with various operations",
                    "parameters": {
                        "operations": "List of operations: lowercase, uppercase, trim, remove_special_chars, remove_extra_spaces",
                        "columns": "List of columns to process (optional)"
                    }
                }
            },
            "normalization": {
                "min_max_scaling": {
                    "description": "Scale features to a fixed range (default 0-1)",
                    "parameters": {
                        "feature_range": "Tuple of (min, max) values (default: (0, 1))",
                        "columns": "List of columns to scale (optional)"
                    }
                },
                "zscore_normalization": {
                    "description": "Standardize features using z-score (mean=0, std=1)",
                    "parameters": {
                        "columns": "List of columns to normalize (optional)"
                    }
                },
                "robust_scaling": {
                    "description": "Scale features using median and IQR (robust to outliers)",
                    "parameters": {
                        "columns": "List of columns to scale (optional)"
                    }
                }
            },
            "aggregation": {
                "groupby_aggregation": {
                    "description": "Group data by columns and apply aggregation functions",
                    "parameters": {
                        "group_by": "Column name or list of column names to group by",
                        "aggregations": "Dictionary of column: aggregation_function pairs"
                    }
                },
                "timeseries_resampling": {
                    "description": "Resample time series data to different frequency",
                    "parameters": {
                        "date_column": "Name of the date column",
                        "frequency": "Resampling frequency (e.g., 'D', 'W', 'M')",
                        "aggregation": "Aggregation method (default: 'mean')"
                    }
                }
            },
            "filtering": {
                "row_filter": {
                    "description": "Filter rows based on conditions",
                    "parameters": {
                        "conditions": "List of condition dictionaries with column, operator, value",
                        "logic": "'and' or 'or' to combine conditions (default: 'and')"
                    }
                },
                "column_filter": {
                    "description": "Select or drop specific columns",
                    "parameters": {
                        "action": "'select' or 'drop'",
                        "columns": "List of column names"
                    }
                }
            },
            "derived": {
                "create_derived_column": {
                    "description": "Create new columns based on existing data",
                    "parameters": {
                        "new_column": "Name of the new column",
                        "expression_type": "arithmetic, conditional, string, or date",
                        "Additional parameters depend on expression_type": "See documentation"
                    }
                }
            }
        }
        
        return transformations_info
    
    async def validate_transformation_pipeline(
        self,
        transformations: List[Dict[str, Any]],
        df_columns: List[str] = None
    ) -> Tuple[bool, List[str]]:
        """Validate a transformation pipeline without executing it."""
        
        errors = []
        
        for i, transformation_config in enumerate(transformations):
            step_name = transformation_config.get("step")
            parameters = transformation_config.get("parameters", {})
            
            # Check if step exists
            if step_name not in self.transformation_registry:
                errors.append(f"Step {i+1}: Unknown transformation step '{step_name}'")
                continue
            
            # Validate step parameters
            step_class = self.transformation_registry[step_name]
            step = step_class(parameters)
            
            is_valid, error_message = step.validate_parameters()
            if not is_valid:
                errors.append(f"Step {i+1} ({step_name}): {error_message}")
            
            # Additional column validation if df_columns provided
            if df_columns:
                columns_param = parameters.get("columns")
                if columns_param:
                    if isinstance(columns_param, str):
                        columns_param = [columns_param]
                    
                    missing_columns = [col for col in columns_param if col not in df_columns]
                    if missing_columns:
                        errors.append(f"Step {i+1} ({step_name}): Columns not found: {missing_columns}")
        
        return len(errors) == 0, errors


class TransformationResult:
    """Result of a transformation pipeline execution."""
    
    def __init__(
        self,
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
        execution_summary: List[Dict[str, Any]],
        transformation_config: List[Dict[str, Any]]
    ):
        self.original_df = original_df
        self.transformed_df = transformed_df
        self.execution_summary = execution_summary
        self.transformation_config = transformation_config
        self.execution_time = sum(step.get("execution_time", 0) for step in execution_summary)
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """Get summary of the transformation results."""
        
        successful_steps = [step for step in self.execution_summary if step.get("status") == "success"]
        failed_steps = [step for step in self.execution_summary if step.get("status") == "failed"]
        
        return {
            "original_shape": self.original_df.shape,
            "transformed_shape": self.transformed_df.shape,
            "total_steps": len(self.execution_summary),
            "successful_steps": len(successful_steps),
            "failed_steps": len(failed_steps),
            "total_execution_time": self.execution_time,
            "rows_changed": self.transformed_df.shape[0] - self.original_df.shape[0],
            "columns_changed": self.transformed_df.shape[1] - self.original_df.shape[1],
            "step_details": self.execution_summary
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "transformation_summary": self.get_transformation_summary(),
            "transformation_config": self.transformation_config,
            "sample_data": {
                "original": self.original_df.head(5).to_dict('records') if len(self.original_df) > 0 else [],
                "transformed": self.transformed_df.head(5).to_dict('records') if len(self.transformed_df) > 0 else []
            }
        }