"""
Data processing and validation services.
"""
import pandas as pd
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple, Union
from io import BytesIO, StringIO
import chardet
from datetime import datetime

from ..domain.entities import (
    DataFormat, DataSchema, DataQualityReport, DataQualityIssue, 
    DataQualityIssueType, DataDomainService
)


class DataFormatProcessor:
    """Processes different data formats with enhanced error handling."""
    
    @staticmethod
    async def read_csv(file_content: bytes, encoding: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Read CSV file content with enhanced error handling."""
        if encoding is None:
            # Detect encoding
            detected = chardet.detect(file_content)
            encoding = detected.get('encoding', 'utf-8')
            confidence = detected.get('confidence', 0)
            
            # If confidence is low, try common encodings
            if confidence < 0.7:
                for fallback_encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        test_decode = file_content[:1000].decode(fallback_encoding)
                        encoding = fallback_encoding
                        break
                    except UnicodeDecodeError:
                        continue
        
        # CSV parsing options
        csv_options = {
            'sep': kwargs.get('delimiter', ','),
            'encoding': encoding,
            'on_bad_lines': kwargs.get('on_bad_lines', 'warn'),
            'dtype': str,  # Read everything as string initially
            'keep_default_na': False,  # Don't convert to NaN automatically
            'na_values': kwargs.get('na_values', ['', 'NULL', 'null', 'None', 'N/A', 'n/a']),
            'skipinitialspace': kwargs.get('skipinitialspace', True),
            'skip_blank_lines': kwargs.get('skip_blank_lines', True)
        }
        
        try:
            # Try with detected encoding
            content_str = file_content.decode(encoding)
            df = pd.read_csv(StringIO(content_str), **csv_options)
            
            # Post-process: attempt to infer better data types
            df = DataFormatProcessor._infer_csv_types(df)
            
            return df
            
        except UnicodeDecodeError as e:
            # Fallback to utf-8 with error handling
            content_str = file_content.decode('utf-8', errors='replace')
            df = pd.read_csv(StringIO(content_str), **csv_options)
            df = DataFormatProcessor._infer_csv_types(df)
            return df
            
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty or contains no data")
        except pd.errors.ParserError as e:
            raise ValueError(f"CSV parsing error: {str(e)}")
    
    @staticmethod
    def _infer_csv_types(df: pd.DataFrame) -> pd.DataFrame:
        """Infer better data types for CSV data."""
        for col in df.columns:
            # Skip if column is already non-object
            if df[col].dtype != 'object':
                continue
            
            # Try to convert to numeric
            try:
                # Check if all non-null values are numeric
                non_null_values = df[col].dropna()
                if len(non_null_values) == 0:
                    continue
                
                # Try integer first
                try:
                    converted = pd.to_numeric(non_null_values, downcast='integer')
                    if (converted == non_null_values.astype(float)).all():
                        df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
                        continue
                except (ValueError, TypeError):
                    pass
                
                # Try float
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    continue
                except (ValueError, TypeError):
                    pass
                
            except (ValueError, TypeError):
                pass
            
            # Try datetime
            try:
                # Only try if values look like dates
                sample_values = non_null_values.head(10).astype(str)
                if any(len(v) > 6 and ('-' in v or '/' in v or ':' in v) for v in sample_values):
                    df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    continue
            except (ValueError, TypeError):
                pass
        
        return df
    
    @staticmethod
    async def read_excel(file_content: bytes, sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Read Excel file content with enhanced error handling."""
        excel_options = {
            'sheet_name': sheet_name or 0,  # Default to first sheet
            'dtype': str,  # Read as strings initially
            'keep_default_na': False,
            'na_values': kwargs.get('na_values', ['', 'NULL', 'null', 'None', 'N/A', 'n/a']),
            'header': kwargs.get('header', 0),
            'skiprows': kwargs.get('skiprows', None),
            'nrows': kwargs.get('nrows', None)
        }
        
        try:
            df = pd.read_excel(BytesIO(file_content), **excel_options)
            
            # Handle Excel-specific issues
            df = DataFormatProcessor._handle_excel_issues(df)
            
            return df
            
        except Exception as e:
            if "No sheet named" in str(e):
                # Try to get available sheet names
                try:
                    excel_file = pd.ExcelFile(BytesIO(file_content))
                    available_sheets = excel_file.sheet_names
                    raise ValueError(f"Sheet '{sheet_name}' not found. Available sheets: {available_sheets}")
                except:
                    raise ValueError(f"Excel file error: {str(e)}")
            else:
                raise ValueError(f"Excel file error: {str(e)}")
    
    @staticmethod
    def _handle_excel_issues(df: pd.DataFrame) -> pd.DataFrame:
        """Handle common Excel-specific issues."""
        # Handle Excel date issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for Excel date serial numbers
                try:
                    # Excel dates are stored as numbers since 1900-01-01
                    sample_values = df[col].dropna().head(10)
                    if len(sample_values) > 0:
                        # Check if values could be Excel dates
                        numeric_values = pd.to_numeric(sample_values, errors='coerce')
                        if not numeric_values.isna().all():
                            # If values are between reasonable date ranges
                            if numeric_values.between(1, 50000).all():
                                # Convert Excel serial dates
                                df[col] = pd.to_datetime(df[col], origin='1899-12-30', unit='D', errors='coerce')
                                continue
                except (ValueError, TypeError):
                    pass
            
            # Handle Excel error values
            if df[col].dtype == 'object':
                error_values = ['#DIV/0!', '#N/A', '#NAME?', '#NULL!', '#NUM!', '#REF!', '#VALUE!']
                for error_val in error_values:
                    df[col] = df[col].replace(error_val, None)
        
        return df
    
    @staticmethod
    async def read_json(file_content: bytes, encoding: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Read JSON file content with enhanced error handling."""
        if encoding is None:
            detected = chardet.detect(file_content)
            encoding = detected.get('encoding', 'utf-8')
        
        try:
            content_str = file_content.decode(encoding)
            data = json.loads(content_str)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of objects - most common case
                if len(data) == 0:
                    return pd.DataFrame()
                
                # Check if all items are objects
                if all(isinstance(item, dict) for item in data):
                    df = pd.DataFrame(data)
                else:
                    # Mixed types in array - convert to single column
                    df = pd.DataFrame({'value': data})
                
            elif isinstance(data, dict):
                # Single object or nested structure
                if all(not isinstance(v, (list, dict)) for v in data.values()):
                    # Simple object - wrap in list
                    df = pd.DataFrame([data])
                else:
                    # Complex nested structure - normalize
                    try:
                        df = pd.json_normalize(data)
                    except Exception:
                        # Fallback: convert to single row with nested columns
                        df = pd.DataFrame([data])
            else:
                # Primitive value
                df = pd.DataFrame({'value': [data]})
            
            # Handle nested JSON strings within the data
            df = DataFormatProcessor._handle_nested_json(df)
            
            return df
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except UnicodeDecodeError as e:
            raise ValueError(f"JSON encoding error: {str(e)}")
    
    @staticmethod
    def _handle_nested_json(df: pd.DataFrame) -> pd.DataFrame:
        """Handle nested JSON strings within DataFrame."""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains JSON strings
                sample_values = df[col].dropna().head(5)
                if len(sample_values) > 0:
                    for val in sample_values:
                        if isinstance(val, str) and (val.strip().startswith(('{', '['))):
                            try:
                                # Try to parse as JSON
                                json.loads(val)
                                # If successful, this column contains JSON strings
                                # For now, leave as is - could be expanded to parse nested JSON
                                break
                            except json.JSONDecodeError:
                                continue
        
        return df
    
    @staticmethod
    async def read_xml(file_content: bytes, encoding: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Read XML file content with enhanced error handling."""
        if encoding is None:
            detected = chardet.detect(file_content)
            encoding = detected.get('encoding', 'utf-8')
        
        try:
            content_str = file_content.decode(encoding)
            
            # Parse XML with error handling
            try:
                root = ET.fromstring(content_str)
            except ET.ParseError as e:
                raise ValueError(f"XML parsing error: {str(e)}")
            
            # Extract data based on XML structure
            records = []
            
            # Try different XML structures
            if len(root) == 0:
                # Root element has no children - might be a single record
                record = DataFormatProcessor._xml_element_to_dict(root)
                if record:
                    records.append(record)
            else:
                # Root has children - each child is likely a record
                for child in root:
                    record = DataFormatProcessor._xml_element_to_dict(child)
                    if record:
                        records.append(record)
            
            if not records:
                # Try alternative: all elements at same level
                all_elements = root.findall('.//*')
                if all_elements:
                    # Group elements by tag name
                    tag_groups = {}
                    for elem in all_elements:
                        if elem.tag not in tag_groups:
                            tag_groups[elem.tag] = []
                        tag_groups[elem.tag].append(elem.text or '')
                    
                    # Create single record from all tags
                    if tag_groups:
                        records.append(tag_groups)
            
            return pd.DataFrame(records) if records else pd.DataFrame()
            
        except UnicodeDecodeError as e:
            raise ValueError(f"XML encoding error: {str(e)}")
        except Exception as e:
            raise ValueError(f"XML processing error: {str(e)}")
    
    @staticmethod
    def _xml_element_to_dict(element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {}
        
        # Add attributes with @ prefix
        for attr_name, attr_value in element.attrib.items():
            result[f"@{attr_name}"] = attr_value
        
        # Add text content
        if element.text and element.text.strip():
            if len(element) == 0:  # No child elements
                return element.text.strip()
            else:
                result['#text'] = element.text.strip()
        
        # Add child elements
        for child in element:
            child_data = DataFormatProcessor._xml_element_to_dict(child)
            
            if child.tag in result:
                # Multiple elements with same tag - convert to list
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        
        return result
    
    @staticmethod
    async def read_tsv(file_content: bytes, encoding: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Read TSV file content with enhanced error handling."""
        if encoding is None:
            detected = chardet.detect(file_content)
            encoding = detected.get('encoding', 'utf-8')
        
        # TSV-specific options
        tsv_options = {
            'sep': '\t',
            'encoding': encoding,
            'on_bad_lines': kwargs.get('on_bad_lines', 'warn'),
            'dtype': str,
            'keep_default_na': False,
            'na_values': kwargs.get('na_values', ['', 'NULL', 'null', 'None', 'N/A', 'n/a']),
            'skipinitialspace': kwargs.get('skipinitialspace', True),
            'skip_blank_lines': kwargs.get('skip_blank_lines', True)
        }
        
        try:
            content_str = file_content.decode(encoding)
            df = pd.read_csv(StringIO(content_str), **tsv_options)
            
            # Post-process for TSV-specific issues
            df = DataFormatProcessor._handle_tsv_issues(df)
            
            return df
            
        except UnicodeDecodeError:
            content_str = file_content.decode('utf-8', errors='replace')
            df = pd.read_csv(StringIO(content_str), **tsv_options)
            df = DataFormatProcessor._handle_tsv_issues(df)
            return df
        except pd.errors.EmptyDataError:
            raise ValueError("TSV file is empty or contains no data")
        except pd.errors.ParserError as e:
            raise ValueError(f"TSV parsing error: {str(e)}")
    
    @staticmethod
    def _handle_tsv_issues(df: pd.DataFrame) -> pd.DataFrame:
        """Handle TSV-specific issues."""
        # Remove embedded tabs from data (they shouldn't be there in TSV)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('\t', ' ', regex=False)
        
        # Infer types similar to CSV
        df = DataFormatProcessor._infer_csv_types(df)
        
        return df
    
    @staticmethod
    async def read_parquet(file_content: bytes, **kwargs) -> pd.DataFrame:
        """Read Parquet file content with enhanced error handling."""
        try:
            df = pd.read_parquet(BytesIO(file_content))
            
            # Parquet files preserve types well, but check for any issues
            df = DataFormatProcessor._handle_parquet_issues(df)
            
            return df
            
        except Exception as e:
            if "not a parquet file" in str(e).lower():
                raise ValueError("File is not a valid Parquet file")
            elif "unsupported parquet" in str(e).lower():
                raise ValueError("Unsupported Parquet file version or format")
            else:
                raise ValueError(f"Parquet file error: {str(e)}")
    
    @staticmethod
    def _handle_parquet_issues(df: pd.DataFrame) -> pd.DataFrame:
        """Handle Parquet-specific issues."""
        # Parquet files usually preserve data types well
        # Check for any obvious corruption or issues
        
        for col in df.columns:
            # Check for mixed types in object columns (shouldn't happen in Parquet)
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(100)
                if len(sample_values) > 0:
                    types_found = set(type(v).__name__ for v in sample_values)
                    if len(types_found) > 2:  # Allow some variation
                        # Try to clean up mixed types
                        df[col] = df[col].astype(str)
        
        return df
    
    @staticmethod
    async def process_file(file_content: bytes, file_format: DataFormat, **kwargs) -> pd.DataFrame:
        """Process file based on format with comprehensive error handling."""
        if not file_content:
            raise ValueError("File content is empty")
        
        # Validate file size
        max_size = kwargs.get('max_file_size_mb', 100)
        if len(file_content) > max_size * 1024 * 1024:
            raise ValueError(f"File size ({len(file_content) / 1024 / 1024:.1f}MB) exceeds maximum allowed size ({max_size}MB)")
        
        processors = {
            DataFormat.CSV: DataFormatProcessor.read_csv,
            DataFormat.EXCEL: DataFormatProcessor.read_excel,
            DataFormat.JSON: DataFormatProcessor.read_json,
            DataFormat.XML: DataFormatProcessor.read_xml,
            DataFormat.TSV: DataFormatProcessor.read_tsv,
            DataFormat.PARQUET: DataFormatProcessor.read_parquet
        }
        
        processor = processors.get(file_format)
        if not processor:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        try:
            # Process the file
            df = await processor(file_content, **kwargs)
            
            # Validate result
            if df is None:
                raise ValueError("File processing returned no data")
            
            if len(df) == 0:
                raise ValueError("File contains no data rows")
            
            if len(df.columns) == 0:
                raise ValueError("File contains no columns")
            
            # Basic cleanup
            df = DataFormatProcessor._basic_cleanup(df)
            
            return df
            
        except ValueError:
            # Re-raise ValueError as-is (these are user-friendly messages)
            raise
        except Exception as e:
            # Wrap other exceptions with format-specific context
            raise ValueError(f"Error processing {file_format.value} file: {str(e)}")
    
    @staticmethod
    def _basic_cleanup(df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic cleanup on the DataFrame."""
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Handle duplicate column names
        cols = df.columns.tolist()
        seen = set()
        for i, col in enumerate(cols):
            if col in seen:
                counter = 1
                new_col = f"{col}_{counter}"
                while new_col in seen:
                    counter += 1
                    new_col = f"{col}_{counter}"
                cols[i] = new_col
            seen.add(cols[i])
        df.columns = cols
        
        return df


class DataValidator:
    """Validates data quality and generates reports."""
    
    @staticmethod
    async def validate_data(df: pd.DataFrame, schema: Optional[DataSchema] = None) -> DataQualityReport:
        """Validate data and generate quality report."""
        issues = []
        
        # Basic statistics
        total_rows = len(df)
        total_columns = len(df.columns)
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            missing_cols = df.columns[df.isnull().any()].tolist()
            issues.append(DataQualityIssue(
                issue_type=DataQualityIssueType.MISSING_VALUES,
                description=f"Found {missing_values} missing values",
                severity="medium",
                affected_columns=missing_cols,
                suggested_fix="Consider filling missing values or removing incomplete rows"
            ))
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            issues.append(DataQualityIssue(
                issue_type=DataQualityIssueType.DUPLICATE_ROWS,
                description=f"Found {duplicate_rows} duplicate rows",
                severity="medium",
                suggested_fix="Remove duplicate rows or investigate data source"
            ))
        
        # Check for inconsistent data types in columns
        for column in df.columns:
            if df[column].dtype == 'object':  # String columns
                # Check if column should be numeric
                non_null_values = df[column].dropna()
                if len(non_null_values) > 0:
                    try:
                        pd.to_numeric(non_null_values)
                        issues.append(DataQualityIssue(
                            issue_type=DataQualityIssueType.INCONSISTENT_TYPES,
                            description=f"Column '{column}' contains numeric data stored as text",
                            severity="low",
                            affected_columns=[column],
                            suggested_fix=f"Convert column '{column}' to numeric type"
                        ))
                    except (ValueError, TypeError):
                        pass
        
        # Schema validation if provided
        if schema:
            schema_issues = await DataValidator._validate_against_schema(df, schema)
            issues.extend(schema_issues)
        
        # Calculate quality score
        quality_score = DataDomainService.calculate_quality_score(
            DataQualityReport(
                total_rows=total_rows,
                total_columns=total_columns,
                missing_values_count=missing_values,
                duplicate_rows_count=duplicate_rows,
                issues=issues,
                quality_score=0  # Will be calculated
            )
        )
        
        return DataQualityReport(
            total_rows=total_rows,
            total_columns=total_columns,
            missing_values_count=missing_values,
            duplicate_rows_count=duplicate_rows,
            issues=issues,
            quality_score=quality_score
        )
    
    @staticmethod
    async def _validate_against_schema(df: pd.DataFrame, schema: DataSchema) -> List[DataQualityIssue]:
        """Validate DataFrame against schema."""
        issues = []
        
        # Check for missing columns
        expected_columns = set(schema.get_column_names())
        actual_columns = set(df.columns)
        
        missing_columns = expected_columns - actual_columns
        if missing_columns:
            issues.append(DataQualityIssue(
                issue_type=DataQualityIssueType.CONSTRAINT_VIOLATION,
                description=f"Missing required columns: {', '.join(missing_columns)}",
                severity="high",
                affected_columns=list(missing_columns),
                suggested_fix="Add missing columns or update schema"
            ))
        
        # Check for extra columns
        extra_columns = actual_columns - expected_columns
        if extra_columns:
            issues.append(DataQualityIssue(
                issue_type=DataQualityIssueType.CONSTRAINT_VIOLATION,
                description=f"Unexpected columns found: {', '.join(extra_columns)}",
                severity="low",
                affected_columns=list(extra_columns),
                suggested_fix="Remove extra columns or update schema"
            ))
        
        # Validate data types for existing columns
        column_types = schema.get_column_types()
        for column in actual_columns.intersection(expected_columns):
            expected_type = column_types.get(column)
            if expected_type and not DataValidator._is_column_type_valid(df[column], expected_type):
                issues.append(DataQualityIssue(
                    issue_type=DataQualityIssueType.INCONSISTENT_TYPES,
                    description=f"Column '{column}' type mismatch. Expected: {expected_type}",
                    severity="medium",
                    affected_columns=[column],
                    suggested_fix=f"Convert column '{column}' to {expected_type} type"
                ))
        
        return issues
    
    @staticmethod
    def _is_column_type_valid(series: pd.Series, expected_type: str) -> bool:
        """Check if column data matches expected type."""
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return True  # Empty column is valid for any type
        
        try:
            if expected_type.lower() == "integer":
                pd.to_numeric(non_null_series, downcast='integer')
                return True
            elif expected_type.lower() == "float":
                pd.to_numeric(non_null_series)
                return True
            elif expected_type.lower() == "boolean":
                # Check if values can be converted to boolean
                bool_values = non_null_series.astype(str).str.lower()
                valid_bool_values = {'true', 'false', '1', '0', 'yes', 'no'}
                return bool_values.isin(valid_bool_values).all()
            elif expected_type.lower() in ["datetime", "date"]:
                pd.to_datetime(non_null_series)
                return True
            else:  # string or unknown type
                return True
        except (ValueError, TypeError):
            return False


class SchemaDetector:
    """Detects and generates data schemas."""
    
    @staticmethod
    async def detect_schema(df: pd.DataFrame, sample_size: int = 1000) -> DataSchema:
        """Detect schema from DataFrame."""
        # Use sample for large datasets
        if len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=42)
        else:
            sample_df = df
        
        # Convert to list of dictionaries for schema generation
        sample_data = sample_df.to_dict('records')
        
        return DataDomainService.generate_schema_from_sample(sample_data)
    
    @staticmethod
    async def suggest_schema_improvements(df: pd.DataFrame, current_schema: DataSchema) -> List[str]:
        """Suggest improvements to current schema."""
        suggestions = []
        
        # Analyze actual data types vs schema types
        column_types = current_schema.get_column_types()
        
        for column in df.columns:
            if column in column_types:
                expected_type = column_types[column]
                actual_series = df[column].dropna()
                
                if len(actual_series) > 0:
                    # Suggest more specific types
                    if expected_type == "string":
                        # Check if it could be a more specific type
                        try:
                            pd.to_numeric(actual_series)
                            if actual_series.dtype == 'int64':
                                suggestions.append(f"Column '{column}' could be 'integer' instead of 'string'")
                            else:
                                suggestions.append(f"Column '{column}' could be 'float' instead of 'string'")
                        except (ValueError, TypeError):
                            try:
                                pd.to_datetime(actual_series)
                                suggestions.append(f"Column '{column}' could be 'datetime' instead of 'string'")
                            except (ValueError, TypeError):
                                pass
        
        return suggestions


class DataProfiler:
    """Generates data profiles and statistics."""
    
    @staticmethod
    async def profile_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile."""
        profile = {
            "basic_stats": {
                "row_count": len(df),
                "column_count": len(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "duplicate_rows": df.duplicated().sum()
            },
            "columns": {}
        }
        
        for column in df.columns:
            col_profile = await DataProfiler._profile_column(df[column])
            profile["columns"][column] = col_profile
        
        return profile
    
    @staticmethod
    async def _profile_column(series: pd.Series) -> Dict[str, Any]:
        """Profile a single column."""
        profile = {
            "dtype": str(series.dtype),
            "null_count": series.isnull().sum(),
            "null_percentage": (series.isnull().sum() / len(series)) * 100,
            "unique_count": series.nunique(),
            "unique_percentage": (series.nunique() / len(series)) * 100 if len(series) > 0 else 0
        }
        
        # Add type-specific statistics
        if pd.api.types.is_numeric_dtype(series):
            profile.update({
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "quartiles": {
                    "q1": series.quantile(0.25),
                    "q3": series.quantile(0.75)
                }
            })
        elif pd.api.types.is_string_dtype(series):
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                profile.update({
                    "min_length": non_null_series.str.len().min(),
                    "max_length": non_null_series.str.len().max(),
                    "avg_length": non_null_series.str.len().mean(),
                    "most_common": non_null_series.value_counts().head(5).to_dict()
                })
        elif pd.api.types.is_datetime64_any_dtype(series):
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                profile.update({
                    "min_date": non_null_series.min(),
                    "max_date": non_null_series.max(),
                    "date_range_days": (non_null_series.max() - non_null_series.min()).days
                })
        
        return profile
    
    
class DataFormatConverter:
    """Handles format conversion operations."""
    
    @staticmethod
    async def convert_dataframe(df: pd.DataFrame, target_format: DataFormat, options: Dict[str, Any] = None) -> bytes:
        """Convert DataFrame to target format."""
        if options is None:
            options = {}
        
        if target_format == DataFormat.CSV:
            output = StringIO()
            df.to_csv(output, index=False, **options)
            return output.getvalue().encode('utf-8')
        
        elif target_format == DataFormat.JSON:
            json_str = df.to_json(orient=options.get('orient', 'records'), **options)
            return json_str.encode('utf-8')
        
        elif target_format == DataFormat.EXCEL:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name=options.get('sheet_name', 'Sheet1'))
            return output.getvalue()
        
        elif target_format == DataFormat.TSV:
            output = StringIO()
            df.to_csv(output, sep='\t', index=False, **options)
            return output.getvalue().encode('utf-8')
        
        elif target_format == DataFormat.PARQUET:
            output = BytesIO()
            df.to_parquet(output, index=False, **options)
            return output.getvalue()
        
        elif target_format == DataFormat.XML:
            # Simple XML conversion
            xml_str = df.to_xml(index=False, **options)
            return xml_str.encode('utf-8')
        
        else:
            raise ValueError(f"Conversion to {target_format} not supported")