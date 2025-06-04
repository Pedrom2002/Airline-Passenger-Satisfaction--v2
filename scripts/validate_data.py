#!/usr/bin/env python3
"""
Data Validation Script
Validates data quality and integrity for model training and inference
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import great_expectations as ge
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation for airline satisfaction data"""
    
    def __init__(self, config_path: str = 'config/validation_rules.json'):
        """Initialize validator with configuration"""
        self.config = self._load_config(config_path)
        self.validation_results = []
        self.ge_context = ge.data_context.DataContext()
        
    def _load_config(self, config_path: str) -> dict:
        """Load validation configuration"""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default validation rules
            return {
                'columns': {
                    'required': [
                        'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
                        'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
                        'Ease of Online booking', 'Gate location', 'Food and drink',
                        'Online boarding', 'Seat comfort', 'Inflight entertainment',
                        'On-board service', 'Leg room service', 'Baggage handling',
                        'Checkin service', 'Inflight service', 'Cleanliness',
                        'Departure Delay in Minutes', 'Arrival Delay in Minutes'
                    ],
                    'target': 'satisfaction'
                },
                'data_types': {
                    'Gender': 'object',
                    'Customer Type': 'object',
                    'Age': 'int64',
                    'Type of Travel': 'object',
                    'Class': 'object',
                    'Flight Distance': 'int64',
                    'Departure Delay in Minutes': 'float64',
                    'Arrival Delay in Minutes': 'float64'
                },
                'value_ranges': {
                    'Age': {'min': 0, 'max': 100},
                    'Flight Distance': {'min': 0, 'max': 10000},
                    'service_ratings': {'min': 1, 'max': 5},
                    'delays': {'min': 0, 'max': 2000}
                },
                'categorical_values': {
                    'Gender': ['Male', 'Female'],
                    'Customer Type': ['Loyal Customer', 'Disloyal Customer'],
                    'Type of Travel': ['Business travel', 'Personal Travel'],
                    'Class': ['Eco', 'Eco Plus', 'Business'],
                    'satisfaction': ['satisfied', 'neutral or dissatisfied']
                },
                'missing_threshold': 0.05,  # 5% maximum missing values
                'outlier_threshold': 3,     # 3 standard deviations
                'imbalance_threshold': 0.7  # 70% maximum class imbalance
            }
    
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataframe schema"""
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required columns
        required_cols = self.config['columns']['required']
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            results['valid'] = False
            results['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Check for unexpected columns
        expected_cols = set(required_cols + [self.config['columns'].get('target', 'satisfaction')])
        extra_cols = set(df.columns) - expected_cols
        
        if extra_cols:
            results['warnings'].append(f"Unexpected columns found: {extra_cols}")
        
        # Check data types
        for col, expected_type in self.config['data_types'].items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    # Try to convert
                    try:
                        if expected_type == 'int64':
                            df[col] = df[col].astype('int64')
                        elif expected_type == 'float64':
                            df[col] = df[col].astype('float64')
                        results['warnings'].append(
                            f"Column '{col}' converted from {actual_type} to {expected_type}"
                        )
                    except:
                        results['errors'].append(
                            f"Column '{col}' has type {actual_type}, expected {expected_type}"
                        )
        
        return results
    
    def validate_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate missing values"""
        results = {
            'valid': True,
            'missing_summary': {},
            'errors': [],
            'warnings': []
        }
        
        # Calculate missing values
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        # Check threshold
        threshold = self.config.get('missing_threshold', 0.05) * 100
        
        for col, pct in missing_percentages.items():
            if pct > 0:
                results['missing_summary'][col] = {
                    'count': int(missing_counts[col]),
                    'percentage': float(pct)
                }
                
                if pct > threshold:
                    results['valid'] = False
                    results['errors'].append(
                        f"Column '{col}' has {pct:.2f}% missing values (threshold: {threshold}%)"
                    )
                elif pct > 0:
                    results['warnings'].append(
                        f"Column '{col}' has {pct:.2f}% missing values"
                    )
        
        return results
    
    def validate_value_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate value ranges for numeric columns"""
        results = {
            'valid': True,
            'out_of_range': {},
            'errors': [],
            'warnings': []
        }
        
        # Age validation
        if 'Age' in df.columns:
            age_range = self.config['value_ranges']['Age']
            invalid_age = df[(df['Age'] < age_range['min']) | (df['Age'] > age_range['max'])]
            if len(invalid_age) > 0:
                results['out_of_range']['Age'] = len(invalid_age)
                results['errors'].append(
                    f"Found {len(invalid_age)} records with invalid age values"
                )
        
        # Flight distance validation
        if 'Flight Distance' in df.columns:
            dist_range = self.config['value_ranges']['Flight Distance']
            invalid_dist = df[(df['Flight Distance'] < dist_range['min']) | 
                            (df['Flight Distance'] > dist_range['max'])]
            if len(invalid_dist) > 0:
                results['out_of_range']['Flight Distance'] = len(invalid_dist)
                results['warnings'].append(
                    f"Found {len(invalid_dist)} records with unusual flight distances"
                )
        
        # Service rating columns (1-5 scale)
        service_cols = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['service', 'comfort', 'cleanliness', 
                                     'entertainment', 'food', 'boarding'])]
        
        rating_range = self.config['value_ranges']['service_ratings']
        for col in service_cols:
            if col in df.columns:
                invalid_ratings = df[(df[col] < rating_range['min']) | 
                                   (df[col] > rating_range['max'])]
                if len(invalid_ratings) > 0:
                    results['out_of_range'][col] = len(invalid_ratings)
                    results['errors'].append(
                        f"Column '{col}' has {len(invalid_ratings)} invalid ratings"
                    )
        
        # Delay validation
        delay_cols = ['Departure Delay in Minutes', 'Arrival Delay in Minutes']
        delay_range = self.config['value_ranges']['delays']
        
        for col in delay_cols:
            if col in df.columns:
                # Check for negative delays
                negative_delays = df[df[col] < 0]
                if len(negative_delays) > 0:
                    results['errors'].append(
                        f"Column '{col}' has {len(negative_delays)} negative values"
                    )
                
                # Check for extreme delays
                extreme_delays = df[df[col] > delay_range['max']]
                if len(extreme_delays) > 0:
                    results['warnings'].append(
                        f"Column '{col}' has {len(extreme_delays)} extreme values (>{delay_range['max']} min)"
                    )
        
        return results
    
    def validate_categorical_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate categorical values"""
        results = {
            'valid': True,
            'invalid_categories': {},
            'errors': [],
            'warnings': []
        }
        
        for col, valid_values in self.config['categorical_values'].items():
            if col in df.columns:
                unique_values = df[col].unique()
                invalid_values = set(unique_values) - set(valid_values)
                
                if invalid_values:
                    invalid_count = df[df[col].isin(invalid_values)].shape[0]
                    results['invalid_categories'][col] = {
                        'invalid_values': list(invalid_values),
                        'count': invalid_count
                    }
                    results['errors'].append(
                        f"Column '{col}' contains invalid values: {invalid_values}"
                    )
        
        return results
    
    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        results = {
            'outliers_detected': {},
            'warnings': []
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df.columns:
                # Z-score method
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                z_threshold = self.config.get('outlier_threshold', 3)
                z_outliers = (z_scores > z_threshold).sum()
                
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                iqr_outliers = ((df[col] < (Q1 - 1.5 * IQR)) | 
                               (df[col] > (Q3 + 1.5 * IQR))).sum()
                
                if z_outliers > 0 or iqr_outliers > 0:
                    results['outliers_detected'][col] = {
                        'z_score_outliers': int(z_outliers),
                        'iqr_outliers': int(iqr_outliers),
                        'percentage': float(max(z_outliers, iqr_outliers) / len(df) * 100)
                    }
                    
                    if results['outliers_detected'][col]['percentage'] > 5:
                        results['warnings'].append(
                            f"Column '{col}' has {results['outliers_detected'][col]['percentage']:.2f}% outliers"
                        )
        
        return results
    
    def validate_data_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate logical consistency in data"""
        results = {
            'valid': True,
            'inconsistencies': [],
            'warnings': []
        }
        
        # Check delay consistency
        if all(col in df.columns for col in ['Departure Delay in Minutes', 'Arrival Delay in Minutes']):
            # Arrival delay should generally be >= departure delay (with some exceptions)
            inconsistent_delays = df[
                (df['Arrival Delay in Minutes'] < df['Departure Delay in Minutes'] - 30)
            ]
            if len(inconsistent_delays) > 0:
                results['warnings'].append(
                    f"Found {len(inconsistent_delays)} records where arrival delay is significantly less than departure delay"
                )
        
        # Check age and travel type consistency
        if 'Age' in df.columns and 'Type of Travel' in df.columns:
            # Very young business travelers might be unusual
            young_business = df[(df['Age'] < 18) & (df['Type of Travel'] == 'Business travel')]
            if len(young_business) > 0:
                results['warnings'].append(
                    f"Found {len(young_business)} business travelers under 18 years old"
                )
        
        # Check service rating consistency
        service_cols = [col for col in df.columns if 'service' in col.lower() or 
                       'comfort' in col.lower() or 'cleanliness' in col.lower()]
        
        if len(service_cols) > 1:
            # Calculate standard deviation of ratings per row
            rating_std = df[service_cols].std(axis=1)
            
            # Flag records where all ratings are identical (std = 0)
            identical_ratings = df[rating_std == 0]
            if len(identical_ratings) > len(df) * 0.05:  # More than 5%
                results['warnings'].append(
                    f"Found {len(identical_ratings)} records with identical ratings across all services"
                )
        
        return results
    
    def check_class_balance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check class balance for target variable"""
        results = {
            'balanced': True,
            'class_distribution': {},
            'warnings': []
        }
        
        target_col = self.config['columns'].get('target', 'satisfaction')
        
        if target_col in df.columns:
            class_counts = df[target_col].value_counts()
            class_percentages = (class_counts / len(df)) * 100
            
            results['class_distribution'] = {
                str(k): {'count': int(v), 'percentage': float(class_percentages[k])}
                for k, v in class_counts.items()
            }
            
            # Check imbalance
            max_percentage = class_percentages.max()
            threshold = self.config.get('imbalance_threshold', 0.7) * 100
            
            if max_percentage > threshold:
                results['balanced'] = False
                results['warnings'].append(
                    f"Class imbalance detected: {class_percentages.idxmax()} represents {max_percentage:.1f}% of data"
                )
        
        return results
    
    def generate_validation_report(self, df: pd.DataFrame, 
                                 output_path: str = 'validation_report.html') -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        logger.info("Starting data validation...")
        
        # Run all validations
        schema_results = self.validate_schema(df)
        missing_results = self.validate_missing_values(df)
        range_results = self.validate_value_ranges(df)
        categorical_results = self.validate_categorical_values(df)
        outlier_results = self.detect_outliers(df)
        consistency_results = self.validate_data_consistency(df)
        balance_results = self.check_class_balance(df)
        
        # Compile overall results
        overall_valid = all([
            schema_results['valid'],
            missing_results['valid'],
            range_results['valid'],
            categorical_results['valid'],
            consistency_results['valid']
        ])
        
        total_errors = len(schema_results['errors']) + len(missing_results['errors']) + \
                      len(range_results['errors']) + len(categorical_results['errors'])
        
        total_warnings = len(schema_results['warnings']) + len(missing_results['warnings']) + \
                        len(range_results['warnings']) + len(outlier_results['warnings']) + \
                        len(consistency_results['warnings']) + len(balance_results['warnings'])
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Validation Report', fontsize=16)
        
        # 1. Missing values heatmap
        ax1 = axes[0, 0]
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            missing_data.plot(kind='barh', ax=ax1, color='coral')
            ax1.set_title('Missing Values by Column')
            ax1.set_xlabel('Number of Missing Values')
        else:
            ax1.text(0.5, 0.5, 'No Missing Values', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=14, color='green')
            ax1.set_title('Missing Values Check')
        
        # 2. Numeric distributions
        ax2 = axes[0, 1]
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # First 5 numeric columns
        
        if len(numeric_cols) > 0:
            df[numeric_cols].hist(ax=ax2, bins=20, alpha=0.7)
            ax2.set_title('Numeric Feature Distributions (Sample)')
        
        # 3. Class distribution
        ax3 = axes[1, 0]
        target_col = self.config['columns'].get('target', 'satisfaction')
        
        if target_col in df.columns:
            class_counts = df[target_col].value_counts()
            class_counts.plot(kind='pie', ax=ax3, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
            ax3.set_title('Target Variable Distribution')
            ax3.set_ylabel('')
        
        # 4. Validation summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        Validation Summary
        
        Total Records: {len(df):,}
        Total Columns: {len(df.columns)}
        
        ✓ Valid: {overall_valid}
        ⚠ Errors: {total_errors}
        ℹ Warnings: {total_warnings}
        
        Missing Data: {len(missing_results['missing_summary'])} columns
        Outliers: {len(outlier_results['outliers_detected'])} columns
        Class Balance: {'Balanced' if balance_results['balanced'] else 'Imbalanced'}
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Generate HTML report
        html_content = f"""
        <html>
        <head>
            <title>Data Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                .valid {{ color: green; font-weight: bold; }}
                .invalid {{ color: red; font-weight: bold; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
                .section {{ 
                    background-color: #f5f5f5; 
                    padding: 20px; 
                    margin: 20px 0;
                    border-radius: 5px;
                    border-left: 4px solid #007bff;
                }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%;
                    margin: 10px 0;
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 8px; 
                    text-align: left;
                }}
                th {{ 
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                .metric {{ 
                    display: inline-block;
                    margin: 10px 20px 10px 0;
                    padding: 10px;
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
            </style>
        </head>
        <body>
            <h1>📊 Data Validation Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Dataset:</strong> {len(df):,} records, {len(df.columns)} columns</p>
            
            <div class="section">
                <h2>Overall Status: <span class="{'valid' if overall_valid else 'invalid'}">
                    {'VALID' if overall_valid else 'INVALID'}</span></h2>
                
                <div class="metric">
                    <strong>Errors:</strong> <span class="error">{total_errors}</span>
                </div>
                <div class="metric">
                    <strong>Warnings:</strong> <span class="warning">{total_warnings}</span>
                </div>
            </div>
        """
        
        # Add detailed results for each validation
        validations = [
            ("Schema Validation", schema_results),
            ("Missing Values", missing_results),
            ("Value Ranges", range_results),
            ("Categorical Values", categorical_results),
            ("Outlier Detection", outlier_results),
            ("Data Consistency", consistency_results),
            ("Class Balance", balance_results)
        ]
        
        for title, results in validations:
            html_content += f"""
            <div class="section">
                <h2>{title}</h2>
            """
            
            # Add errors
            if 'errors' in results and results['errors']:
                html_content += "<h3>Errors:</h3><ul>"
                for error in results['errors']:
                    html_content += f'<li class="error">{error}</li>'
                html_content += "</ul>"
            
            # Add warnings
            if 'warnings' in results and results['warnings']:
                html_content += "<h3>Warnings:</h3><ul>"
                for warning in results['warnings']:
                    html_content += f'<li class="warning">{warning}</li>'
                html_content += "</ul>"
            
            # Add specific details based on validation type
            if title == "Missing Values" and results['missing_summary']:
                html_content += """
                <h3>Missing Data Summary:</h3>
                <table>
                    <tr><th>Column</th><th>Missing Count</th><th>Percentage</th></tr>
                """
                for col, info in results['missing_summary'].items():
                    html_content += f"""
                    <tr>
                        <td>{col}</td>
                        <td>{info['count']}</td>
                        <td>{info['percentage']:.2f}%</td>
                    </tr>
                    """
                html_content += "</table>"
            
            elif title == "Class Balance" and results['class_distribution']:
                html_content += """
                <h3>Class Distribution:</h3>
                <table>
                    <tr><th>Class</th><th>Count</th><th>Percentage</th></tr>
                """
                for class_name, info in results['class_distribution'].items():
                    html_content += f"""
                    <tr>
                        <td>{class_name}</td>
                        <td>{info['count']:,}</td>
                        <td>{info['percentage']:.1f}%</td>
                    </tr>
                    """
                html_content += "</table>"
            
            html_content += "</div>"
        
        html_content += """
            <div class="section">
                <h2>Visualizations</h2>
                <img src="validation_plots.png" alt="Validation Charts" style="max-width: 100%;">
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
        """
        
        # Add recommendations
        if total_errors > 0:
            html_content += "<li><strong>Fix all errors before proceeding with model training</strong></li>"
        
        if len(missing_results['missing_summary']) > 0:
            html_content += "<li>Handle missing values through imputation or removal</li>"
        
        if len(outlier_results['outliers_detected']) > 0:
            html_content += "<li>Review outliers and decide on treatment strategy</li>"
        
        if not balance_results['balanced']:
            html_content += "<li>Consider using class balancing techniques (SMOTE, class weights)</li>"
        
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save report and plots
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        plot_path = output_path.replace('.html', '_plots.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Validation report saved to {output_path}")
        
        return {
            'valid': overall_valid,
            'errors': total_errors,
            'warnings': total_warnings,
            'report_path': output_path,
            'details': {
                'schema': schema_results,
                'missing': missing_results,
                'ranges': range_results,
                'categorical': categorical_results,
                'outliers': outlier_results,
                'consistency': consistency_results,
                'balance': balance_results
            }
        }
    
    def validate_prediction_input(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Quick validation for prediction inputs"""
        errors = []
        
        # Check required fields
        required_fields = self.config['columns']['required']
        missing_fields = set(required_fields) - set(data.keys())
        
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
        
        # Validate categorical values
        for field, valid_values in self.config['categorical_values'].items():
            if field in data and data[field] not in valid_values:
                errors.append(f"Invalid value for {field}: {data[field]}")
        
        # Validate numeric ranges
        if 'Age' in data:
            age_range = self.config['value_ranges']['Age']
            if not age_range['min'] <= data['Age'] <= age_range['max']:
                errors.append(f"Age {data['Age']} out of valid range")
        
        # Validate service ratings
        service_fields = [f for f in data.keys() if 'service' in f.lower() or 
                         'comfort' in f.lower() or 'cleanliness' in f.lower()]
        
        rating_range = self.config['value_ranges']['service_ratings']
        for field in service_fields:
            if field in data:
                if not rating_range['min'] <= data[field] <= rating_range['max']:
                    errors.append(f"{field} rating {data[field]} out of valid range (1-5)")
        
        return len(errors) == 0, errors


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate airline satisfaction data')
    parser.add_argument('input_file', help='Input CSV file to validate')
    parser.add_argument('--output', default='validation_report.html',
                       help='Output report path')
    parser.add_argument('--config', default='config/validation_rules.json',
                       help='Validation configuration file')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick validation only')
    
    args = parser.parse_args()
    
    # Load data
    try:
        df = pd.read_csv(args.input_file)
        logger.info(f"Loaded {len(df)} records from {args.input_file}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Initialize validator
    validator = DataValidator(args.config)
    
    if args.quick:
        # Quick validation
        schema_results = validator.validate_schema(df)
        missing_results = validator.validate_missing_values(df)
        
        print("\n=== QUICK VALIDATION RESULTS ===")
        print(f"Schema Valid: {schema_results['valid']}")
        print(f"Missing Values: {len(missing_results['missing_summary'])} columns")
        
        if schema_results['errors']:
            print("\nErrors:")
            for error in schema_results['errors']:
                print(f"  - {error}")
    else:
        # Full validation
        results = validator.generate_validation_report(df, args.output)
        
        print("\n=== VALIDATION COMPLETE ===")
        print(f"Overall Valid: {results['valid']}")
        print(f"Total Errors: {results['errors']}")
        print(f"Total Warnings: {results['warnings']}")
        print(f"\nDetailed report saved to: {results['report_path']}")


if __name__ == "__main__":
    main()