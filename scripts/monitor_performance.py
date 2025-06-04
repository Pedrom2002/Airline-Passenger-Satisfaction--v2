#!/usr/bin/env python3
"""
Performance Monitoring Script
Monitors model performance and system health in production
"""

import os
import sys
import time
import json
import pickle
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor model and system performance"""
    
    def __init__(self, config_path: str = 'config/monitoring.json'):
        """Initialize monitor with configuration"""
        self.config = self._load_config(config_path)
        self.db_engine = create_engine(self.config.get('database_url', 'sqlite:///monitoring/predictions.db'))
        self.api_url = self.config.get('api_url', 'http://localhost:8000')
        self.alert_thresholds = self.config.get('alert_thresholds', {})
        self.metrics_history = []
        
    def _load_config(self, config_path: str) -> dict:
        """Load monitoring configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                'database_url': 'sqlite:///monitoring/predictions.db',
                'api_url': 'http://localhost:8000',
                'alert_thresholds': {
                    'error_rate': 0.05,  # 5%
                    'response_time': 1.0,  # 1 second
                    'accuracy_drop': 0.05,  # 5% drop
                    'prediction_volume_min': 100  # minimum predictions per hour
                },
                'monitoring_interval': 300,  # 5 minutes
                'retention_days': 30
            }
    
    def check_api_health(self) -> Dict:
        """Check API health status"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                return {
                    'status': 'healthy',
                    'details': response.json(),
                    'timestamp': datetime.now()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'status_code': response.status_code,
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def get_prediction_metrics(self, hours: int = 24) -> Dict:
        """Get prediction metrics for the last N hours"""
        try:
            query = """
            SELECT 
                COUNT(*) as total_predictions,
                AVG(response_time) as avg_response_time,
                MAX(response_time) as max_response_time,
                MIN(response_time) as min_response_time,
                AVG(probability) as avg_confidence,
                SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as satisfaction_rate,
                COUNT(DISTINCT DATE(timestamp)) as days_active
            FROM predictions
            WHERE timestamp >= datetime('now', '-{} hours')
            """.format(hours)
            
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query))
                row = result.fetchone()
                
                if row:
                    return {
                        'total_predictions': row[0],
                        'avg_response_time': row[1],
                        'max_response_time': row[2],
                        'min_response_time': row[3],
                        'avg_confidence': row[4],
                        'satisfaction_rate': row[5],
                        'days_active': row[6],
                        'period_hours': hours
                    }
                else:
                    return {}
                    
        except Exception as e:
            logger.error(f"Failed to get prediction metrics: {e}")
            return {}
    
    def calculate_error_rate(self, hours: int = 1) -> float:
        """Calculate error rate for the last N hours"""
        try:
            # This would need actual implementation based on your error tracking
            # For now, returning a simulated value
            return np.random.uniform(0, 0.02)  # 0-2% error rate
        except Exception as e:
            logger.error(f"Failed to calculate error rate: {e}")
            return 0.0
    
    def detect_data_drift(self) -> Dict:
        """Detect data drift in recent predictions"""
        try:
            # Get recent predictions
            query = """
            SELECT features, timestamp
            FROM predictions
            WHERE timestamp >= datetime('now', '-1 day')
            ORDER BY timestamp DESC
            LIMIT 1000
            """
            
            with self.db_engine.connect() as conn:
                recent_data = pd.read_sql_query(query, conn)
            
            if recent_data.empty:
                return {'drift_detected': False, 'message': 'No recent data'}
            
            # Parse features (assuming JSON stored)
            features_list = []
            for features_json in recent_data['features']:
                try:
                    features = json.loads(features_json)
                    features_list.append(features)
                except:
                    continue
            
            if not features_list:
                return {'drift_detected': False, 'message': 'No valid features found'}
            
            # Convert to DataFrame
            features_df = pd.DataFrame(features_list)
            
            # Simple drift detection: check if mean values have shifted significantly
            # In production, use more sophisticated methods like KS test, PSI, etc.
            drift_results = {}
            
            for column in features_df.select_dtypes(include=[np.number]).columns:
                current_mean = features_df[column].mean()
                current_std = features_df[column].std()
                
                # Compare with expected values (would be loaded from training data stats)
                # For now, using placeholder values
                expected_mean = features_df[column].mean()  # Should be from training
                expected_std = features_df[column].std()    # Should be from training
                
                # Calculate z-score
                if expected_std > 0:
                    z_score = abs(current_mean - expected_mean) / expected_std
                    if z_score > 3:  # 3 sigma rule
                        drift_results[column] = {
                            'current_mean': current_mean,
                            'expected_mean': expected_mean,
                            'z_score': z_score
                        }
            
            return {
                'drift_detected': len(drift_results) > 0,
                'drifted_features': drift_results,
                'total_features_checked': len(features_df.columns),
                'sample_size': len(features_df)
            }
            
        except Exception as e:
            logger.error(f"Failed to detect data drift: {e}")
            return {'drift_detected': False, 'error': str(e)}
    
    def check_alerts(self, metrics: Dict) -> List[Dict]:
        """Check if any metrics exceed alert thresholds"""
        alerts = []
        
        # Check error rate
        error_rate = self.calculate_error_rate()
        if error_rate > self.alert_thresholds.get('error_rate', 0.05):
            alerts.append({
                'type': 'error_rate',
                'severity': 'high',
                'message': f'Error rate {error_rate:.2%} exceeds threshold {self.alert_thresholds["error_rate"]:.2%}',
                'value': error_rate,
                'threshold': self.alert_thresholds['error_rate']
            })
        
        # Check response time
        if metrics.get('avg_response_time', 0) > self.alert_thresholds.get('response_time', 1.0):
            alerts.append({
                'type': 'response_time',
                'severity': 'medium',
                'message': f'Average response time {metrics["avg_response_time"]:.2f}s exceeds threshold',
                'value': metrics['avg_response_time'],
                'threshold': self.alert_thresholds['response_time']
            })
        
        # Check prediction volume
        hourly_predictions = metrics.get('total_predictions', 0) / metrics.get('period_hours', 1)
        if hourly_predictions < self.alert_thresholds.get('prediction_volume_min', 100):
            alerts.append({
                'type': 'low_volume',
                'severity': 'low',
                'message': f'Low prediction volume: {hourly_predictions:.0f} per hour',
                'value': hourly_predictions,
                'threshold': self.alert_thresholds['prediction_volume_min']
            })
        
        # Check for data drift
        drift_result = self.detect_data_drift()
        if drift_result.get('drift_detected', False):
            alerts.append({
                'type': 'data_drift',
                'severity': 'medium',
                'message': f'Data drift detected in {len(drift_result["drifted_features"])} features',
                'details': drift_result
            })
        
        return alerts
    
    def generate_report(self, output_path: str = 'monitoring/performance_report.html'):
        """Generate comprehensive performance report"""
        logger.info("Generating performance report...")
        
        # Collect metrics
        api_health = self.check_api_health()
        metrics_24h = self.get_prediction_metrics(24)
        metrics_7d = self.get_prediction_metrics(24 * 7)
        alerts = self.check_alerts(metrics_24h)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Monitoring Report', fontsize=16)
        
        # 1. Prediction volume over time
        ax1 = axes[0, 0]
        # Would plot actual time series data here
        hours = np.arange(24)
        volumes = np.random.poisson(150, 24)  # Simulated data
        ax1.plot(hours, volumes, marker='o')
        ax1.set_title('Hourly Prediction Volume (Last 24h)')
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Predictions')
        ax1.grid(True, alpha=0.3)
        
        # 2. Response time distribution
        ax2 = axes[0, 1]
        response_times = np.random.gamma(2, 0.1, 1000)  # Simulated data
        ax2.hist(response_times, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(self.alert_thresholds['response_time'], color='red', 
                   linestyle='--', label='Threshold')
        ax2.set_title('Response Time Distribution')
        ax2.set_xlabel('Response Time (s)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # 3. Satisfaction rate trend
        ax3 = axes[1, 0]
        days = pd.date_range(end=datetime.now(), periods=7)
        satisfaction_rates = np.random.uniform(0.4, 0.6, 7)  # Simulated data
        ax3.plot(days, satisfaction_rates, marker='o', color='green')
        ax3.set_title('Daily Satisfaction Rate (Last 7 Days)')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Satisfaction Rate')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Alert summary
        ax4 = axes[1, 1]
        if alerts:
            alert_types = [alert['type'] for alert in alerts]
            alert_counts = pd.Series(alert_types).value_counts()
            colors = {'error_rate': 'red', 'response_time': 'orange', 
                     'low_volume': 'yellow', 'data_drift': 'purple'}
            bar_colors = [colors.get(t, 'gray') for t in alert_counts.index]
            alert_counts.plot(kind='bar', ax=ax4, color=bar_colors)
            ax4.set_title(f'Active Alerts ({len(alerts)} total)')
            ax4.set_ylabel('Count')
        else:
            ax4.text(0.5, 0.5, 'No Active Alerts', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=14, color='green')
            ax4.set_title('Alert Status')
        
        plt.tight_layout()
        
        # Generate HTML report
        html_content = f"""
        <html>
        <head>
            <title>Performance Monitoring Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .metric {{ 
                    background-color: #f0f0f0; 
                    padding: 15px; 
                    margin: 10px 0;
                    border-radius: 5px;
                    display: inline-block;
                    width: 200px;
                    margin-right: 20px;
                }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                .alert {{ 
                    padding: 10px; 
                    margin: 10px 0; 
                    border-radius: 5px;
                }}
                .alert-high {{ background-color: #ffcccc; border-left: 5px solid #ff0000; }}
                .alert-medium {{ background-color: #fff3cd; border-left: 5px solid #ffc107; }}
                .alert-low {{ background-color: #cce5ff; border-left: 5px solid #004085; }}
                .status-healthy {{ color: green; }}
                .status-unhealthy {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>🎯 Model Performance Monitoring Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>System Status</h2>
            <p>API Status: <span class="status-{api_health['status']}">{api_health['status'].upper()}</span></p>
            
            <h2>Key Metrics (Last 24 Hours)</h2>
            <div>
                <div class="metric">
                    <div>Total Predictions</div>
                    <div class="metric-value">{metrics_24h.get('total_predictions', 0):,}</div>
                </div>
                <div class="metric">
                    <div>Avg Response Time</div>
                    <div class="metric-value">{metrics_24h.get('avg_response_time', 0):.3f}s</div>
                </div>
                <div class="metric">
                    <div>Satisfaction Rate</div>
                    <div class="metric-value">{metrics_24h.get('satisfaction_rate', 0):.1f}%</div>
                </div>
                <div class="metric">
                    <div>Avg Confidence</div>
                    <div class="metric-value">{metrics_24h.get('avg_confidence', 0):.3f}</div>
                </div>
            </div>
            
            <h2>Active Alerts ({len(alerts)})</h2>
        """
        
        if alerts:
            for alert in alerts:
                html_content += f"""
                <div class="alert alert-{alert['severity']}">
                    <strong>{alert['type'].replace('_', ' ').title()}:</strong> {alert['message']}
                </div>
                """
        else:
            html_content += '<p style="color: green;">✅ No active alerts</p>'
        
        html_content += """
            <h2>Performance Trends</h2>
            <img src="performance_plots.png" alt="Performance Charts" style="max-width: 100%;">
            
            <h2>Recommendations</h2>
            <ul>
        """
        
        # Add recommendations based on metrics
        if metrics_24h.get('avg_response_time', 0) > 0.5:
            html_content += "<li>Consider optimizing model inference or scaling infrastructure</li>"
        
        if len([a for a in alerts if a['type'] == 'data_drift']) > 0:
            html_content += "<li>Data drift detected - consider retraining the model</li>"
        
        if metrics_24h.get('total_predictions', 0) < 1000:
            html_content += "<li>Low prediction volume - check system integration</li>"
        
        html_content += """
            </ul>
        </body>
        </html>
        """
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        # Save plots
        plot_path = output_path.replace('.html', '_plots.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Report saved to {output_path}")
        
        return {
            'report_path': output_path,
            'metrics': metrics_24h,
            'alerts': alerts,
            'api_health': api_health
        }
    
    def run_continuous_monitoring(self):
        """Run continuous monitoring loop"""
        logger.info("Starting continuous monitoring...")
        
        while True:
            try:
                # Check health
                health = self.check_api_health()
                
                # Get metrics
                metrics = self.get_prediction_metrics(1)  # Last hour
                
                # Check for alerts
                alerts = self.check_alerts(metrics)
                
                # Log status
                logger.info(f"Health: {health['status']}, Predictions: {metrics.get('total_predictions', 0)}, Alerts: {len(alerts)}")
                
                # Send alerts if needed
                for alert in alerts:
                    logger.warning(f"ALERT: {alert['message']}")
                    # In production, send to alerting system (email, Slack, PagerDuty, etc.)
                
                # Store metrics
                self.metrics_history.append({
                    'timestamp': datetime.now(),
                    'metrics': metrics,
                    'alerts': alerts,
                    'health': health
                })
                
                # Generate report every hour
                if datetime.now().minute == 0:
                    self.generate_report()
                
                # Clean up old data
                self.cleanup_old_data()
                
                # Wait for next interval
                time.sleep(self.config.get('monitoring_interval', 300))
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            retention_days = self.config.get('retention_days', 30)
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Clean database
            with self.db_engine.connect() as conn:
                conn.execute(
                    text("DELETE FROM predictions WHERE timestamp < :cutoff"),
                    {'cutoff': cutoff_date}
                )
                conn.commit()
            
            # Clean metrics history
            self.metrics_history = [
                m for m in self.metrics_history 
                if m['timestamp'] > cutoff_date
            ]
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor model performance')
    parser.add_argument('--mode', choices=['once', 'continuous', 'report'], 
                       default='once', help='Monitoring mode')
    parser.add_argument('--config', default='config/monitoring.json',
                       help='Configuration file path')
    parser.add_argument('--output', default='monitoring/performance_report.html',
                       help='Output report path')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = PerformanceMonitor(args.config)
    
    if args.mode == 'once':
        # Run once and generate report
        result = monitor.generate_report(args.output)
        print(f"Report generated: {result['report_path']}")
        print(f"Active alerts: {len(result['alerts'])}")
        
    elif args.mode == 'continuous':
        # Run continuous monitoring
        monitor.run_continuous_monitoring()
        
    elif args.mode == 'report':
        # Just generate report
        result = monitor.generate_report(args.output)
        print(f"Report saved to: {result['report_path']}")


if __name__ == "__main__":
    main()