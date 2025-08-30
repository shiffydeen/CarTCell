#!/usr/bin/env python3
"""
TensorBoard Metrics Exporter

This script fetches metrics data from a running TensorBoard instance
and exports them to CSV files for analysis.
"""

import requests
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import time
import sys
from urllib.parse import urljoin

class TensorBoardCSVExporter:
    def __init__(self, tensorboard_url="http://localhost:6006", output_dir="exported_metrics"):
        """Initialize the TensorBoard CSV exporter."""
        self.tensorboard_url = tensorboard_url.rstrip('/')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Test connection
        if not self._test_connection():
            raise ConnectionError(f"Cannot connect to TensorBoard at {self.tensorboard_url}")
    
    def _test_connection(self):
        """Test if TensorBoard is accessible."""
        try:
            response = requests.get(self.tensorboard_url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_available_runs(self):
        """Get list of available runs from TensorBoard."""
        try:
            response = requests.get(f"{self.tensorboard_url}/data/runs")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get runs: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error getting runs: {e}")
            return []
    
    def get_available_scalars(self, run_name):
        """Get available scalar metrics for a specific run."""
        try:
            url = f"{self.tensorboard_url}/data/plugin/scalars/tags"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return data.get(run_name, {})
            else:
                print(f"Failed to get scalars: {response.status_code}")
                return {}
        except Exception as e:
            print(f"Error getting scalars: {e}")
            return {}
    
    def fetch_scalar_data(self, run_name, tag_name):
        """Fetch scalar data for a specific run and tag."""
        try:
            # URL encode the tag name to handle special characters
            import urllib.parse
            encoded_tag = urllib.parse.quote(tag_name)
            url = f"{self.tensorboard_url}/data/plugin/scalars/scalars"
            params = {
                'tag': tag_name,
                'run': run_name,
                'format': 'json'
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to fetch data for {tag_name}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching scalar data: {e}")
            return None
    
    def convert_to_dataframe(self, scalar_data):
        """Convert TensorBoard scalar data to pandas DataFrame."""
        if not scalar_data:
            return None
        
        # Extract data points
        data_points = []
        for point in scalar_data:
            # TensorBoard data format: [wall_time, step, value]
            if len(point) >= 3:
                wall_time, step, value = point[0], point[1], point[2]
                data_points.append({
                    'Wall time': wall_time,
                    'Step': step,
                    'Value': value
                })
        
        if data_points:
            return pd.DataFrame(data_points)
        return None
    
    def export_run_to_csv(self, run_name):
        """Export all metrics for a specific run to CSV files."""
        print(f"📊 Exporting metrics for run: {run_name}")
        
        # Get available scalars for this run
        scalars = self.get_available_scalars(run_name)
        
        if not scalars:
            print(f"   ⚠️  No scalar metrics found for run {run_name}")
            return
        
        exported_count = 0
        run_output_dir = self.output_dir / run_name
        run_output_dir.mkdir(exist_ok=True)
        
        for tag_name in scalars:
            print(f"   📈 Fetching {tag_name}...")
            
            # Fetch the data
            scalar_data = self.fetch_scalar_data(run_name, tag_name)
            
            if scalar_data:
                # Convert to DataFrame
                df = self.convert_to_dataframe(scalar_data)
                
                if df is not None and not df.empty:
                    # Create filename (replace special characters)
                    safe_tag_name = tag_name.replace('/', '_').replace('\\', '_').replace(':', '_')
                    csv_filename = f"{run_name}_{safe_tag_name}.csv"
                    csv_path = run_output_dir / csv_filename
                    
                    # Save to CSV
                    df.to_csv(csv_path, index=False)
                    print(f"   ✅ Saved {len(df)} data points to {csv_path}")
                    exported_count += 1
                else:
                    print(f"   ⚠️  No data points found for {tag_name}")
            else:
                print(f"   ❌ Failed to fetch data for {tag_name}")
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.1)
        
        print(f"   🎯 Exported {exported_count} metrics for {run_name}\n")
        return exported_count
    
    def export_all_metrics(self):
        """Export all available metrics to CSV files."""
        print("🚀 Starting TensorBoard metrics export...")
        print(f"📡 Connecting to TensorBoard at {self.tensorboard_url}")
        
        # Get available runs
        runs = self.get_available_runs()
        
        if not runs:
            print("❌ No runs found in TensorBoard")
            return
        
        print(f"📊 Found {len(runs)} run(s): {runs}")
        
        total_exported = 0
        for run_name in runs:
            exported = self.export_run_to_csv(run_name)
            if exported:
                total_exported += exported
        
        print(f"✅ Export complete! Total metrics exported: {total_exported}")
        print(f"📁 Files saved to: {self.output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(description='Export TensorBoard metrics to CSV')
    parser.add_argument('--url', default='http://localhost:6006',
                       help='TensorBoard URL (default: http://localhost:6006)')
    parser.add_argument('--output-dir', default='exported_metrics',
                       help='Output directory for CSV files (default: exported_metrics)')
    
    args = parser.parse_args()
    
    try:
        exporter = TensorBoardCSVExporter(args.url, args.output_dir)
        exporter.export_all_metrics()
    except ConnectionError as e:
        print(f"❌ {e}")
        print("Make sure TensorBoard is running at the specified URL")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
