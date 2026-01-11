"""
End-to-End Test for Forecasting Workflow
Tests the complete flow: Upload -> Train -> Predict
"""
import requests
import time
import json
from pathlib import Path


class ForecastingE2ETest:
    """Test complete forecasting workflow."""
    
    def __init__(self, base_url="http://localhost:8081"):
        self.base_url = base_url
        self.token = None
        self.dataset_id = None
        self.job_id = None
        self.model_id = None
    
    def login(self):
        """Authenticate and get JWT token."""
        print("🔐 Step 1: Authenticating...")
        
        response = requests.post(
            f"{self.base_url}/api/v1/auth/login",
            json={
                "email": "admin@astralytiq.com",
                "password": "admin123"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            self.token = data['access_token']
            print(f"✅ Logged in as: {data['user']['name']}")
            return True
        else:
            print(f"❌ Login failed: {response.status_code}")
            print(response.text)
            return False
    
    def upload_data(self, file_path):
        """Upload CSV file for training."""
        print(f"\n📤 Step 2: Uploading data from {file_path}...")
        
        if not Path(file_path).exists():
            print(f"❌ File not found: {file_path}")
            return False
        
        with open(file_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/api/v1/ml/upload-data",
                headers={"Authorization": f"Bearer {self.token}"},
                files={"file": f}
            )
        
        if response.status_code == 200:
            data = response.json()
            self.dataset_id = data['dataset_id']
            print(f"✅ Upload successful!")
            print(f"   Dataset ID: {self.dataset_id}")
            print(f"   Rows: {data['rows']}, Columns: {data['columns']}")
            print(f"   Date Range: {data['date_range']['start']} to {data['date_range']['end']}")
            print(f"   Detected date columns: {data['preview']['detected_date_columns']}")
            print(f"   Detected numeric columns: {data['preview']['detected_numeric_columns']}")
            return True
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(response.text)
            return False
    
    def start_training(self, date_column, value_column, forecast_periods=30):
        """Start model training."""
        print(f"\n🚀 Step 3: Starting training...")
        
        response = requests.post(
            f"{self.base_url}/api/v1/ml/forecast/train",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            },
            json={
                "dataset_id": self.dataset_id,
                "date_column": date_column,
                "value_column": value_column,
                "model_type": "prophet",
                "forecast_periods": forecast_periods,
                "seasonality_mode": "additive",
                "include_holidays": False
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            self.job_id = data['job_id']
            print(f"✅ Training job started!")
            print(f"   Job ID: {self.job_id}")
            print(f"   Status: {data['status']}")
            return True
        else:
            print(f"❌ Training failed: {response.status_code}")
            print(response.text)
            return False
    
    def wait_for_training(self, max_wait=120, poll_interval=2):
        """Wait for training to complete."""
        print(f"\n⏳ Step 4: Waiting for training to complete...")
        
        start_time = time.time()
        last_progress = 0
        
        while time.time() - start_time < max_wait:
            response = requests.get(
                f"{self.base_url}/api/v1/ml/forecast/jobs/{self.job_id}",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            
            if response.status_code == 200:
                data = response.json()
                status = data['status']
                progress = data.get('progress', 0)
                
                # Show progress updates
                if progress != last_progress:
                    print(f"   Progress: {progress}% - Status: {status}")
                    last_progress = progress
                
                if status == "completed":
                    self.model_id = data['model_id']
                    metrics = data.get('metrics', {})
                    print(f"\n✅ Training completed!")
                    print(f"   Model ID: {self.model_id}")
                    print(f"   Metrics:")
                    for key, value in metrics.items():
                        if value is not None:
                            print(f"      {key}: {value:.4f}")
                    return True
                
                elif status == "failed":
                    error = data.get('error', 'Unknown error')
                    print(f"\n❌ Training failed: {error}")
                    return False
                
                time.sleep(poll_interval)
            else:
                print(f"❌ Failed to get job status: {response.status_code}")
                return False
        
        print(f"\n❌ Training timeout after {max_wait} seconds")
        return False
    
    def get_predictions(self, periods=30):
        """Get forecast predictions."""
        print(f"\n📊 Step 5: Getting predictions for {periods} periods...")
        
        response = requests.get(
            f"{self.base_url}/api/v1/ml/forecast/{self.model_id}",
            params={"periods": periods},
            headers={"Authorization": f"Bearer {self.token}"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Predictions retrieved!")
            print(f"   Forecast periods: {len(data['forecast_dates'])}")
            print(f"   First forecast date: {data['forecast_dates'][0]}")
            print(f"   Last forecast date: {data['forecast_dates'][-1]}")
            print(f"\n   Sample predictions (first 5):")
            for i in range(min(5, len(data['forecast_dates']))):
                date = data['forecast_dates'][i]
                value = data['forecast_values'][i]
                lower = data['lower_bound'][i]
                upper = data['upper_bound'][i]
                print(f"      {date}: {value:.2f} (CI: {lower:.2f} - {upper:.2f})")
            
            return data
        else:
            print(f"❌ Failed to get predictions: {response.status_code}")
            print(response.text)
            return None
    
    def run_full_test(self, csv_file, date_column, value_column):
        """Run complete end-to-end test."""
        print("=" * 60)
        print("🧪 FORECASTING E2E TEST")
        print("=" * 60)
        
        # Step 1: Login
        if not self.login():
            return False
        
        # Step 2: Upload data
        if not self.upload_data(csv_file):
            return False
        
        # Step 3: Start training
        if not self.start_training(date_column, value_column):
            return False
        
        # Step 4: Wait for training
        if not self.wait_for_training():
            return False
        
        # Step 5: Get predictions
        predictions = self.get_predictions()
        if predictions is None:
            return False
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nTest Summary:")
        print(f"  Dataset ID: {self.dataset_id}")
        print(f"  Model ID: {self.model_id}")
        print(f"  Forecast periods: {len(predictions['forecast_dates'])}")
        
        return True


if __name__ == "__main__":
    import sys
    
    # Test with sample sales data
    test = ForecastingE2ETest()
    
    # Path to sample data
    csv_file = "tests/fixtures/sample_sales.csv"
    
    if not Path(csv_file).exists():
        print(f"❌ Sample data not found: {csv_file}")
        print("   Run: python tests/fixtures/generate_sample_data.py")
        sys.exit(1)
    
    success = test.run_full_test(
        csv_file=csv_file,
        date_column="date",
        value_column="sales"
    )
    
    sys.exit(0 if success else 1)
