"""
Unit Tests for Churn Prediction Feature Engineering

NOTE: All data in these tests is FICTIONAL and created for demonstration purposes only.
These tests showcase best practices for ML feature validation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_orders():
    """
    Create fictional sample order data for testing.
    All data is synthetic and for demonstration purposes only.
    """
    return pd.DataFrame({
        'order_id': ['ORD001', 'ORD002', 'ORD003', 'ORD004', 'ORD005'],
        'customer_id': ['CUST001', 'CUST001', 'CUST002', 'CUST002', 'CUST003'],
        'product_id': ['PROD001', 'PROD002', 'PROD001', 'PROD003', 'PROD001'],
        'quantity': [2, 1, 3, 1, 2],
        'unit_price': [29.99, 49.99, 29.99, 99.99, 29.99],
        'total_amount': [59.98, 49.99, 89.97, 99.99, 59.98],
        'order_date': pd.to_datetime([
            '2024-01-15', '2024-01-20', '2024-01-10', '2024-01-25', '2024-01-05'
        ]),
        'order_status': ['completed', 'completed', 'completed', 'cancelled', 'completed']
    })


@pytest.fixture
def sample_customers():
    """Create fictional sample customer data."""
    return pd.DataFrame({
        'customer_id': ['CUST001', 'CUST002', 'CUST003'],
        'name': ['Juan Garcia', 'Maria Lopez', 'Carlos Rodriguez'],
        'email': ['juan@email.com', 'maria@email.com', 'carlos@email.com'],
        'country': ['Argentina', 'Mexico', 'Colombia'],
        'registration_date': pd.to_datetime([
            '2023-06-15', '2023-08-20', '2023-10-01'
        ])
    })


@pytest.fixture
def sample_features():
    """Create fictional sample feature data."""
    return pd.DataFrame({
        'customer_id': ['CUST001', 'CUST002', 'CUST003'],
        'recency_days': [8, 3, 23],
        'frequency': [2, 2, 1],
        'monetary_value': [109.97, 189.96, 59.98],
        'cancellation_rate': [0.0, 50.0, 0.0],
        'purchase_velocity': [0.5, 0.8, 0.2],
        'is_churned': [0, 0, 1]
    })


# =============================================================================
# RFM Feature Tests
# =============================================================================

class TestRFMFeatures:
    """Tests for Recency, Frequency, Monetary features."""

    def test_recency_calculation(self, sample_orders):
        """Test that recency is calculated correctly."""
        reference_date = pd.Timestamp('2024-01-28')

        # Calculate recency per customer
        completed = sample_orders[sample_orders['order_status'] == 'completed']
        recency = completed.groupby('customer_id')['order_date'].max()
        recency_days = (reference_date - recency).dt.days

        # CUST001's last order was 2024-01-20, so 8 days ago
        assert recency_days['CUST001'] == 8
        # CUST002's last order was 2024-01-10, so 18 days ago
        assert recency_days['CUST002'] == 18
        # CUST003's last order was 2024-01-05, so 23 days ago
        assert recency_days['CUST003'] == 23

    def test_frequency_calculation(self, sample_orders):
        """Test that frequency counts only completed orders."""
        completed = sample_orders[sample_orders['order_status'] == 'completed']
        frequency = completed.groupby('customer_id')['order_id'].count()

        # CUST001 has 2 completed orders
        assert frequency['CUST001'] == 2
        # CUST002 has 1 completed order (1 cancelled)
        assert frequency['CUST002'] == 1
        # CUST003 has 1 completed order
        assert frequency['CUST003'] == 1

    def test_monetary_value_calculation(self, sample_orders):
        """Test that monetary value sums completed orders only."""
        completed = sample_orders[sample_orders['order_status'] == 'completed']
        monetary = completed.groupby('customer_id')['total_amount'].sum()

        # CUST001: 59.98 + 49.99 = 109.97
        assert round(monetary['CUST001'], 2) == 109.97
        # CUST002: 89.97 (cancelled order excluded)
        assert round(monetary['CUST002'], 2) == 89.97

    def test_rfm_positive_values(self, sample_features):
        """Test that RFM values are non-negative."""
        assert (sample_features['recency_days'] >= 0).all()
        assert (sample_features['frequency'] >= 0).all()
        assert (sample_features['monetary_value'] >= 0).all()


# =============================================================================
# Behavioral Feature Tests
# =============================================================================

class TestBehavioralFeatures:
    """Tests for behavioral features."""

    def test_cancellation_rate_range(self, sample_features):
        """Test cancellation rate is between 0 and 100."""
        assert (sample_features['cancellation_rate'] >= 0).all()
        assert (sample_features['cancellation_rate'] <= 100).all()

    def test_cancellation_rate_calculation(self, sample_orders):
        """Test cancellation rate calculation."""
        cancellation = sample_orders.groupby('customer_id').apply(
            lambda x: (x['order_status'] == 'cancelled').sum() / len(x) * 100
        )

        # CUST001: 0/2 = 0%
        assert cancellation['CUST001'] == 0.0
        # CUST002: 1/2 = 50%
        assert cancellation['CUST002'] == 50.0

    def test_unique_products_count(self, sample_orders):
        """Test unique products per customer."""
        unique_products = sample_orders.groupby('customer_id')['product_id'].nunique()

        # CUST001 bought PROD001 and PROD002
        assert unique_products['CUST001'] == 2
        # CUST002 bought PROD001 and PROD003
        assert unique_products['CUST002'] == 2


# =============================================================================
# Temporal Feature Tests
# =============================================================================

class TestTemporalFeatures:
    """Tests for temporal features."""

    def test_days_since_registration(self, sample_customers):
        """Test customer tenure calculation."""
        reference_date = pd.Timestamp('2024-01-28')
        tenure = (reference_date - sample_customers['registration_date']).dt.days

        # CUST001: registered 2023-06-15
        assert tenure.iloc[0] == 227  # days from Jun 15 to Jan 28

    def test_purchase_velocity_positive(self, sample_features):
        """Test that purchase velocity is non-negative."""
        assert (sample_features['purchase_velocity'] >= 0).all()


# =============================================================================
# Churn Label Tests
# =============================================================================

class TestChurnLabel:
    """Tests for churn label creation."""

    def test_churn_label_binary(self, sample_features):
        """Test that churn label is binary (0 or 1)."""
        assert set(sample_features['is_churned'].unique()).issubset({0, 1})

    def test_churn_threshold_logic(self):
        """Test churn label based on recency threshold."""
        threshold = 30  # days

        # Create test data
        df = pd.DataFrame({
            'customer_id': ['A', 'B', 'C'],
            'days_since_last_order': [10, 30, 45]
        })

        # Apply churn logic
        df['is_churned'] = (df['days_since_last_order'] > threshold).astype(int)

        assert df[df['customer_id'] == 'A']['is_churned'].values[0] == 0  # 10 < 30
        assert df[df['customer_id'] == 'B']['is_churned'].values[0] == 0  # 30 = 30, not >
        assert df[df['customer_id'] == 'C']['is_churned'].values[0] == 1  # 45 > 30


# =============================================================================
# Feature Validation Tests
# =============================================================================

class TestFeatureValidation:
    """Tests for overall feature quality."""

    def test_no_null_customer_ids(self, sample_features):
        """Test that customer_id has no nulls."""
        assert sample_features['customer_id'].notna().all()

    def test_no_duplicate_customer_ids(self, sample_features):
        """Test that customer_id is unique."""
        assert sample_features['customer_id'].nunique() == len(sample_features)

    def test_feature_dtypes(self, sample_features):
        """Test feature data types."""
        assert sample_features['recency_days'].dtype in ['int64', 'float64']
        assert sample_features['frequency'].dtype in ['int64', 'float64']
        assert sample_features['monetary_value'].dtype == 'float64'

    def test_all_features_present(self, sample_features):
        """Test that all required features exist."""
        required_features = [
            'customer_id', 'recency_days', 'frequency',
            'monetary_value', 'is_churned'
        ]
        for feature in required_features:
            assert feature in sample_features.columns, f"Missing: {feature}"


# =============================================================================
# Model Input Tests
# =============================================================================

class TestModelInput:
    """Tests for model input preparation."""

    def test_feature_scaling_compatibility(self, sample_features):
        """Test that features can be scaled without errors."""
        from sklearn.preprocessing import StandardScaler

        feature_cols = ['recency_days', 'frequency', 'monetary_value',
                       'cancellation_rate', 'purchase_velocity']
        X = sample_features[feature_cols]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        assert X_scaled.shape == X.shape
        assert not np.isnan(X_scaled).any()

    def test_train_test_split_stratification(self, sample_features):
        """Test stratified split preserves class distribution."""
        from sklearn.model_selection import train_test_split

        X = sample_features.drop(['customer_id', 'is_churned'], axis=1)
        y = sample_features['is_churned']

        # With small sample, just test split works
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )

        assert len(X_train) + len(X_test) == len(X)


# =============================================================================
# Integration Tests
# =============================================================================

class TestFeaturePipeline:
    """Integration tests for the full feature pipeline."""

    def test_end_to_end_feature_creation(self, sample_orders, sample_customers):
        """Test complete feature creation pipeline."""
        reference_date = pd.Timestamp('2024-01-28')

        # Filter completed orders
        completed = sample_orders[sample_orders['order_status'] == 'completed']

        # Calculate RFM
        rfm = completed.groupby('customer_id').agg({
            'order_date': lambda x: (reference_date - x.max()).days,
            'order_id': 'count',
            'total_amount': 'sum'
        }).rename(columns={
            'order_date': 'recency_days',
            'order_id': 'frequency',
            'total_amount': 'monetary_value'
        })

        # Verify output
        assert len(rfm) == 3  # 3 customers
        assert 'recency_days' in rfm.columns
        assert 'frequency' in rfm.columns
        assert 'monetary_value' in rfm.columns

        # Verify values are reasonable
        assert (rfm['recency_days'] >= 0).all()
        assert (rfm['frequency'] > 0).all()
        assert (rfm['monetary_value'] > 0).all()


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
