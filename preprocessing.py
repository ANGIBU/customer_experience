# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, QuantileTransformer, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.feature_selector = None
        self.imputers = {}
        self.selected_features = None
        self.temporal_cutoff = None
        self.feature_stats = {}
        
    def safe_data_conversion(self, df):
        """안전한 데이터 변환"""
        df_clean = df.copy()
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col].fillna(0)
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], 0)
        
        return df_clean
        
    def apply_temporal_filtering(self, train_df, temporal_info=None):
        """엄격한 시간적 필터링"""
        if temporal_info is None or 'temporal_threshold' not in temporal_info:
            return train_df
        
        threshold = temporal_info['temporal_threshold']
        
        if 'temporal_id' not in train_df.columns:
            return train_df
        
        # 매우 엄격한 필터링
        safe_mask = train_df['temporal_id'] <= threshold
        safe_data = train_df[safe_mask].copy()
        
        # 최소 데이터 보장
        if len(safe_data) < len(train_df) * 0.3:  # 30% 미만이면 50% 지점으로 조정
            percentile_50 = np.percentile(train_df['temporal_id'], 50)
            safe_mask = train_df['temporal_id'] <= percentile_50
            safe_data = train_df[safe_mask].copy()
        
        print(f"시간적 필터링: {len(train_df)} → {len(safe_data)} ({len(safe_data)/len(train_df):.1%})")
        
        return safe_data if len(safe_data) > 5000 else train_df
    
    def detect_outliers_simple(self, train_df):
        """단순한 이상치 탐지"""
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_cols = [col for col in numeric_cols if col in train_df.columns]
        
        if len(available_cols) < 3:
            return train_df
        
        X_numeric = train_df[available_cols].fillna(0)
        
        # Isolation Forest (간소화)
        try:
            iso_forest = IsolationForest(
                contamination=0.02,  # 2% 이상치
                random_state=42,
                n_jobs=1
            )
            outlier_mask = iso_forest.fit_predict(X_numeric) == -1
            
            # 이상치 제거 (최대 3% 제거)
            outlier_ratio = outlier_mask.sum() / len(train_df)
            if outlier_ratio > 0.03:
                # 가장 극단적인 3%만 제거
                outlier_scores = iso_forest.decision_function(X_numeric)
                threshold = np.percentile(outlier_scores, 3)
                outlier_mask = outlier_scores <= threshold
            
            clean_data = train_df[~outlier_mask].copy()
            print(f"이상치 제거: {len(train_df)} → {len(clean_data)} ({outlier_mask.sum()}개 제거)")
            
            return clean_data if len(clean_data) > len(train_df) * 0.9 else train_df
            
        except Exception:
            return train_df
    
    def handle_missing_values_simple(self, train_df, test_df):
        """단순한 결측치 처리"""
        train_clean = self.safe_data_conversion(train_df)
        test_clean = self.safe_data_conversion(test_df)
        
        # 수치형 변수 - 중앙값으로 대체
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        common_numeric = [col for col in numeric_cols if col in train_clean.columns and col in test_clean.columns]
        
        for col in common_numeric:
            if col not in self.imputers:
                self.imputers[col] = train_clean[col].median()
            
            train_clean[col].fillna(self.imputers[col], inplace=True)
            test_clean[col].fillna(self.imputers[col], inplace=True)
        
        # 범주형 변수 - 최빈값으로 대체
        categorical_cols = ['gender', 'subscription_type']
        
        for col in categorical_cols:
            if col in train_clean.columns and col in test_clean.columns:
                if col not in self.imputers:
                    mode_val = train_clean[col].mode()
                    self.imputers[col] = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
                
                train_clean[col].fillna(self.imputers[col], inplace=True)
                test_clean[col].fillna(self.imputers[col], inplace=True)
        
        # 생성된 피처들도 처리
        for col in train_clean.columns:
            if col not in ['ID', 'support_needs'] and col not in common_numeric + categorical_cols:
                if train_clean[col].dtype in ['float64', 'int64']:
                    median_val = train_clean[col].median()
                    train_clean[col].fillna(median_val, inplace=True)
                    if col in test_clean.columns:
                        test_clean[col].fillna(median_val, inplace=True)
        
        return train_clean, test_clean
    
    def select_features_simple(self, train_df, max_features=50):  # 75 → 50으로 축소
        """단순한 피처 선택"""
        if 'support_needs' not in train_df.columns:
            feature_cols = [col for col in train_df.columns if col not in ['ID']]
            return feature_cols[:min(max_features, len(feature_cols))]
        
        feature_cols = [col for col in train_df.columns if col not in ['ID', 'support_needs']]
        
        if len(feature_cols) <= max_features:
            self.selected_features = feature_cols
            return feature_cols
        
        X = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = np.clip(train_df['support_needs'], 0, 2)
        
        # 분산 필터링
        variance_selector = VarianceThreshold(threshold=0.01)  # 더 엄격한 기준
        X_variance = variance_selector.fit_transform(X)
        variance_features = [feature_cols[i] for i, selected in enumerate(variance_selector.get_support()) if selected]
        
        if len(variance_features) <= max_features:
            self.selected_features = variance_features
            return variance_features
        
        # 상호정보량 기반 선택
        X_var_df = pd.DataFrame(X_variance, columns=variance_features)
        mi_scores = mutual_info_classif(X_var_df, y, random_state=42)
        
        feature_scores = list(zip(variance_features, mi_scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_features = [f[0] for f in feature_scores[:max_features]]
        
        self.selected_features = selected_features
        return selected_features
    
    def apply_scaling_simple(self, train_df, test_df):
        """단순한 스케일링"""
        train_clean = self.safe_data_conversion(train_df)
        test_clean = self.safe_data_conversion(test_df)
        
        numeric_cols = [col for col in train_clean.select_dtypes(include=[np.number]).columns 
                        if col not in ['ID', 'support_needs']]
        
        common_numeric = [col for col in numeric_cols if col in test_clean.columns]
        
        if not common_numeric:
            return train_clean, test_clean
        
        # RobustScaler 사용 (간단하고 안정적)
        if 'robust' not in self.scalers:
            self.scalers['robust'] = RobustScaler()
            train_scaled = self.scalers['robust'].fit_transform(train_clean[common_numeric])
        else:
            train_scaled = self.scalers['robust'].transform(train_clean[common_numeric])
        
        test_scaled = self.scalers['robust'].transform(test_clean[common_numeric])
        
        train_clean[common_numeric] = train_scaled
        test_clean[common_numeric] = test_scaled
        
        return train_clean, test_clean
    
    def apply_dimensionality_reduction_simple(self, train_df, test_df):
        """단순한 차원 축소"""
        if 'support_needs' not in train_df.columns:
            return train_df, test_df
        
        feature_cols = [col for col in train_df.columns if col not in ['ID', 'support_needs']]
        
        if len(feature_cols) < 20:  # 피처가 적으면 스킵
            return train_df, test_df
        
        X_train = train_df[feature_cols].fillna(0)
        X_test = test_df[feature_cols].fillna(0)
        
        train_enhanced = train_df.copy()
        test_enhanced = test_df.copy()
        
        # PCA만 사용 (간소화)
        try:
            n_components = min(10, len(feature_cols) // 5)  # 더 적은 컴포넌트
            
            if 'pca' not in self.scalers:
                self.scalers['pca'] = PCA(n_components=n_components, random_state=42)
                pca_train = self.scalers['pca'].fit_transform(X_train)
            else:
                pca_train = self.scalers['pca'].transform(X_train)
            
            pca_test = self.scalers['pca'].transform(X_test)
            
            # PCA 컴포넌트 추가
            for i in range(pca_train.shape[1]):
                train_enhanced[f'pca_{i}'] = pca_train[:, i]
                test_enhanced[f'pca_{i}'] = pca_test[:, i]
                
        except Exception:
            pass
        
        return train_enhanced, test_enhanced
    
    def validate_data_quality(self, train_df, test_df):
        """데이터 품질 검증"""
        issues = []
        
        # 훈련 데이터 수치형 컬럼
        train_numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        test_numeric_cols = test_df.select_dtypes(include=[np.number]).columns
        
        # 무한값 확인
        train_inf = np.isinf(train_df[train_numeric_cols]).sum().sum()
        test_inf = np.isinf(test_df[test_numeric_cols]).sum().sum()
        
        if train_inf > 0:
            issues.append(f"train_infinite_values: {train_inf}")
        if test_inf > 0:
            issues.append(f"test_infinite_values: {test_inf}")
        
        # 결측치 확인
        train_nan = train_df[train_numeric_cols].isnull().sum().sum()
        test_nan = test_df[test_numeric_cols].isnull().sum().sum()
        
        if train_nan > 0:
            issues.append(f"train_missing_values: {train_nan}")
        if test_nan > 0:
            issues.append(f"test_missing_values: {test_nan}")
        
        return len(issues) == 0, issues
    
    def process_data(self, train_df, test_df, temporal_info=None):
        """단순화된 전처리 파이프라인"""
        if train_df is None or test_df is None or train_df.empty or test_df.empty:
            return None, None
        
        print("전처리 시작:")
        
        # 시간적 필터링
        if temporal_info is not None:
            train_df = self.apply_temporal_filtering(train_df, temporal_info)
        
        # 이상치 탐지 및 제거 (간소화)
        train_df = self.detect_outliers_simple(train_df)
        
        # 결측치 처리 (간소화)
        train_df, test_df = self.handle_missing_values_simple(train_df, test_df)
        
        # 피처 선택 (축소)
        if 'support_needs' in train_df.columns:
            selected_features = self.select_features_simple(train_df, max_features=50)
            
            keep_cols_train = ['ID', 'support_needs'] + selected_features
            keep_cols_test = ['ID'] + [f for f in selected_features if f in test_df.columns]
            
            # 존재하는 컬럼만 선택
            available_train_cols = [col for col in keep_cols_train if col in train_df.columns]
            available_test_cols = [col for col in keep_cols_test if col in test_df.columns]
            
            train_df = train_df[available_train_cols]
            test_df = test_df[available_test_cols]
        
        # 스케일링 (간소화)
        train_df, test_df = self.apply_scaling_simple(train_df, test_df)
        
        # 차원 축소 (간소화) - 피처가 많을 때만
        if len(train_df.columns) > 30:
            train_df, test_df = self.apply_dimensionality_reduction_simple(train_df, test_df)
        
        # 품질 검증
        quality_ok, issues = self.validate_data_quality(train_df, test_df)
        if not quality_ok:
            print(f"데이터 품질 문제: {len(issues)}개")
        
        print("✓ 전처리 완료")
        
        return train_df, test_df
    
    def prepare_data_temporal_optimized(self, train_df, test_df, val_size=0.2, gap_size=0.01):  # 검증 크기 증가
        """최적화된 데이터 준비"""
        if train_df is None or test_df is None:
            return None, None, None, None, None, None
            
        if 'support_needs' not in train_df.columns:
            # 타겟이 없는 경우 기본 분할
            feature_cols = [col for col in train_df.columns if col not in ['ID']]
            common_features = [col for col in feature_cols if col in test_df.columns]
            
            if not common_features:
                raise ValueError("공통 피처 없음")
                
            X_train = train_df[common_features].fillna(0).replace([np.inf, -np.inf], 0)
            X_test = test_df[common_features].fillna(0).replace([np.inf, -np.inf], 0)
            test_ids = test_df['ID'] if 'ID' in test_df.columns else pd.Series(range(len(test_df)))
            
            # 더미 분할
            X_train_split, X_val_split, y_dummy, y_val_dummy = train_test_split(
                X_train, np.zeros(len(X_train)), test_size=val_size, random_state=42
            )
            
            return X_train_split, X_val_split, y_dummy[:len(X_train_split)], y_val_dummy, X_test, test_ids
        
        # 공통 피처 식별
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        common_features = list((train_cols & test_cols) - {'ID', 'support_needs'})
        
        if not common_features:
            raise ValueError("공통 피처 없음")
        
        common_features = sorted(common_features)
        
        X = train_df[common_features].fillna(0).replace([np.inf, -np.inf], 0)
        y = np.clip(train_df['support_needs'], 0, 2)
        X_test = test_df[common_features].fillna(0).replace([np.inf, -np.inf], 0)
        test_ids = test_df['ID'] if 'ID' in test_df.columns else pd.Series(range(len(test_df)))
        
        # 시간 기반 분할 (단순화)
        if 'temporal_id' in X.columns and len(X) > 2000:
            temporal_ids = X['temporal_id'].values
            sorted_indices = np.argsort(temporal_ids)
            
            total_samples = len(sorted_indices)
            gap_samples = int(total_samples * gap_size)
            val_samples = int(total_samples * val_size)
            train_samples = total_samples - val_samples - gap_samples
            
            if train_samples > 1000 and val_samples > 500:
                # 시간 순서 분할
                train_indices = sorted_indices[:train_samples]
                val_indices = sorted_indices[train_samples + gap_samples:]
                
                X_train = X.iloc[train_indices]
                X_val = X.iloc[val_indices]
                y_train = y.iloc[train_indices]
                y_val = y.iloc[val_indices]
            else:
                # 계층화 분할
                sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
                train_idx, val_idx = next(sss.split(X, y))
                
                X_train = X.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_train = y.iloc[train_idx]
                y_val = y.iloc[val_idx]
        else:
            # 계층화 분할
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
            train_idx, val_idx = next(sss.split(X, y))
            
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]
        
        return X_train, X_val, y_train, y_val, X_test, test_ids

def main():
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        preprocessor = DataPreprocessor()
        train_processed, test_processed = preprocessor.process_data(train_df, test_df)
        
        if train_processed is not None and test_processed is not None:
            X_train, X_val, y_train, y_val, X_test, test_ids = preprocessor.prepare_data_temporal_optimized(
                train_processed, test_processed
            )
            
            return preprocessor, X_train, X_val, y_train, y_val, X_test, test_ids
        else:
            return preprocessor, None, None, None, None, None, None
            
    except Exception as e:
        print(f"전처리 오류: {e}")
        return None, None, None, None, None, None, None

if __name__ == "__main__":
    main()