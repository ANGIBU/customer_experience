# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.feature_selector = None
        self.imputers = {}
        self.selected_features = None
        self.temporal_cutoff = None
        
    def safe_data_conversion(self, df):
        """데이터 변환"""
        df_clean = df.copy()
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col].fillna(0)
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], 0)
        
        return df_clean
        
    def apply_temporal_split(self, train_df):
        """시간 기반 분할"""
        print("시간 기반 분할")
        
        if 'temporal_id' not in train_df.columns:
            return train_df
        
        if self.temporal_cutoff is None:
            self.temporal_cutoff = 13224
        
        # 시간적 누수 방지
        safe_mask = train_df['temporal_id'] > self.temporal_cutoff
        safe_data = train_df[safe_mask].copy()
        
        removed_count = len(train_df) - len(safe_data)
        if removed_count > 0:
            print(f"시간적 누수 데이터 제거: {removed_count}개")
        
        return safe_data if len(safe_data) > 1000 else train_df
    
    def handle_missing_values(self, train_df, test_df):
        """결측치 처리"""
        print("결측치 처리")
        
        train_clean = self.safe_data_conversion(train_df)
        test_clean = self.safe_data_conversion(test_df)
        
        # 수치형 변수 - MICE 기법 사용
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        common_numeric = [col for col in numeric_cols if col in train_clean.columns and col in test_clean.columns]
        
        if common_numeric:
            # IterativeImputer 사용
            if 'numeric' not in self.imputers:
                self.imputers['numeric'] = IterativeImputer(
                    max_iter=10, 
                    random_state=42,
                    initial_strategy='median'
                )
                train_imputed = self.imputers['numeric'].fit_transform(train_clean[common_numeric])
            else:
                train_imputed = self.imputers['numeric'].transform(train_clean[common_numeric])
            
            test_imputed = self.imputers['numeric'].transform(test_clean[common_numeric])
            
            train_clean[common_numeric] = train_imputed
            test_clean[common_numeric] = test_imputed
        
        # 범주형 변수 - 최빈값
        categorical_cols = ['gender', 'subscription_type']
        
        for col in categorical_cols:
            if col in train_clean.columns and col in test_clean.columns:
                if col not in self.imputers:
                    mode_val = train_clean[col].mode()
                    self.imputers[col] = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
                
                train_clean[col].fillna(self.imputers[col], inplace=True)
                test_clean[col].fillna(self.imputers[col], inplace=True)
        
        return train_clean, test_clean
    
    def remove_outliers(self, train_df):
        """이상치 처리"""
        print("이상치 처리")
        
        train_clean = self.safe_data_conversion(train_df)
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_cols = [col for col in numeric_cols if col in train_clean.columns]
        
        for col in available_cols:
            values = train_clean[col].dropna()
            
            if len(values) > 100:
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower = Q1 - 2.0 * IQR
                    upper = Q3 + 2.0 * IQR
                    
                    train_clean[col] = np.clip(train_clean[col], lower, upper)
        
        return train_clean
    
    def select_features(self, train_df, k=50):
        """피처 선택"""
        print("피처 선택")
        
        if 'support_needs' not in train_df.columns:
            feature_cols = [col for col in train_df.columns if col != 'ID']
            return feature_cols[:min(k, len(feature_cols))]
        
        feature_cols = [col for col in train_df.columns if col not in ['ID', 'support_needs']]
        
        if len(feature_cols) <= k:
            self.selected_features = feature_cols
            return feature_cols
        
        X = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = np.clip(train_df['support_needs'], 0, 2)
        
        # 분산 필터링
        variance_selector = VarianceThreshold(threshold=0.005)
        X_variance = variance_selector.fit_transform(X)
        variance_features = [feature_cols[i] for i, selected in enumerate(variance_selector.get_support()) if selected]
        
        # 상호정보량 기반 선택
        if len(variance_features) > k:
            X_var_df = pd.DataFrame(X_variance, columns=variance_features)
            
            mi_selector = SelectKBest(score_func=mutual_info_classif, k=k)
            mi_selector.fit(X_var_df, y)
            
            selected_features = [variance_features[i] for i, selected in enumerate(mi_selector.get_support()) if selected]
        else:
            selected_features = variance_features
        
        self.selected_features = selected_features
        print(f"선택된 피처: {len(self.selected_features)}개")
        
        return self.selected_features
    
    def apply_scaling(self, train_df, test_df):
        """스케일링"""
        print("스케일링")
        
        train_clean = self.safe_data_conversion(train_df)
        test_clean = self.safe_data_conversion(test_df)
        
        numeric_cols = [col for col in train_clean.select_dtypes(include=[np.number]).columns 
                        if col not in ['ID', 'support_needs']]
        
        common_numeric = [col for col in numeric_cols if col in test_clean.columns]
        
        if common_numeric:
            train_numeric = train_clean[common_numeric].fillna(0)
            test_numeric = test_clean[common_numeric].fillna(0)
            
            if 'robust' not in self.scalers:
                self.scalers['robust'] = RobustScaler()
                train_scaled = self.scalers['robust'].fit_transform(train_numeric)
            else:
                train_scaled = self.scalers['robust'].transform(train_numeric)
            
            test_scaled = self.scalers['robust'].transform(test_numeric)
            
            train_clean[common_numeric] = train_scaled
            test_clean[common_numeric] = test_scaled
            
            print(f"스케일링 완료: {len(common_numeric)}개")
        
        return train_clean, test_clean
    
    def validate_data_quality(self, train_df, test_df):
        """데이터 품질 검증"""
        print("데이터 품질 검증")
        
        issues = []
        
        # 무한값 확인
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        
        train_inf = np.isinf(train_df[numeric_cols]).sum().sum()
        test_inf = np.isinf(test_df[numeric_cols]).sum().sum()
        
        if train_inf > 0:
            issues.append(f"훈련 무한값: {train_inf}")
        if test_inf > 0:
            issues.append(f"테스트 무한값: {test_inf}")
        
        # 결측치 확인
        train_nan = train_df[numeric_cols].isnull().sum().sum()
        test_nan = test_df[numeric_cols].isnull().sum().sum()
        
        if train_nan > 0:
            issues.append(f"훈련 결측치: {train_nan}")
        if test_nan > 0:
            issues.append(f"테스트 결측치: {test_nan}")
        
        # 타겟 유효성
        if 'support_needs' in train_df.columns:
            invalid_count = (~train_df['support_needs'].isin([0, 1, 2])).sum()
            if invalid_count > 0:
                issues.append(f"잘못된 타겟: {invalid_count}")
        
        if issues:
            print("품질 문제:")
            for issue in issues:
                print(f"- {issue}")
        else:
            print("품질 검증 통과")
        
        return len(issues) == 0, issues
    
    def process_data(self, train_df, test_df):
        """전처리 파이프라인"""
        print("데이터 전처리 시작")
        print("=" * 40)
        
        # 시간 분할
        train_df = self.apply_temporal_split(train_df)
        
        # 결측치 처리
        train_df, test_df = self.handle_missing_values(train_df, test_df)
        
        # 이상치 제거
        train_df = self.remove_outliers(train_df)
        
        # 피처 선택
        if 'support_needs' in train_df.columns:
            selected_features = self.select_features(train_df, k=50)
            
            keep_cols_train = ['ID'] + selected_features + ['support_needs']
            keep_cols_test = ['ID'] + [f for f in selected_features if f in test_df.columns]
            
            train_df = train_df[keep_cols_train]
            test_df = test_df[keep_cols_test]
        
        # 스케일링
        train_df, test_df = self.apply_scaling(train_df, test_df)
        
        # 품질 검증
        quality_ok, issues = self.validate_data_quality(train_df, test_df)
        
        print(f"전처리 완료: 훈련 {train_df.shape}, 테스트 {test_df.shape}")
        
        return train_df, test_df
    
    def prepare_data_temporal(self, train_df, test_df, val_size=0.25):
        """시간 기반 데이터 준비"""
        print("시간 기반 데이터 준비")
        
        # 공통 피처 식별
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        common_features = list((train_cols & test_cols) - {'ID', 'support_needs'})
        
        if not common_features:
            raise ValueError("공통 피처 없음")
        
        if 'support_needs' not in train_df.columns:
            raise ValueError("타겟 변수 없음")
        
        common_features = sorted(common_features)
        
        X = train_df[common_features].fillna(0).replace([np.inf, -np.inf], 0)
        y = np.clip(train_df['support_needs'], 0, 2)
        X_test = test_df[common_features].fillna(0).replace([np.inf, -np.inf], 0)
        test_ids = test_df['ID']
        
        # 시간 기반 분할
        if 'temporal_id' in train_df.columns:
            temporal_ids = train_df['temporal_id'].values
            sorted_indices = np.argsort(temporal_ids)
            
            # 후반부를 검증용으로 사용
            split_point = int(len(sorted_indices) * (1 - val_size))
            
            train_indices = sorted_indices[:split_point]
            val_indices = sorted_indices[split_point:]
            
            X_train = X.iloc[train_indices]
            X_val = X.iloc[val_indices]
            y_train = y.iloc[train_indices]
            y_val = y.iloc[val_indices]
            
            print("시간 기반 분할 적용")
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=42, stratify=y
            )
            print("계층화 분할 적용")
        
        print(f"훈련: {X_train.shape}, 검증: {X_val.shape}, 테스트: {X_test.shape}")
        
        return X_train, X_val, y_train, y_val, X_test, test_ids

def main():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    preprocessor = DataPreprocessor()
    train_processed, test_processed = preprocessor.process_data(train_df, test_df)
    
    X_train, X_val, y_train, y_val, X_test, test_ids = preprocessor.prepare_data_temporal(
        train_processed, test_processed
    )
    
    return preprocessor, X_train, X_val, y_train, y_val, X_test, test_ids

if __name__ == "__main__":
    main()