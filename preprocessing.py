# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.feature_selector = None
        self.imputers = {}
        self.selected_features = None
        self.variance_selector = None
        self.feature_order = None
        self.temporal_cutoff = None
        self.is_fitted = False
        
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
        
    def apply_temporal_split(self, train_df):
        """시간 기반 데이터 분할"""
        print("시간 기반 분할 적용")
        
        if 'temporal_id' not in train_df.columns:
            return train_df
        
        # 테스트 데이터와의 겹침 제거
        if self.temporal_cutoff is None:
            self.temporal_cutoff = 13224  # 테스트 데이터 최대 ID
        
        # 시간적 누수 방지
        safe_mask = train_df['temporal_id'] > self.temporal_cutoff
        safe_data = train_df[safe_mask].copy()
        
        removed_count = len(train_df) - len(safe_data)
        if removed_count > 0:
            print(f"시간적 누수 데이터 제거: {removed_count}개")
            print(f"안전한 훈련 데이터: {len(safe_data)}개")
        
        return safe_data
    
    def handle_missing_values(self, train_df, test_df):
        """결측치 처리"""
        print("결측치 처리")
        
        train_clean = self.safe_data_conversion(train_df)
        test_clean = self.safe_data_conversion(test_df)
        
        # 수치형 변수 결측치 처리
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        common_numeric = [col for col in numeric_cols if col in train_clean.columns and col in test_clean.columns]
        
        for col in common_numeric:
            try:
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy='median')
                    train_clean[col] = self.imputers[col].fit_transform(train_clean[[col]]).flatten()
                else:
                    train_clean[col] = self.imputers[col].transform(train_clean[[col]]).flatten()
                
                test_clean[col] = self.imputers[col].transform(test_clean[[col]]).flatten()
                
            except Exception as e:
                print(f"  {col} 결측치 처리 오류: {e}")
                train_clean[col] = train_clean[col].fillna(0)
                test_clean[col] = test_clean[col].fillna(0)
        
        # 범주형 변수 결측치 처리
        categorical_cols = ['gender', 'subscription_type']
        
        for col in categorical_cols:
            if col in train_clean.columns and col in test_clean.columns:
                try:
                    if col not in self.imputers:
                        mode_val = train_clean[col].mode()
                        if len(mode_val) > 0:
                            mode_val = mode_val.iloc[0]
                        else:
                            mode_val = 'Unknown'
                        self.imputers[col] = mode_val
                    
                    train_clean[col].fillna(self.imputers[col], inplace=True)
                    test_clean[col].fillna(self.imputers[col], inplace=True)
                    
                except Exception as e:
                    print(f"  {col} 범주형 결측치 처리 오류: {e}")
                    train_clean[col] = train_clean[col].fillna('Unknown')
                    test_clean[col] = test_clean[col].fillna('Unknown')
        
        return train_clean, test_clean
    
    def remove_outliers(self, train_df):
        """이상치 제거"""
        print("이상치 처리")
        
        train_clean = self.safe_data_conversion(train_df)
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_cols = [col for col in numeric_cols if col in train_clean.columns]
        
        for col in available_cols:
            try:
                values = train_clean[col].dropna()
                
                if len(values) > 100:
                    # IQR 방법으로 이상치 제거
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if IQR > 0:
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        
                        train_clean[col] = np.clip(train_clean[col], lower, upper)
                        
            except Exception as e:
                print(f"  {col} 이상치 처리 오류: {e}")
                continue
        
        return train_clean
    
    def select_features_optimized(self, train_df, k=60):
        """최적화된 피처 선택"""
        print("피처 선택")
        
        if 'support_needs' not in train_df.columns:
            feature_cols = [col for col in train_df.columns if col != 'ID']
            return feature_cols[:min(k, len(feature_cols))]
        
        feature_cols = [col for col in train_df.columns if col not in ['ID', 'support_needs']]
        
        if len(feature_cols) <= k:
            self.selected_features = feature_cols
            return feature_cols
        
        try:
            X = train_df[feature_cols]
            y = train_df['support_needs']
            
            X = X.fillna(0).replace([np.inf, -np.inf], 0)
            y_clean = np.clip(y, 0, 2)
            
            # 1단계: 분산 기반 필터링
            variance_selector = VarianceThreshold(threshold=0.01)
            X_variance = variance_selector.fit_transform(X)
            variance_features = [feature_cols[i] for i, selected in enumerate(variance_selector.get_support()) if selected]
            
            # 2단계: 상호정보량 기반 선택
            if len(variance_features) > k:
                X_var_df = pd.DataFrame(X_variance, columns=variance_features)
                
                mi_selector = SelectKBest(score_func=mutual_info_classif, k=k)
                mi_selector.fit(X_var_df, y_clean)
                
                selected_features = [variance_features[i] for i, selected in enumerate(mi_selector.get_support()) if selected]
            else:
                selected_features = variance_features
            
            self.selected_features = selected_features
            
            print(f"선택된 피처: {len(self.selected_features)}개")
            
            return self.selected_features
            
        except Exception as e:
            print(f"피처 선택 오류: {e}")
            fallback_features = feature_cols[:min(k, len(feature_cols))]
            self.selected_features = fallback_features
            return fallback_features
    
    def apply_scaling(self, train_df, test_df):
        """스케일링 적용"""
        print("스케일링 적용")
        
        train_clean = self.safe_data_conversion(train_df)
        test_clean = self.safe_data_conversion(test_df)
        
        # 수치형 컬럼 식별
        numeric_cols = [col for col in train_clean.select_dtypes(include=[np.number]).columns 
                        if col not in ['ID', 'support_needs']]
        
        common_numeric = [col for col in numeric_cols if col in test_clean.columns]
        
        if not common_numeric:
            return train_clean, test_clean
        
        try:
            train_numeric = train_clean[common_numeric].fillna(0)
            test_numeric = test_clean[common_numeric].fillna(0)
            
            # StandardScaler 적용
            if 'standard' not in self.scalers:
                self.scalers['standard'] = StandardScaler()
                train_scaled = self.scalers['standard'].fit_transform(train_numeric)
            else:
                train_scaled = self.scalers['standard'].transform(train_numeric)
            
            test_scaled = self.scalers['standard'].transform(test_numeric)
            
            train_clean[common_numeric] = train_scaled
            test_clean[common_numeric] = test_scaled
            
            print(f"스케일링 완료: {len(common_numeric)}개 피처")
            
        except Exception as e:
            print(f"스케일링 오류: {e}")
            
        return train_clean, test_clean
    
    def validate_data_quality(self, train_df, test_df):
        """데이터 품질 검증"""
        print("데이터 품질 검증")
        
        issues = []
        
        try:
            # 무한값 확인
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            
            train_inf = np.isinf(train_df[numeric_cols]).sum().sum()
            test_inf = np.isinf(test_df[numeric_cols]).sum().sum()
            
            if train_inf > 0 or test_inf > 0:
                issues.append(f"무한값 - 훈련: {train_inf}, 테스트: {test_inf}")
            
            # 결측치 확인
            train_nan = train_df[numeric_cols].isnull().sum().sum()
            test_nan = test_df[numeric_cols].isnull().sum().sum()
            
            if train_nan > 0 or test_nan > 0:
                issues.append(f"결측치 - 훈련: {train_nan}, 테스트: {test_nan}")
                
            # 타겟 변수 유효성
            if 'support_needs' in train_df.columns:
                invalid_targets = ((train_df['support_needs'] < 0) | (train_df['support_needs'] > 2)).sum()
                if invalid_targets > 0:
                    issues.append(f"잘못된 타겟값: {invalid_targets}개")
            
            if issues:
                print("품질 문제:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("품질 검증 통과")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            print(f"품질 검증 오류: {e}")
            return False, [f"검증 오류: {e}"]
    
    def process_data(self, train_df, test_df):
        """전체 전처리 파이프라인"""
        print("데이터 전처리 시작")
        print("=" * 40)
        
        try:
            # 시간적 분할 적용
            train_df = self.apply_temporal_split(train_df)
            
            # 결측치 처리
            train_df, test_df = self.handle_missing_values(train_df, test_df)
            
            # 이상치 제거
            train_df = self.remove_outliers(train_df)
            
            # 피처 선택
            if 'support_needs' in train_df.columns:
                selected_features = self.select_features_optimized(train_df, k=60)
                
                # 선택된 피처만 유지
                keep_cols_train = ['ID'] + selected_features
                if 'support_needs' in train_df.columns:
                    keep_cols_train.append('support_needs')
                
                keep_cols_test = ['ID'] + [f for f in selected_features if f in test_df.columns]
                
                train_df = train_df[keep_cols_train]
                test_df = test_df[keep_cols_test]
                
                print(f"선택된 피처 수: {len(selected_features)}")
            
            # 스케일링 적용
            train_df, test_df = self.apply_scaling(train_df, test_df)
            
            # 품질 검증
            quality_ok, issues = self.validate_data_quality(train_df, test_df)
            
            print(f"\n전처리 완료:")
            print(f"  훈련: {train_df.shape}")
            print(f"  테스트: {test_df.shape}")
            print(f"  품질: {'통과' if quality_ok else '문제있음'}")
            
            return train_df, test_df
            
        except Exception as e:
            print(f"전처리 파이프라인 오류: {e}")
            return train_df, test_df
    
    def prepare_data_temporal(self, train_df, test_df, val_size=0.2):
        """시간 기반 데이터 준비"""
        print("시간 기반 데이터 준비")
        
        try:
            # 공통 피처 식별
            train_cols = set(train_df.columns)
            test_cols = set(test_df.columns)
            common_features = list((train_cols & test_cols) - {'ID', 'support_needs'})
            
            if not common_features:
                raise ValueError("공통 피처가 없습니다")
            
            if 'support_needs' not in train_df.columns:
                raise ValueError("훈련 데이터에 support_needs 컬럼이 없습니다")
            
            common_features = sorted(common_features)
            
            X = train_df[common_features]
            y = train_df['support_needs']
            X_test = test_df[common_features]
            test_ids = test_df['ID']
            
            # 데이터 정리
            X = X.fillna(0).replace([np.inf, -np.inf], 0)
            X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)
            y = np.clip(y, 0, 2)
            
            # 시간 기반 분할
            if 'temporal_id' in train_df.columns:
                # 시간 순서대로 정렬
                temporal_ids = train_df['temporal_id'].values
                sorted_indices = np.argsort(temporal_ids)
                
                # 뒤쪽 데이터를 검증용으로 사용
                split_point = int(len(sorted_indices) * (1 - val_size))
                
                train_indices = sorted_indices[:split_point]
                val_indices = sorted_indices[split_point:]
                
                X_train = X.iloc[train_indices]
                X_val = X.iloc[val_indices]
                y_train = y.iloc[train_indices]
                y_val = y.iloc[val_indices]
                
                print("시간 기반 분할 적용됨")
            else:
                # 일반 분할 (계층화)
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=val_size, random_state=42, stratify=y
                )
                
                print("계층화 분할 적용됨")
            
            print(f"훈련: {X_train.shape}")
            print(f"검증: {X_val.shape}")
            print(f"테스트: {X_test.shape}")
            print(f"공통 피처: {len(common_features)}개")
            
            return X_train, X_val, y_train, y_val, X_test, test_ids
            
        except Exception as e:
            print(f"데이터 준비 오류: {e}")
            raise e

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