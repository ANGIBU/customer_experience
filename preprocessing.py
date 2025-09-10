# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, QuantileTransformer, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.feature_selector = None
        self.imputers = {}
        self.selected_features = None
        self.temporal_cutoff = None
        self.pca_transformer = None
        
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
        """시간적 필터링 - 개선"""
        if temporal_info is None or 'temporal_threshold' not in temporal_info:
            return train_df
        
        threshold = temporal_info['temporal_threshold']
        
        if 'temporal_id' not in train_df.columns:
            return train_df
        
        # 더 관대한 필터링 (최소 보존율 94%)
        safe_mask = train_df['temporal_id'] <= threshold
        safe_data = train_df[safe_mask].copy()
        
        removal_ratio = 1 - (len(safe_data) / len(train_df))
        if removal_ratio > 0.06:  # 0.08 -> 0.06 (6% 이상 제거 방지)
            percentile_96 = np.percentile(train_df['temporal_id'], 96)  # 95 -> 96
            safe_mask = train_df['temporal_id'] <= percentile_96
            safe_data = train_df[safe_mask].copy()
        
        return safe_data if len(safe_data) > 10000 else train_df  # 8000 -> 10000
    
    def handle_missing_values(self, train_df, test_df):
        """결측치 처리 - 개선"""
        train_clean = self.safe_data_conversion(train_df)
        test_clean = self.safe_data_conversion(test_df)
        
        # 수치형 변수 - 적응적 대체
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        # 시간 관련 피처가 있다면 추가
        if 'temporal_position' in train_clean.columns:
            numeric_cols.append('temporal_position')
        
        # after_interaction 관련 피처 처리
        for col in train_clean.columns:
            if 'after_interaction' in col:
                numeric_cols.append(col)
        
        common_numeric = [col for col in numeric_cols if col in train_clean.columns and col in test_clean.columns]
        
        if common_numeric:
            # 결측치 비율 확인
            missing_ratios = {}
            for col in common_numeric:
                missing_ratio = train_clean[col].isnull().mean()
                missing_ratios[col] = missing_ratio
            
            # 적응적 대체 전략 (개선)
            high_missing_cols = [col for col, ratio in missing_ratios.items() if ratio > 0.3]  # 0.25 -> 0.3
            medium_missing_cols = [col for col, ratio in missing_ratios.items() if 0.05 < ratio <= 0.3]
            low_missing_cols = [col for col, ratio in missing_ratios.items() if ratio <= 0.05]
            
            # KNN Imputer (높은 결측치) - 개선
            if high_missing_cols:
                if 'knn' not in self.imputers:
                    self.imputers['knn'] = KNNImputer(n_neighbors=7, weights='distance')  # 5 -> 7
                    train_imputed_knn = self.imputers['knn'].fit_transform(train_clean[high_missing_cols])
                else:
                    train_imputed_knn = self.imputers['knn'].transform(train_clean[high_missing_cols])
                
                test_imputed_knn = self.imputers['knn'].transform(test_clean[high_missing_cols])
                
                train_clean[high_missing_cols] = train_imputed_knn
                test_clean[high_missing_cols] = test_imputed_knn
            
            # Iterative Imputer (중간 결측치) - 개선
            if medium_missing_cols:
                if 'iterative' not in self.imputers:
                    self.imputers['iterative'] = IterativeImputer(
                        max_iter=10,  # 8 -> 10
                        random_state=42,
                        initial_strategy='median',
                        min_value=0  # 추가
                    )
                    train_imputed_iter = self.imputers['iterative'].fit_transform(train_clean[medium_missing_cols])
                else:
                    train_imputed_iter = self.imputers['iterative'].transform(train_clean[medium_missing_cols])
                
                test_imputed_iter = self.imputers['iterative'].transform(test_clean[medium_missing_cols])
                
                train_clean[medium_missing_cols] = train_imputed_iter
                test_clean[medium_missing_cols] = test_imputed_iter
            
            # 중앙값 대체 (낮은 결측치)
            for col in low_missing_cols:
                if col not in self.imputers:
                    self.imputers[col] = train_clean[col].median()
                
                train_clean[col].fillna(self.imputers[col], inplace=True)
                test_clean[col].fillna(self.imputers[col], inplace=True)
        
        # 범주형 변수 - 타겟 기반 대체 (개선)
        categorical_cols = ['gender', 'subscription_type']
        
        for col in categorical_cols:
            if col in train_clean.columns and col in test_clean.columns:
                if col not in self.imputers:
                    # 타겟별 최빈값 계산
                    if 'support_needs' in train_clean.columns:
                        target_modes = {}
                        for target in [0, 1, 2]:
                            subset = train_clean[train_clean['support_needs'] == target]
                            if len(subset) > 0 and not subset[col].mode().empty:
                                target_modes[target] = subset[col].mode().iloc[0]
                        
                        # 클래스별 가중 최빈값
                        weighted_mode = None
                        if target_modes:
                            # 클래스 1에 더 높은 가중치
                            class_weights = {0: 0.33, 1: 0.4, 2: 0.27}
                            mode_counts = {}
                            for target, mode in target_modes.items():
                                if mode not in mode_counts:
                                    mode_counts[mode] = 0
                                mode_counts[mode] += class_weights[target]
                            weighted_mode = max(mode_counts, key=mode_counts.get)
                        
                        # 전체 최빈값을 기본값으로
                        overall_mode = train_clean[col].mode()
                        default_mode = weighted_mode if weighted_mode else (overall_mode.iloc[0] if len(overall_mode) > 0 else 'Unknown')
                        
                        self.imputers[col] = {'target_modes': target_modes, 'default': default_mode}
                    else:
                        mode_val = train_clean[col].mode()
                        self.imputers[col] = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
                
                # 적용
                if isinstance(self.imputers[col], dict):
                    # 타겟 기반 대체
                    for idx in train_clean[train_clean[col].isnull()].index:
                        if 'support_needs' in train_clean.columns:
                            target = train_clean.loc[idx, 'support_needs']
                            if target in self.imputers[col]['target_modes']:
                                train_clean.loc[idx, col] = self.imputers[col]['target_modes'][target]
                            else:
                                train_clean.loc[idx, col] = self.imputers[col]['default']
                        else:
                            train_clean.loc[idx, col] = self.imputers[col]['default']
                    
                    # 테스트 데이터는 기본값 사용
                    test_clean[col].fillna(self.imputers[col]['default'], inplace=True)
                else:
                    # 단순 대체
                    train_clean[col].fillna(self.imputers[col], inplace=True)
                    test_clean[col].fillna(self.imputers[col], inplace=True)
        
        return train_clean, test_clean
    
    def remove_outliers_robust(self, train_df):
        """이상치 제거 - 개선"""
        train_clean = self.safe_data_conversion(train_df)
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        # 생성된 피처들도 포함
        for col in train_clean.columns:
            if any(keyword in col for keyword in ['_log', '_sqrt', '_rank', 'customer_value', 'payment_stability', 'usage_intensity']):
                if col not in numeric_cols:
                    numeric_cols.append(col)
        
        available_cols = [col for col in numeric_cols if col in train_clean.columns]
        
        for col in available_cols:
            values = train_clean[col].dropna()
            
            if len(values) > 100:
                # 더 관대한 이상치 제거 (0.3% ~ 99.7%)
                lower_bound = values.quantile(0.003)  # 0.005 -> 0.003
                upper_bound = values.quantile(0.997)  # 0.995 -> 0.997
                
                # IQR 방법도 병행 사용
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                iqr_lower = Q1 - 2.5 * IQR  # 3 -> 2.5
                iqr_upper = Q3 + 2.5 * IQR
                
                # 두 방법 중 더 관대한 것 선택
                final_lower = min(lower_bound, iqr_lower)
                final_upper = max(upper_bound, iqr_upper)
                
                train_clean[col] = np.clip(train_clean[col], final_lower, final_upper)
        
        return train_clean
    
    def select_features_optimized(self, train_df, max_features=75):  # 70 -> 75
        """피처 선택 - 개선"""
        if 'support_needs' not in train_df.columns:
            feature_cols = [col for col in train_df.columns if col not in ['ID']]
            return feature_cols[:min(max_features, len(feature_cols))]
        
        feature_cols = [col for col in train_df.columns if col not in ['ID', 'support_needs']]
        
        if len(feature_cols) <= max_features:
            self.selected_features = feature_cols
            return feature_cols
        
        X = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = np.clip(train_df['support_needs'], 0, 2)
        
        # 1단계: 분산 필터링 (더 관대한 기준)
        variance_selector = VarianceThreshold(threshold=0.00005)  # 0.0001 -> 0.00005
        X_variance = variance_selector.fit_transform(X)
        variance_features = [feature_cols[i] for i, selected in enumerate(variance_selector.get_support()) if selected]
        
        if len(variance_features) <= max_features:
            self.selected_features = variance_features
            return variance_features
        
        # 2단계: 상관관계 기반 중복 제거 (더 관대한 기준)
        X_var_df = pd.DataFrame(X_variance, columns=variance_features)
        correlation_matrix = X_var_df.corr().abs()
        
        # 높은 상관관계 (>0.98) 피처 제거
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.98)]  # 0.97 -> 0.98
        reduced_features = [f for f in variance_features if f not in high_corr_features]
        
        if len(reduced_features) <= max_features:
            self.selected_features = reduced_features
            return reduced_features
        
        # 3단계: 상호정보량 기반 선택 (개선)
        X_reduced = train_df[reduced_features].fillna(0)
        
        # 클래스별 가중치를 고려한 상호정보량
        class_weights = {0: 0.98, 1: 1.15, 2: 1.03}  # 모델과 동일한 가중치
        sample_weight = np.ones(len(y))
        for cls, weight in class_weights.items():
            mask = y == cls
            sample_weight[mask] = weight
        
        mi_scores = mutual_info_classif(X_reduced, y, random_state=42)
        
        # 클래스 1 예측에 중요한 피처 우선 선택
        feature_scores = list(zip(reduced_features, mi_scores))
        
        # 피처별 클래스 1 중요도 계산
        class_1_mask = y == 1
        if class_1_mask.sum() > 100:
            class_1_mi_scores = mutual_info_classif(X_reduced, class_1_mask.astype(int), random_state=42)
            # 종합 점수 (전체 MI와 클래스 1 MI의 가중 평균)
            combined_scores = 0.7 * mi_scores + 0.3 * class_1_mi_scores
            feature_scores = list(zip(reduced_features, combined_scores))
        
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_features = [f[0] for f in feature_scores[:max_features]]
        
        self.selected_features = selected_features
        return selected_features
    
    def apply_scaling(self, train_df, test_df):
        """스케일링 - 개선"""
        train_clean = self.safe_data_conversion(train_df)
        test_clean = self.safe_data_conversion(test_df)
        
        numeric_cols = [col for col in train_clean.select_dtypes(include=[np.number]).columns 
                        if col not in ['ID', 'support_needs']]
        
        common_numeric = [col for col in numeric_cols if col in test_clean.columns]
        
        if not common_numeric:
            return train_clean, test_clean
        
        # 혼합 스케일링 전략
        # 대부분 피처는 RobustScaler, 일부는 QuantileTransformer
        
        # 왜도가 높은 피처 식별
        skewed_features = []
        for col in common_numeric:
            skewness = train_clean[col].skew()
            if abs(skewness) > 2:  # 왜도가 높은 피처
                skewed_features.append(col)
        
        normal_features = [col for col in common_numeric if col not in skewed_features]
        
        # RobustScaler for normal features
        if normal_features and 'robust' not in self.scalers:
            self.scalers['robust'] = RobustScaler(quantile_range=(5, 95))  # (25, 75) -> (5, 95)
            train_scaled_robust = self.scalers['robust'].fit_transform(train_clean[normal_features])
            test_scaled_robust = self.scalers['robust'].transform(test_clean[normal_features])
            
            train_clean[normal_features] = train_scaled_robust
            test_clean[normal_features] = test_scaled_robust
        
        # QuantileTransformer for skewed features
        if skewed_features and 'quantile' not in self.scalers:
            self.scalers['quantile'] = QuantileTransformer(
                n_quantiles=100, 
                output_distribution='normal',
                random_state=42
            )
            train_scaled_qt = self.scalers['quantile'].fit_transform(train_clean[skewed_features])
            test_scaled_qt = self.scalers['quantile'].transform(test_clean[skewed_features])
            
            train_clean[skewed_features] = train_scaled_qt
            test_clean[skewed_features] = test_scaled_qt
        
        return train_clean, test_clean
    
    def validate_data_quality(self, train_df, test_df):
        """데이터 품질 검증 - 개선"""
        issues = []
        
        # 훈련 데이터 수치형 컬럼
        train_numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        # 테스트 데이터 수치형 컬럼
        test_numeric_cols = test_df.select_dtypes(include=[np.number]).columns
        
        # 무한값 확인
        train_inf = np.isinf(train_df[train_numeric_cols]).sum().sum()
        test_inf = np.isinf(test_df[test_numeric_cols]).sum().sum()
        
        if train_inf > 0:
            issues.append(f"train_infinite_values: {train_inf}")
            # 무한값 자동 처리
            for col in train_numeric_cols:
                train_df[col] = train_df[col].replace([np.inf, -np.inf], train_df[col].median())
        
        if test_inf > 0:
            issues.append(f"test_infinite_values: {test_inf}")
            # 무한값 자동 처리
            for col in test_numeric_cols:
                test_df[col] = test_df[col].replace([np.inf, -np.inf], test_df[col].median())
        
        # 결측치 확인
        train_nan = train_df[train_numeric_cols].isnull().sum().sum()
        test_nan = test_df[test_numeric_cols].isnull().sum().sum()
        
        # 허용 가능한 결측치 수준
        acceptable_missing = len(train_df) * 0.01  # 1%
        
        if train_nan > acceptable_missing:
            issues.append(f"train_missing_values: {train_nan}")
        if test_nan > acceptable_missing:
            issues.append(f"test_missing_values: {test_nan}")
        
        # 타겟 유효성
        if 'support_needs' in train_df.columns:
            invalid_count = (~train_df['support_needs'].isin([0, 1, 2])).sum()
            if invalid_count > 0:
                issues.append(f"invalid_targets: {invalid_count}")
                # 자동 수정
                train_df['support_needs'] = np.clip(train_df['support_needs'], 0, 2)
        
        return len(issues) == 0, issues
    
    def process_data(self, train_df, test_df, temporal_info=None):
        """전처리 파이프라인 - 개선"""
        if train_df is None or test_df is None or train_df.empty or test_df.empty:
            return None, None
        
        # 시간적 필터링
        if temporal_info is not None:
            train_df = self.apply_temporal_filtering(train_df, temporal_info)
        
        # 결측치 처리
        train_df, test_df = self.handle_missing_values(train_df, test_df)
        
        # 이상치 제거
        train_df = self.remove_outliers_robust(train_df)
        
        # 피처 선택
        if 'support_needs' in train_df.columns:
            selected_features = self.select_features_optimized(train_df, max_features=75)  # 65 -> 75
            
            keep_cols_train = ['ID', 'support_needs'] + selected_features
            keep_cols_test = ['ID'] + [f for f in selected_features if f in test_df.columns]
            
            # 존재하는 컬럼만 선택
            available_train_cols = [col for col in keep_cols_train if col in train_df.columns]
            available_test_cols = [col for col in keep_cols_test if col in test_df.columns]
            
            train_df = train_df[available_train_cols]
            test_df = test_df[available_test_cols]
        
        # 스케일링
        train_df, test_df = self.apply_scaling(train_df, test_df)
        
        # 품질 검증
        quality_ok, issues = self.validate_data_quality(train_df, test_df)
        
        if not quality_ok and issues:
            print(f"데이터 품질 문제: {issues[:3]}")  # 처음 3개만 표시
        
        return train_df, test_df
    
    def prepare_data_temporal_optimized(self, train_df, test_df, val_size=0.2, gap_size=0.01):
        """시간 기반 데이터 준비 - 개선"""
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
            from sklearn.model_selection import train_test_split
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
        
        # 시간 기반 분할 (개선)
        if 'temporal_id' in X.columns:
            temporal_ids = X['temporal_id'].values
            sorted_indices = np.argsort(temporal_ids)
            
            total_samples = len(sorted_indices)
            gap_samples = int(total_samples * gap_size)
            val_samples = int(total_samples * val_size)
            train_samples = total_samples - val_samples - gap_samples
            
            # 최소 샘플 수 확보
            if train_samples < 800 or val_samples < 400:  # 600, 300 -> 800, 400
                # 데이터 부족시 계층화 분할
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=val_size, random_state=42, stratify=y
                )
            else:
                # 시간 기반 분할 (개선된 인덱싱)
                train_end = train_samples
                val_start = train_end + gap_samples
                
                train_indices = sorted_indices[:train_end]
                val_indices = sorted_indices[val_start:]
                
                # 클래스 균형 확인
                train_class_dist = np.bincount(y.iloc[train_indices], minlength=3)
                val_class_dist = np.bincount(y.iloc[val_indices], minlength=3)
                
                # 클래스 1이 너무 적으면 재조정
                if train_class_dist[1] < 100 or val_class_dist[1] < 50:
                    # 계층화 분할로 대체
                    from sklearn.model_selection import train_test_split
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=val_size, random_state=42, stratify=y
                    )
                else:
                    X_train = X.iloc[train_indices]
                    X_val = X.iloc[val_indices]
                    y_train = y.iloc[train_indices]
                    y_val = y.iloc[val_indices]
        else:
            # 계층화 분할
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=42, stratify=y
            )
        
        return X_train, X_val, y_train, y_val, X_test, test_ids

def main():
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        preprocessor = DataPreprocessor()
        train_processed, test_processed = preprocessor.process_data(train_df, test_df)
        
        if train_processed is not None and test_processed is not None:
            X_train, X_val, y_train, y_val, X_test, test_ids = preprocessor.prepare_data_temporal_optimized(
                train_processed, test_processed, val_size=0.2
            )
            
            print(f"전처리 완료:")
            print(f"훈련 세트: {X_train.shape}")
            print(f"검증 세트: {X_val.shape}")
            print(f"테스트 세트: {X_test.shape}")
            
            return preprocessor, X_train, X_val, y_train, y_val, X_test, test_ids
        else:
            print("전처리 실패")
            return preprocessor, None, None, None, None, None, None
            
    except Exception as e:
        print(f"전처리 오류: {e}")
        return None, None, None, None, None, None, None

if __name__ == "__main__":
    main()