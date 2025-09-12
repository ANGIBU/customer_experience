# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, QuantileTransformer, StandardScaler, PowerTransformer, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold, RFECV, SelectFromModel
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from scipy import stats
from scipy.stats import boxcox
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
        self.power_transformer = None
        self.outlier_detector = None
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
        """시간적 필터링"""
        if temporal_info is None or 'temporal_threshold' not in temporal_info:
            return train_df
        
        threshold = temporal_info['temporal_threshold']
        
        if 'temporal_id' not in train_df.columns:
            return train_df
        
        # 관대한 필터링 (최소 보존율 95%)
        safe_mask = train_df['temporal_id'] <= threshold
        safe_data = train_df[safe_mask].copy()
        
        removal_ratio = 1 - (len(safe_data) / len(train_df))
        if removal_ratio > 0.05:  # 5% 이상 제거되면 더 관대한 기준 적용
            percentile_97 = np.percentile(train_df['temporal_id'], 97)
            safe_mask = train_df['temporal_id'] <= percentile_97
            safe_data = train_df[safe_mask].copy()
        
        return safe_data if len(safe_data) > 9000 else train_df
    
    def detect_outliers_advanced(self, train_df):
        """이상치 탐지"""
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        # 생성된 피처들도 포함
        for col in train_df.columns:
            if any(keyword in col for keyword in ['_log', '_sqrt', '_rank', 'customer_value', 'payment_stability', 'usage_intensity']):
                if col not in numeric_cols:
                    numeric_cols.append(col)
        
        available_cols = [col for col in numeric_cols if col in train_df.columns]
        
        if len(available_cols) < 3:
            return train_df
        
        X_numeric = train_df[available_cols].fillna(0)
        
        # Isolation Forest
        try:
            iso_forest = IsolationForest(
                contamination=0.02,  # 2% 이상치
                random_state=42,
                n_jobs=-1
            )
            outlier_mask_iso = iso_forest.fit_predict(X_numeric) == -1
        except:
            outlier_mask_iso = np.zeros(len(train_df), dtype=bool)
        
        # Local Outlier Factor
        try:
            lof = LocalOutlierFactor(
                n_neighbors=20,
                contamination=0.015,  # 1.5% 이상치
                n_jobs=-1
            )
            outlier_mask_lof = lof.fit_predict(X_numeric) == -1
        except:
            outlier_mask_lof = np.zeros(len(train_df), dtype=bool)
        
        # 통계적 이상치 (IQR 방법)
        outlier_mask_stat = np.zeros(len(train_df), dtype=bool)
        for col in available_cols:
            values = train_df[col].dropna()
            if len(values) > 100:
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                
                # 더 관대한 이상치 기준 (3.0 * IQR)
                lower_bound = Q1 - 3.0 * IQR
                upper_bound = Q3 + 3.0 * IQR
                
                col_outliers = (train_df[col] < lower_bound) | (train_df[col] > upper_bound)
                outlier_mask_stat |= col_outliers.fillna(False)
        
        # 투표 기반 이상치 제거 (2개 이상 방법에서 이상치로 판정)
        combined_outliers = (outlier_mask_iso.astype(int) + 
                           outlier_mask_lof.astype(int) + 
                           outlier_mask_stat.astype(int)) >= 2
        
        # 이상치 제거 (최대 3% 제거)
        outlier_ratio = combined_outliers.sum() / len(train_df)
        if outlier_ratio > 0.03:
            # 가장 극단적인 3%만 제거
            outlier_scores = (outlier_mask_iso.astype(int) + 
                            outlier_mask_lof.astype(int) + 
                            outlier_mask_stat.astype(int))
            threshold = np.percentile(outlier_scores, 97)
            combined_outliers = outlier_scores >= threshold
        
        clean_data = train_df[~combined_outliers].copy()
        
        return clean_data if len(clean_data) > 8000 else train_df
    
    def handle_missing_values_advanced(self, train_df, test_df):
        """결측치 처리"""
        train_clean = self.safe_data_conversion(train_df)
        test_clean = self.safe_data_conversion(test_df)
        
        # 수치형 변수
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        # 시간 관련 피처가 있다면 추가
        for col in train_clean.columns:
            if any(keyword in col for keyword in ['temporal', 'after_interaction', '_log', '_sqrt', 'customer_value']):
                if col not in numeric_cols:
                    numeric_cols.append(col)
        
        common_numeric = [col for col in numeric_cols if col in train_clean.columns and col in test_clean.columns]
        
        if common_numeric:
            # 결측치 패턴 분석
            missing_patterns = {}
            for col in common_numeric:
                missing_ratio = train_clean[col].isnull().mean()
                missing_patterns[col] = missing_ratio
            
            # 그룹별 대체 전략
            very_high_missing = [col for col, ratio in missing_patterns.items() if ratio > 0.35]
            high_missing_cols = [col for col, ratio in missing_patterns.items() if 0.15 < ratio <= 0.35]
            medium_missing_cols = [col for col, ratio in missing_patterns.items() if 0.03 < ratio <= 0.15]
            low_missing_cols = [col for col, ratio in missing_patterns.items() if ratio <= 0.03]
            
            # 매우 높은 결측치: 고급 KNN + 다중 대체
            if very_high_missing:
                # 여러 KNN 모델의 평균 사용
                for col in very_high_missing:
                    predictions = []
                    
                    for n_neighbors in [3, 5, 7]:
                        if f'knn_{n_neighbors}' not in self.imputers:
                            self.imputers[f'knn_{n_neighbors}'] = KNNImputer(
                                n_neighbors=n_neighbors, 
                                weights='distance'
                            )
                            train_imputed = self.imputers[f'knn_{n_neighbors}'].fit_transform(train_clean[[col]])
                        else:
                            train_imputed = self.imputers[f'knn_{n_neighbors}'].transform(train_clean[[col]])
                        
                        test_imputed = self.imputers[f'knn_{n_neighbors}'].transform(test_clean[[col]])
                        
                        predictions.append((train_imputed.flatten(), test_imputed.flatten()))
                    
                    # 앙상블 평균
                    if predictions:
                        train_ensemble = np.mean([pred[0] for pred in predictions], axis=0)
                        test_ensemble = np.mean([pred[1] for pred in predictions], axis=0)
                        
                        train_clean[col] = train_ensemble
                        test_clean[col] = test_ensemble
            
            # 높은 결측치: 반복 대체 + KNN 앙상블
            if high_missing_cols:
                # Iterative Imputer
                if 'iterative_advanced' not in self.imputers:
                    self.imputers['iterative_advanced'] = IterativeImputer(
                        max_iter=12, 
                        random_state=42,
                        initial_strategy='median',
                        estimator=RandomForestClassifier(n_estimators=50, random_state=42)
                    )
                    train_iter = self.imputers['iterative_advanced'].fit_transform(train_clean[high_missing_cols])
                else:
                    train_iter = self.imputers['iterative_advanced'].transform(train_clean[high_missing_cols])
                
                test_iter = self.imputers['iterative_advanced'].transform(test_clean[high_missing_cols])
                
                # KNN Imputer
                if 'knn_advanced' not in self.imputers:
                    self.imputers['knn_advanced'] = KNNImputer(n_neighbors=7, weights='distance')
                    train_knn = self.imputers['knn_advanced'].fit_transform(train_clean[high_missing_cols])
                else:
                    train_knn = self.imputers['knn_advanced'].transform(train_clean[high_missing_cols])
                
                test_knn = self.imputers['knn_advanced'].transform(test_clean[high_missing_cols])
                
                # 가중 평균 (Iterative 70%, KNN 30%)
                train_combined = 0.7 * train_iter + 0.3 * train_knn
                test_combined = 0.7 * test_iter + 0.3 * test_knn
                
                train_clean[high_missing_cols] = train_combined
                test_clean[high_missing_cols] = test_combined
            
            # 중간 결측치: Iterative Imputer
            if medium_missing_cols:
                if 'iterative_medium' not in self.imputers:
                    self.imputers['iterative_medium'] = IterativeImputer(
                        max_iter=10, 
                        random_state=42,
                        initial_strategy='mean'
                    )
                    train_imputed_med = self.imputers['iterative_medium'].fit_transform(train_clean[medium_missing_cols])
                else:
                    train_imputed_med = self.imputers['iterative_medium'].transform(train_clean[medium_missing_cols])
                
                test_imputed_med = self.imputers['iterative_medium'].transform(test_clean[medium_missing_cols])
                
                train_clean[medium_missing_cols] = train_imputed_med
                test_clean[medium_missing_cols] = test_imputed_med
            
            # 낮은 결측치: 조건부 평균 대체
            for col in low_missing_cols:
                if col not in self.imputers:
                    # 타겟별 조건부 평균 계산
                    if 'support_needs' in train_clean.columns:
                        target_means = {}
                        for target in [0, 1, 2]:
                            subset = train_clean[train_clean['support_needs'] == target]
                            if len(subset) > 0:
                                target_means[target] = subset[col].median()
                        
                        overall_median = train_clean[col].median()
                        self.imputers[col] = {'target_means': target_means, 'default': overall_median}
                    else:
                        self.imputers[col] = train_clean[col].median()
                
                # 적용
                if isinstance(self.imputers[col], dict):
                    # 타겟 기반 대체
                    for idx in train_clean[train_clean[col].isnull()].index:
                        if 'support_needs' in train_clean.columns:
                            target = train_clean.loc[idx, 'support_needs']
                            if target in self.imputers[col]['target_means']:
                                train_clean.loc[idx, col] = self.imputers[col]['target_means'][target]
                            else:
                                train_clean.loc[idx, col] = self.imputers[col]['default']
                        else:
                            train_clean.loc[idx, col] = self.imputers[col]['default']
                    
                    # 테스트 데이터는 전체 중앙값 사용
                    test_clean[col].fillna(self.imputers[col]['default'], inplace=True)
                else:
                    # 단순 대체
                    train_clean[col].fillna(self.imputers[col], inplace=True)
                    test_clean[col].fillna(self.imputers[col], inplace=True)
        
        # 범주형 변수 - 타겟 기반 대체
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
                        
                        # 전체 최빈값을 기본값으로
                        overall_mode = train_clean[col].mode()
                        default_mode = overall_mode.iloc[0] if len(overall_mode) > 0 else 'Unknown'
                        
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
    
    def apply_power_transforms(self, train_df, test_df):
        """전력 변환 적용"""
        train_transformed = train_df.copy()
        test_transformed = test_df.copy()
        
        # 전력 변환에 적합한 피처 선별
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        transform_cols = [col for col in numeric_cols if col in train_df.columns and col in test_df.columns]
        
        for col in transform_cols:
            train_values = train_df[col].dropna()
            
            # 양수 값만 있는지 확인
            if (train_values > 0).all() and len(train_values) > 100:
                try:
                    # Box-Cox 변환 시도
                    transformed_values, lambda_param = boxcox(train_values + 1e-8)
                    
                    # 변환 적용
                    train_transformed[f'{col}_boxcox'] = boxcox(train_df[col].fillna(train_values.median()) + 1e-8, lmbda=lambda_param)
                    test_transformed[f'{col}_boxcox'] = boxcox(test_df[col].fillna(train_values.median()) + 1e-8, lmbda=lambda_param)
                    
                    # 변환 파라미터 저장
                    self.feature_stats[f'{col}_boxcox_lambda'] = lambda_param
                    
                except:
                    # Box-Cox 실패시 Yeo-Johnson 변환
                    try:
                        if 'power_transformer' not in self.scalers:
                            self.scalers['power_transformer'] = PowerTransformer(method='yeo-johnson')
                            train_yj = self.scalers['power_transformer'].fit_transform(train_df[[col]].fillna(train_values.median()))
                        else:
                            train_yj = self.scalers['power_transformer'].transform(train_df[[col]].fillna(train_values.median()))
                        
                        test_yj = self.scalers['power_transformer'].transform(test_df[[col]].fillna(train_values.median()))
                        
                        train_transformed[f'{col}_yj'] = train_yj.flatten()
                        test_transformed[f'{col}_yj'] = test_yj.flatten()
                        
                    except:
                        continue
        
        return train_transformed, test_transformed
    
    def select_features_ensemble(self, train_df, max_features=75):
        """앙상블 피처 선택"""
        if 'support_needs' not in train_df.columns:
            feature_cols = [col for col in train_df.columns if col not in ['ID']]
            return feature_cols[:min(max_features, len(feature_cols))]
        
        feature_cols = [col for col in train_df.columns if col not in ['ID', 'support_needs']]
        
        if len(feature_cols) <= max_features:
            self.selected_features = feature_cols
            return feature_cols
        
        X = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = np.clip(train_df['support_needs'], 0, 2)
        
        # 1단계: 분산 필터링
        variance_selector = VarianceThreshold(threshold=0.00005)
        X_variance = variance_selector.fit_transform(X)
        variance_features = [feature_cols[i] for i, selected in enumerate(variance_selector.get_support()) if selected]
        
        if len(variance_features) <= max_features:
            self.selected_features = variance_features
            return variance_features
        
        # 2단계: 상관관계 기반 중복 제거
        X_var_df = pd.DataFrame(X_variance, columns=variance_features)
        correlation_matrix = X_var_df.corr().abs()
        
        # 높은 상관관계 (>0.97) 피처 제거
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.97)]
        reduced_features = [f for f in variance_features if f not in high_corr_features]
        
        if len(reduced_features) <= max_features:
            self.selected_features = reduced_features
            return reduced_features
        
        # 3단계: 다중 선택 방법 앙상블
        X_reduced = train_df[reduced_features].fillna(0)
        
        selected_features_sets = []
        
        # 상호정보량 기반 선택
        try:
            mi_scores = mutual_info_classif(X_reduced, y, random_state=42)
            mi_features = [reduced_features[i] for i in np.argsort(mi_scores)[-max_features:]]
            selected_features_sets.append(set(mi_features))
        except:
            pass
        
        # 트리 기반 피처 중요도
        try:
            rf_selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=42),
                max_features=max_features
            )
            rf_selector.fit(X_reduced, y)
            rf_features = [reduced_features[i] for i, selected in enumerate(rf_selector.get_support()) if selected]
            selected_features_sets.append(set(rf_features))
        except:
            pass
        
        # 순환 피처 제거
        try:
            rfecv = RFECV(
                RandomForestClassifier(n_estimators=50, random_state=42),
                step=5,
                cv=3,
                min_features_to_select=min(max_features, len(reduced_features) // 2)
            )
            rfecv.fit(X_reduced, y)
            rfecv_features = [reduced_features[i] for i, selected in enumerate(rfecv.support_) if selected]
            selected_features_sets.append(set(rfecv_features))
        except:
            pass
        
        # 투표 기반 최종 선택
        if selected_features_sets:
            feature_votes = {}
            for feature_set in selected_features_sets:
                for feature in feature_set:
                    feature_votes[feature] = feature_votes.get(feature, 0) + 1
            
            # 투표수 순으로 정렬하여 상위 선택
            sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
            final_features = [feature for feature, votes in sorted_features[:max_features]]
            
            self.selected_features = final_features
            return final_features
        else:
            # 앙상블 실패시 상호정보량만 사용
            mi_scores = mutual_info_classif(X_reduced, y, random_state=42)
            feature_scores = list(zip(reduced_features, mi_scores))
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            selected_features = [f[0] for f in feature_scores[:max_features]]
            self.selected_features = selected_features
            return selected_features
    
    def apply_scaling_ensemble(self, train_df, test_df):
        """앙상블 스케일링"""
        train_clean = self.safe_data_conversion(train_df)
        test_clean = self.safe_data_conversion(test_df)
        
        numeric_cols = [col for col in train_clean.select_dtypes(include=[np.number]).columns 
                        if col not in ['ID', 'support_needs']]
        
        common_numeric = [col for col in numeric_cols if col in test_clean.columns]
        
        if not common_numeric:
            return train_clean, test_clean
        
        # 피처별 최적 스케일러 선택
        scaler_assignments = {}
        
        for col in common_numeric:
            values = train_clean[col].dropna()
            
            if len(values) > 100:
                # 분포 분석
                skewness = stats.skew(values)
                kurtosis = stats.kurtosis(values)
                
                # 스케일러 선택 규칙
                if abs(skewness) > 2 or abs(kurtosis) > 7:
                    # 심하게 치우친 분포: QuantileTransformer
                    scaler_assignments[col] = 'quantile'
                elif abs(skewness) > 1:
                    # 치우친 분포: RobustScaler
                    scaler_assignments[col] = 'robust'
                else:
                    # 정규분포에 가까움: StandardScaler
                    scaler_assignments[col] = 'standard'
            else:
                scaler_assignments[col] = 'robust'  # 기본값
        
        # 스케일러별 그룹화
        quantile_features = [col for col, scaler in scaler_assignments.items() if scaler == 'quantile']
        robust_features = [col for col, scaler in scaler_assignments.items() if scaler == 'robust']
        standard_features = [col for col, scaler in scaler_assignments.items() if scaler == 'standard']
        
        # QuantileTransformer 적용
        if quantile_features:
            if 'quantile' not in self.scalers:
                self.scalers['quantile'] = QuantileTransformer(n_quantiles=min(1000, len(train_clean)), random_state=42)
                train_quantile = self.scalers['quantile'].fit_transform(train_clean[quantile_features])
            else:
                train_quantile = self.scalers['quantile'].transform(train_clean[quantile_features])
            
            test_quantile = self.scalers['quantile'].transform(test_clean[quantile_features])
            
            train_clean[quantile_features] = train_quantile
            test_clean[quantile_features] = test_quantile
        
        # RobustScaler 적용
        if robust_features:
            if 'robust' not in self.scalers:
                self.scalers['robust'] = RobustScaler()
                train_robust = self.scalers['robust'].fit_transform(train_clean[robust_features])
            else:
                train_robust = self.scalers['robust'].transform(train_clean[robust_features])
            
            test_robust = self.scalers['robust'].transform(test_clean[robust_features])
            
            train_clean[robust_features] = train_robust
            test_clean[robust_features] = test_robust
        
        # StandardScaler 적용
        if standard_features:
            if 'standard' not in self.scalers:
                self.scalers['standard'] = StandardScaler()
                train_standard = self.scalers['standard'].fit_transform(train_clean[standard_features])
            else:
                train_standard = self.scalers['standard'].transform(train_clean[standard_features])
            
            test_standard = self.scalers['standard'].transform(test_clean[standard_features])
            
            train_clean[standard_features] = train_standard
            test_clean[standard_features] = test_standard
        
        return train_clean, test_clean
    
    def apply_dimensionality_reduction(self, train_df, test_df):
        """차원 축소"""
        if 'support_needs' not in train_df.columns:
            return train_df, test_df
        
        feature_cols = [col for col in train_df.columns if col not in ['ID', 'support_needs']]
        
        if len(feature_cols) < 20:
            return train_df, test_df
        
        X_train = train_df[feature_cols].fillna(0)
        X_test = test_df[feature_cols].fillna(0)
        
        train_enhanced = train_df.copy()
        test_enhanced = test_df.copy()
        
        # PCA
        try:
            if self.pca_transformer is None:
                self.pca_transformer = PCA(n_components=min(15, len(feature_cols) // 4), random_state=42)
                pca_train = self.pca_transformer.fit_transform(X_train)
            else:
                pca_train = self.pca_transformer.transform(X_train)
            
            pca_test = self.pca_transformer.transform(X_test)
            
            # PCA 컴포넌트 추가
            for i in range(pca_train.shape[1]):
                train_enhanced[f'pca_{i}'] = pca_train[:, i]
                test_enhanced[f'pca_{i}'] = pca_test[:, i]
                
        except:
            pass
        
        # TruncatedSVD
        try:
            if 'svd' not in self.scalers:
                self.scalers['svd'] = TruncatedSVD(n_components=min(10, len(feature_cols) // 6), random_state=42)
                svd_train = self.scalers['svd'].fit_transform(X_train)
            else:
                svd_train = self.scalers['svd'].transform(X_train)
            
            svd_test = self.scalers['svd'].transform(X_test)
            
            # SVD 컴포넌트 추가
            for i in range(svd_train.shape[1]):
                train_enhanced[f'svd_{i}'] = svd_train[:, i]
                test_enhanced[f'svd_{i}'] = svd_test[:, i]
                
        except:
            pass
        
        # FastICA
        try:
            if 'ica' not in self.scalers:
                self.scalers['ica'] = FastICA(n_components=min(8, len(feature_cols) // 8), random_state=42)
                ica_train = self.scalers['ica'].fit_transform(X_train)
            else:
                ica_train = self.scalers['ica'].transform(X_train)
            
            ica_test = self.scalers['ica'].transform(X_test)
            
            # ICA 컴포넌트 추가
            for i in range(ica_train.shape[1]):
                train_enhanced[f'ica_{i}'] = ica_train[:, i]
                test_enhanced[f'ica_{i}'] = ica_test[:, i]
                
        except:
            pass
        
        return train_enhanced, test_enhanced
    
    def validate_data_quality(self, train_df, test_df):
        """데이터 품질 검증"""
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
        if test_inf > 0:
            issues.append(f"test_infinite_values: {test_inf}")
        
        # 결측치 확인
        train_nan = train_df[train_numeric_cols].isnull().sum().sum()
        test_nan = test_df[test_numeric_cols].isnull().sum().sum()
        
        if train_nan > 0:
            issues.append(f"train_missing_values: {train_nan}")
        if test_nan > 0:
            issues.append(f"test_missing_values: {test_nan}")
        
        # 타겟 유효성
        if 'support_needs' in train_df.columns:
            invalid_count = (~train_df['support_needs'].isin([0, 1, 2])).sum()
            if invalid_count > 0:
                issues.append(f"invalid_targets: {invalid_count}")
        
        # 피처 분산 확인
        for col in train_numeric_cols:
            if col in train_df.columns:
                variance = train_df[col].var()
                if variance < 1e-10:
                    issues.append(f"zero_variance_{col}")
        
        return len(issues) == 0, issues
    
    def process_data(self, train_df, test_df, temporal_info=None):
        """전처리 파이프라인"""
        if train_df is None or test_df is None or train_df.empty or test_df.empty:
            return None, None
        
        # 시간적 필터링
        if temporal_info is not None:
            train_df = self.apply_temporal_filtering(train_df, temporal_info)
        
        # 이상치 탐지 및 제거
        train_df = self.detect_outliers_advanced(train_df)
        
        # 결측치 처리
        train_df, test_df = self.handle_missing_values_advanced(train_df, test_df)
        
        # 전력 변환
        train_df, test_df = self.apply_power_transforms(train_df, test_df)
        
        # 피처 선택
        if 'support_needs' in train_df.columns:
            selected_features = self.select_features_ensemble(train_df, max_features=75)
            
            keep_cols_train = ['ID', 'support_needs'] + selected_features
            keep_cols_test = ['ID'] + [f for f in selected_features if f in test_df.columns]
            
            # 존재하는 컬럼만 선택
            available_train_cols = [col for col in keep_cols_train if col in train_df.columns]
            available_test_cols = [col for col in keep_cols_test if col in test_df.columns]
            
            train_df = train_df[available_train_cols]
            test_df = test_df[available_test_cols]
        
        # 스케일링
        train_df, test_df = self.apply_scaling_ensemble(train_df, test_df)
        
        # 차원 축소
        train_df, test_df = self.apply_dimensionality_reduction(train_df, test_df)
        
        # 품질 검증
        quality_ok, issues = self.validate_data_quality(train_df, test_df)
        
        return train_df, test_df
    
    def prepare_data_temporal_optimized(self, train_df, test_df, val_size=0.18, gap_size=0.004):
        """시간 기반 데이터 준비"""
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
        
        # 시간 기반 분할 with 축소된 갭 (0.4%)
        if 'temporal_id' in X.columns:
            temporal_ids = X['temporal_id'].values
            sorted_indices = np.argsort(temporal_ids)
            
            total_samples = len(sorted_indices)
            gap_samples = int(total_samples * gap_size)  # 0.4%
            val_samples = int(total_samples * val_size)
            train_samples = total_samples - val_samples - gap_samples
            
            if train_samples < 700 or val_samples < 350:
                # 데이터 부족시 계층화 분할
                sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
                train_idx, val_idx = next(sss.split(X, y))
                
                X_train = X.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_train = y.iloc[train_idx]
                y_val = y.iloc[val_idx]
            else:
                # 훈련-갭-검증 순서로 분할
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
        return None, None, None, None, None, None, None

if __name__ == "__main__":
    main()