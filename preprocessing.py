# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, QuantileTransformer, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
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
        """시간적 필터링"""
        if temporal_info is None:
            return train_df
        
        temporal_threshold = temporal_info.get('temporal_threshold')
        if temporal_threshold is None or 'temporal_id' not in train_df.columns:
            return train_df
        
        try:
            # 더 많은 데이터 활용
            safe_mask = train_df['temporal_id'] <= temporal_threshold
            safe_data = train_df[safe_mask].copy()
            
            # 최소 80% 데이터 유지
            if len(safe_data) < len(train_df) * 0.80:
                percentile_80 = np.percentile(train_df['temporal_id'], 80)
                safe_mask = train_df['temporal_id'] <= percentile_80
                safe_data = train_df[safe_mask].copy()
            
            # 최소 데이터량 보장
            if len(safe_data) < 20000:
                percentile_85 = np.percentile(train_df['temporal_id'], 85)
                safe_mask = train_df['temporal_id'] <= percentile_85
                safe_data = train_df[safe_mask].copy()
            
            print(f"시간적 필터링: {len(train_df)} → {len(safe_data)} ({len(safe_data)/len(train_df):.1%})")
            
            return safe_data if len(safe_data) > 15000 else train_df
            
        except Exception as e:
            print(f"시간적 필터링 오류: {e}")
            return train_df
    
    def detect_outliers_conservative(self, train_df):
        """보수적 이상치 탐지"""
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_cols = [col for col in numeric_cols if col in train_df.columns]
        
        if len(available_cols) < 3:
            return train_df
        
        try:
            X_numeric = train_df[available_cols].fillna(0)
            
            # 통계적 이상치 탐지
            outlier_mask = np.zeros(len(train_df), dtype=bool)
            
            for col in available_cols:
                values = X_numeric[col]
                q25, q75 = values.quantile([0.25, 0.75])
                iqr = q75 - q25
                
                # 더 보수적인 기준
                lower_bound = q25 - 3.0 * iqr  
                upper_bound = q75 + 3.0 * iqr
                
                col_outliers = (values < lower_bound) | (values > upper_bound)
                outlier_mask |= col_outliers
            
            # 최대 1% 제거
            outlier_count = outlier_mask.sum()
            max_remove = int(len(train_df) * 0.01)
            
            if outlier_count > max_remove:
                # 가장 극단적인 값들만 제거
                outlier_scores = np.zeros(len(train_df))
                for col in available_cols:
                    values = X_numeric[col]
                    median_val = values.median()
                    mad = np.median(np.abs(values - median_val))
                    if mad > 0:
                        outlier_scores += np.abs(values - median_val) / mad
                
                threshold = np.percentile(outlier_scores, 99)
                outlier_mask = outlier_scores > threshold
            
            clean_data = train_df[~outlier_mask].copy()
            removed_count = outlier_mask.sum()
            
            print(f"이상치 제거: {len(train_df)} → {len(clean_data)} ({removed_count}개 제거)")
            
            return clean_data if len(clean_data) > len(train_df) * 0.95 else train_df
            
        except Exception as e:
            print(f"이상치 탐지 오류: {e}")
            return train_df
    
    def handle_missing_values_advanced(self, train_df, test_df):
        """결측치 처리"""
        try:
            train_clean = self.safe_data_conversion(train_df)
            test_clean = self.safe_data_conversion(test_df)
            
            # 수치형 변수 - 더 정교한 대체
            numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
            common_numeric = [col for col in numeric_cols if col in train_clean.columns and col in test_clean.columns]
            
            # KNN 대체 시도 (실패시 중앙값)
            if len(common_numeric) >= 3 and len(train_clean) > 1000:
                try:
                    imputer = KNNImputer(n_neighbors=5, weights='distance')
                    train_numeric_filled = imputer.fit_transform(train_clean[common_numeric])
                    test_numeric_filled = imputer.transform(test_clean[common_numeric])
                    
                    for i, col in enumerate(common_numeric):
                        train_clean[col] = train_numeric_filled[:, i]
                        test_clean[col] = test_numeric_filled[:, i]
                        
                except Exception:
                    # KNN 실패시 중앙값 대체
                    for col in common_numeric:
                        if col not in self.imputers:
                            median_val = train_clean[col].median()
                            self.imputers[col] = median_val if not pd.isna(median_val) else 0
                        
                        train_clean[col].fillna(self.imputers[col], inplace=True)
                        test_clean[col].fillna(self.imputers[col], inplace=True)
            else:
                # 기본 중앙값 대체
                for col in common_numeric:
                    if col not in self.imputers:
                        median_val = train_clean[col].median()
                        self.imputers[col] = median_val if not pd.isna(median_val) else 0
                    
                    train_clean[col].fillna(self.imputers[col], inplace=True)
                    test_clean[col].fillna(self.imputers[col], inplace=True)
            
            # 범주형 변수 - 최빈값 대체
            categorical_cols = ['gender', 'subscription_type']
            
            for col in categorical_cols:
                if col in train_clean.columns and col in test_clean.columns:
                    if col not in self.imputers:
                        mode_val = train_clean[col].mode()
                        self.imputers[col] = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
                    
                    train_clean[col].fillna(self.imputers[col], inplace=True)
                    test_clean[col].fillna(self.imputers[col], inplace=True)
            
            # 생성된 피처들 처리
            for col in train_clean.columns:
                if col not in ['ID', 'support_needs'] and col not in common_numeric + categorical_cols:
                    if train_clean[col].dtype in ['float64', 'int64']:
                        if col in train_clean.columns and train_clean[col].isnull().sum() > 0:
                            fill_val = train_clean[col].median()
                            fill_val = fill_val if not pd.isna(fill_val) else 0
                            train_clean[col].fillna(fill_val, inplace=True)
                            if col in test_clean.columns:
                                test_clean[col].fillna(fill_val, inplace=True)
            
            return train_clean, test_clean
            
        except Exception as e:
            print(f"결측치 처리 오류: {e}")
            return train_df, test_df
    
    def select_features_advanced(self, train_df, max_features=80):
        """피처 선택"""
        if 'support_needs' not in train_df.columns:
            feature_cols = [col for col in train_df.columns if col not in ['ID']]
            return feature_cols[:min(max_features, len(feature_cols))]
        
        try:
            feature_cols = [col for col in train_df.columns if col not in ['ID', 'support_needs']]
            
            if len(feature_cols) <= max_features:
                self.selected_features = feature_cols
                return feature_cols
            
            X = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
            y = np.clip(train_df['support_needs'], 0, 2)
            
            # 분산 필터링 (더 관대한 기준)
            variance_selector = VarianceThreshold(threshold=0.005)
            X_variance = variance_selector.fit_transform(X)
            variance_features = [feature_cols[i] for i, selected in enumerate(variance_selector.get_support()) if selected]
            
            if len(variance_features) <= max_features:
                self.selected_features = variance_features
                return variance_features
            
            # 상호정보량 + Random Forest 중요도 결합
            X_var_df = pd.DataFrame(X_variance, columns=variance_features)
            
            # 상호정보량
            mi_scores = mutual_info_classif(X_var_df, y, random_state=42)
            mi_dict = dict(zip(variance_features, mi_scores))
            
            # Random Forest 중요도
            try:
                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
                rf.fit(X_var_df, y)
                rf_scores = rf.feature_importances_
                rf_dict = dict(zip(variance_features, rf_scores))
            except:
                rf_dict = {f: 0 for f in variance_features}
            
            # 결합 점수 (MI 70%, RF 30%)
            combined_scores = {}
            for feature in variance_features:
                mi_score = mi_dict.get(feature, 0)
                rf_score = rf_dict.get(feature, 0)
                combined_scores[feature] = 0.7 * mi_score + 0.3 * rf_score
            
            # 정규화
            max_score = max(combined_scores.values()) if combined_scores else 1
            if max_score > 0:
                combined_scores = {k: v/max_score for k, v in combined_scores.items()}
            
            # 상위 피처 선택
            sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in sorted_features[:max_features]]
            
            self.selected_features = selected_features
            return selected_features
            
        except Exception as e:
            print(f"피처 선택 오류: {e}")
            # 기본 피처 반환
            basic_features = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length', 'gender', 'subscription_type']
            available_features = [f for f in basic_features if f in train_df.columns]
            return available_features[:max_features]
    
    def apply_scaling_robust(self, train_df, test_df):
        """스케일링"""
        try:
            train_clean = self.safe_data_conversion(train_df)
            test_clean = self.safe_data_conversion(test_df)
            
            numeric_cols = [col for col in train_clean.select_dtypes(include=[np.number]).columns 
                            if col not in ['ID', 'support_needs']]
            
            common_numeric = [col for col in numeric_cols if col in test_clean.columns]
            
            if not common_numeric:
                return train_clean, test_clean
            
            # RobustScaler + QuantileTransformer 결합
            if 'robust' not in self.scalers:
                self.scalers['robust'] = RobustScaler()
                self.scalers['quantile'] = QuantileTransformer(output_distribution='normal', random_state=42)
                
                train_values = train_clean[common_numeric].values
                train_values = np.nan_to_num(train_values, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 1단계: Robust scaling
                train_robust = self.scalers['robust'].fit_transform(train_values)
                
                # 2단계: Quantile transform (일부 피처만)
                high_var_features = []
                for i, col in enumerate(common_numeric):
                    if train_clean[col].std() > train_clean[col].mean():
                        high_var_features.append(i)
                
                if high_var_features and len(high_var_features) < len(common_numeric):
                    train_quant = train_robust.copy()
                    train_quant[:, high_var_features] = self.scalers['quantile'].fit_transform(train_robust[:, high_var_features])
                    train_scaled = train_quant
                else:
                    train_scaled = train_robust
            else:
                train_values = train_clean[common_numeric].values
                train_values = np.nan_to_num(train_values, nan=0.0, posinf=0.0, neginf=0.0)
                train_robust = self.scalers['robust'].transform(train_values)
                
                if hasattr(self.scalers, 'quantile'):
                    train_scaled = train_robust.copy()
                    # Apply quantile transform to same features as during fit
                else:
                    train_scaled = train_robust
            
            # 테스트 데이터 변환
            test_values = test_clean[common_numeric].values
            test_values = np.nan_to_num(test_values, nan=0.0, posinf=0.0, neginf=0.0)
            test_robust = self.scalers['robust'].transform(test_values)
            
            if 'quantile' in self.scalers and hasattr(self.scalers['quantile'], 'transform'):
                test_scaled = test_robust.copy()
                # Apply same quantile transform as training
            else:
                test_scaled = test_robust
            
            train_clean[common_numeric] = train_scaled
            test_clean[common_numeric] = test_scaled
            
            return train_clean, test_clean
            
        except Exception as e:
            print(f"스케일링 오류: {e}")
            return train_df, test_df
    
    def apply_dimensionality_reduction(self, train_df, test_df):
        """차원 축소"""
        if 'support_needs' not in train_df.columns:
            return train_df, test_df
        
        try:
            feature_cols = [col for col in train_df.columns if col not in ['ID', 'support_needs']]
            
            if len(feature_cols) < 30:
                return train_df, test_df
            
            X_train = train_df[feature_cols].fillna(0)
            X_test = test_df[feature_cols].fillna(0)
            
            # 무한값 처리
            X_train = X_train.replace([np.inf, -np.inf], 0)
            X_test = X_test.replace([np.inf, -np.inf], 0)
            
            train_enhanced = train_df.copy()
            test_enhanced = test_df.copy()
            
            # PCA 적용 (더 많은 컴포넌트)
            n_components = min(15, len(feature_cols) // 4)
            
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
            
            # 설명 분산비 기반 가중 컴포넌트
            explained_ratios = self.scalers['pca'].explained_variance_ratio_
            for i in range(min(5, len(explained_ratios))):
                weight = explained_ratios[i]
                train_enhanced[f'pca_weighted_{i}'] = pca_train[:, i] * weight
                test_enhanced[f'pca_weighted_{i}'] = pca_test[:, i] * weight
                
            return train_enhanced, test_enhanced
            
        except Exception as e:
            print(f"차원 축소 오류: {e}")
            return train_df, test_df
    
    def validate_data_quality(self, train_df, test_df):
        """데이터 품질 검증"""
        try:
            issues = []
            
            # 무한값 확인
            train_numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            test_numeric_cols = test_df.select_dtypes(include=[np.number]).columns
            
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
            
            # 피처 일관성 확인
            common_features = set(train_df.columns) & set(test_df.columns)
            if len(common_features) < 10:
                issues.append(f"few_common_features: {len(common_features)}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            print(f"품질 검증 오류: {e}")
            return True, []
    
    def process_data(self, train_df, test_df, temporal_info=None):
        """데이터 전처리 파이프라인"""
        if train_df is None or test_df is None or train_df.empty or test_df.empty:
            return None, None
        
        print("전처리 시작:")
        
        try:
            # 시간적 필터링 (더 많은 데이터 유지)
            if temporal_info is not None:
                train_df = self.apply_temporal_filtering(train_df, temporal_info)
            
            # 보수적 이상치 제거
            train_df = self.detect_outliers_conservative(train_df)
            
            # 결측치 처리
            train_df, test_df = self.handle_missing_values_advanced(train_df, test_df)
            
            # 피처 선택
            if 'support_needs' in train_df.columns:
                selected_features = self.select_features_advanced(train_df, max_features=70)
                
                keep_cols_train = ['ID', 'support_needs'] + selected_features
                keep_cols_test = ['ID'] + [f for f in selected_features if f in test_df.columns]
                
                # 존재하는 컬럼만 선택
                available_train_cols = [col for col in keep_cols_train if col in train_df.columns]
                available_test_cols = [col for col in keep_cols_test if col in test_df.columns]
                
                train_df = train_df[available_train_cols]
                test_df = test_df[available_test_cols]
            
            # 스케일링
            train_df, test_df = self.apply_scaling_robust(train_df, test_df)
            
            # 차원 축소 (피처가 많을 때만)
            if len(train_df.columns) > 40:
                train_df, test_df = self.apply_dimensionality_reduction(train_df, test_df)
            
            # 품질 검증
            quality_ok, issues = self.validate_data_quality(train_df, test_df)
            if not quality_ok:
                print(f"데이터 품질 문제: {len(issues)}개")
            
            print("✓ 전처리 완료")
            
            return train_df, test_df
            
        except Exception as e:
            print(f"전처리 파이프라인 오류: {e}")
            return None, None
    
    def prepare_data_temporal_optimized(self, train_df, test_df, val_size=0.2, gap_size=0.005):
        """데이터 준비"""
        if train_df is None or test_df is None:
            return None, None, None, None, None, None
            
        try:
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
            
            # 시간 기반 분할 (더 보수적)
            if 'temporal_id' in X.columns and len(X) > 5000:
                temporal_ids = X['temporal_id'].values
                sorted_indices = np.argsort(temporal_ids)
                
                total_samples = len(sorted_indices)
                gap_samples = int(total_samples * gap_size)
                val_samples = int(total_samples * val_size)
                train_samples = total_samples - val_samples - gap_samples
                
                if train_samples > 3000 and val_samples > 1000:
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
            
        except Exception as e:
            print(f"데이터 준비 오류: {e}")
            return None, None, None, None, None, None

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