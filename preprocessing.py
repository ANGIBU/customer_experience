# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.feature_selector = None
        self.imputer = None
        self.selected_features = None
        self.temporal_cutoff = None
        self.smote = None
        self.label_encoders = {}
        
    def apply_temporal_filtering(self, train_df):
        """시간적 필터링"""
        if 'ID' not in train_df.columns:
            return train_df
        
        try:
            temporal_ids = []
            for id_val in train_df['ID']:
                try:
                    if '_' in str(id_val):
                        num = int(str(id_val).split('_')[1])
                        temporal_ids.append(num)
                    else:
                        temporal_ids.append(0)
                except:
                    temporal_ids.append(0)
            
            if temporal_ids and len(temporal_ids) > 100:
                threshold = np.percentile(temporal_ids, 80)
                safe_indices = [i for i, tid in enumerate(temporal_ids) if tid <= threshold]
                
                if len(safe_indices) > len(temporal_ids) * 0.6:
                    filtered_data = train_df.iloc[safe_indices].copy()
                else:
                    filtered_data = train_df.copy()
            else:
                filtered_data = train_df.copy()
            
            return filtered_data
            
        except Exception as e:
            print(f"시간적 필터링 오류: {e}")
            return train_df
    
    def encode_categorical_features(self, train_df, test_df):
        """범주형 피처 인코딩"""
        train_encoded = train_df.copy()
        test_encoded = test_df.copy()
        
        categorical_cols = ['gender', 'subscription_type']
        
        for col in categorical_cols:
            if col in train_df.columns and col in test_df.columns:
                train_encoded[col] = train_encoded[col].astype(str).fillna('Unknown')
                test_encoded[col] = test_encoded[col].astype(str).fillna('Unknown')
                
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    combined_data = pd.concat([train_encoded[col], test_encoded[col]])
                    self.label_encoders[col].fit(combined_data)
                
                train_encoded[col] = self.label_encoders[col].transform(train_encoded[col])
                test_encoded[col] = self.label_encoders[col].transform(test_encoded[col])
        
        return train_encoded, test_encoded
    
    def handle_missing_values(self, train_df, test_df):
        """결측치 처리"""
        train_clean = train_df.copy()
        test_clean = test_df.copy()
        
        categorical_cols = ['gender', 'subscription_type']
        numeric_cols = []
        
        for col in train_clean.columns:
            if col not in categorical_cols and col not in ['ID', 'support_needs']:
                try:
                    train_clean[col] = pd.to_numeric(train_clean[col], errors='coerce')
                    train_clean[col] = train_clean[col].fillna(train_clean[col].median())
                    train_clean[col] = train_clean[col].replace([np.inf, -np.inf], 0)
                    numeric_cols.append(col)
                except:
                    continue
        
        for col in test_clean.columns:
            if col in numeric_cols:
                try:
                    test_clean[col] = pd.to_numeric(test_clean[col], errors='coerce')
                    test_clean[col] = test_clean[col].fillna(test_clean[col].median())
                    test_clean[col] = test_clean[col].replace([np.inf, -np.inf], 0)
                except:
                    test_clean[col] = 0
        
        common_numeric = [col for col in numeric_cols if col in test_clean.columns]
        
        if common_numeric and len(common_numeric) > 0:
            try:
                if self.imputer is None:
                    self.imputer = KNNImputer(n_neighbors=7, weights='distance')
                    train_imputed = self.imputer.fit_transform(train_clean[common_numeric])
                else:
                    train_imputed = self.imputer.transform(train_clean[common_numeric])
                
                test_imputed = self.imputer.transform(test_clean[common_numeric])
                
                train_clean[common_numeric] = train_imputed
                test_clean[common_numeric] = test_imputed
            except Exception as e:
                print(f"KNN imputer 오류, 기본값으로 대체: {e}")
                for col in common_numeric:
                    median_val = train_clean[col].median()
                    train_clean[col] = train_clean[col].fillna(median_val)
                    test_clean[col] = test_clean[col].fillna(median_val)
        
        return train_clean, test_clean
    
    def remove_outliers(self, train_df):
        """이상치 제거"""
        train_clean = train_df.copy()
        
        categorical_cols = ['gender', 'subscription_type']
        
        for col in train_clean.columns:
            if col not in categorical_cols and col not in ['ID', 'support_needs']:
                try:
                    if train_clean[col].dtype in [np.number]:
                        values = train_clean[col].dropna()
                        
                        if len(values) > 100:
                            q1 = values.quantile(0.05)
                            q99 = values.quantile(0.95)
                            
                            train_clean[col] = np.clip(train_clean[col], q1, q99)
                except Exception as e:
                    continue
        
        return train_clean
    
    def select_features(self, train_df, max_features=25):
        """피처 선택"""
        if 'support_needs' not in train_df.columns:
            feature_cols = [col for col in train_df.columns if col not in ['ID']]
            return feature_cols[:min(max_features, len(feature_cols))]
        
        categorical_cols = ['gender', 'subscription_type']
        all_feature_cols = [col for col in train_df.columns if col not in ['ID', 'support_needs']]
        
        if len(all_feature_cols) <= max_features:
            self.selected_features = all_feature_cols
            return all_feature_cols
        
        numeric_cols = []
        for col in all_feature_cols:
            if col not in categorical_cols and train_df[col].dtype in [np.number]:
                numeric_cols.append(col)
        
        categorical_cols_available = [col for col in categorical_cols if col in all_feature_cols]
        
        if not numeric_cols:
            self.selected_features = all_feature_cols[:max_features]
            return all_feature_cols[:max_features]
        
        try:
            X_numeric = train_df[numeric_cols].copy()
            y = train_df['support_needs'].copy()
            
            variance_selector = VarianceThreshold(threshold=0.005)
            X_variance = variance_selector.fit_transform(X_numeric)
            variance_features = [numeric_cols[i] for i, selected in enumerate(variance_selector.get_support()) if selected]
            
            available_slots = max_features - len(categorical_cols_available)
            
            if len(variance_features) <= available_slots:
                selected_numeric = variance_features
            else:
                X_var_df = pd.DataFrame(X_variance, columns=variance_features)
                mi_selector = SelectKBest(score_func=mutual_info_classif, k=available_slots)
                mi_selector.fit(X_var_df, y)
                
                selected_numeric = [variance_features[i] for i, selected in enumerate(mi_selector.get_support()) if selected]
            
            selected_features = selected_numeric + categorical_cols_available
            
        except Exception as e:
            print(f"피처 선택 오류: {e}")
            selected_features = all_feature_cols[:max_features]
        
        self.selected_features = selected_features
        return selected_features
    
    def apply_scaling(self, train_df, test_df):
        """스케일링 적용"""
        train_clean = train_df.copy()
        test_clean = test_df.copy()
        
        categorical_cols = ['gender', 'subscription_type']
        
        numeric_cols = []
        for col in train_clean.columns:
            if col not in categorical_cols and col not in ['ID', 'support_needs']:
                if train_clean[col].dtype in [np.number]:
                    numeric_cols.append(col)
        
        common_numeric = [col for col in numeric_cols if col in test_clean.columns and test_clean[col].dtype in [np.number]]
        
        if not common_numeric:
            return train_clean, test_clean
        
        try:
            if self.scaler is None:
                self.scaler = RobustScaler(quantile_range=(5.0, 95.0))
                scaled_train = self.scaler.fit_transform(train_clean[common_numeric])
            else:
                scaled_train = self.scaler.transform(train_clean[common_numeric])
            
            scaled_test = self.scaler.transform(test_clean[common_numeric])
            
            train_clean[common_numeric] = scaled_train
            test_clean[common_numeric] = scaled_test
        except Exception as e:
            print(f"스케일링 오류: {e}")
        
        return train_clean, test_clean
    
    def apply_smote_optimized(self, X_train, y_train):
        """SMOTE 적용 최적화"""
        try:
            y_train_int = y_train.astype(int)
            class_counts = np.bincount(y_train_int)
            min_class_count = np.min(class_counts[class_counts > 0])
            
            if min_class_count < 50:
                return X_train, y_train
            
            max_class_count = np.max(class_counts)
            imbalance_ratio = max_class_count / min_class_count
            
            if imbalance_ratio < 1.5:
                return X_train, y_train
            
            k_neighbors = min(7, min_class_count - 1)
            
            if k_neighbors <= 0:
                return X_train, y_train
            
            if self.smote is None:
                self.smote = SMOTE(
                    sampling_strategy='auto',
                    k_neighbors=k_neighbors,
                    random_state=42
                )
            
            X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train_int)
            
            if len(X_resampled) > len(X_train) * 2.0:
                return X_train, y_train
            
            return X_resampled, y_resampled
            
        except Exception as e:
            return X_train, y_train
    
    def validate_data_quality(self, train_df, test_df):
        """데이터 품질 검증"""
        issues = []
        
        try:
            categorical_cols = ['gender', 'subscription_type']
            
            for col in train_df.columns:
                if col not in categorical_cols and col not in ['ID', 'support_needs']:
                    if train_df[col].dtype in [np.number]:
                        train_inf = np.isinf(train_df[col]).sum()
                        train_nan = train_df[col].isnull().sum()
                        
                        if train_inf > 0:
                            issues.append(f"train_{col}_infinite: {train_inf}")
                        if train_nan > 0:
                            issues.append(f"train_{col}_missing: {train_nan}")
            
            for col in test_df.columns:
                if col not in categorical_cols and col not in ['ID']:
                    if test_df[col].dtype in [np.number]:
                        test_inf = np.isinf(test_df[col]).sum()
                        test_nan = test_df[col].isnull().sum()
                        
                        if test_inf > 0:
                            issues.append(f"test_{col}_infinite: {test_inf}")
                        if test_nan > 0:
                            issues.append(f"test_{col}_missing: {test_nan}")
            
            if 'support_needs' in train_df.columns:
                invalid_count = (~train_df['support_needs'].isin([0, 1, 2])).sum()
                if invalid_count > 0:
                    issues.append(f"invalid_targets: {invalid_count}")
        except Exception as e:
            print(f"품질 검증 오류: {e}")
        
        return len(issues) == 0, issues
    
    def process_data(self, train_df, test_df):
        """전처리 파이프라인"""
        if train_df is None or test_df is None or train_df.empty or test_df.empty:
            return None, None
        
        try:
            train_df = self.apply_temporal_filtering(train_df)
            
            train_df, test_df = self.encode_categorical_features(train_df, test_df)
            
            train_df, test_df = self.handle_missing_values(train_df, test_df)
            
            train_df = self.remove_outliers(train_df)
            
            if 'support_needs' in train_df.columns:
                selected_features = self.select_features(train_df, max_features=25)
                
                keep_cols_train = ['ID', 'support_needs'] + selected_features
                keep_cols_test = ['ID'] + [f for f in selected_features if f in test_df.columns]
                
                available_train_cols = [col for col in keep_cols_train if col in train_df.columns]
                available_test_cols = [col for col in keep_cols_test if col in test_df.columns]
                
                train_df = train_df[available_train_cols]
                test_df = test_df[available_test_cols]
            
            train_df, test_df = self.apply_scaling(train_df, test_df)
            
            quality_ok, issues = self.validate_data_quality(train_df, test_df)
            
            return train_df, test_df
            
        except Exception as e:
            print(f"전처리 파이프라인 오류: {e}")
            return None, None
    
    def prepare_temporal_split(self, train_df, test_df, val_size=0.25):
        """시간적 분할"""
        if train_df is None or test_df is None:
            return None, None, None, None, None, None
            
        try:
            if 'support_needs' not in train_df.columns:
                feature_cols = [col for col in train_df.columns if col not in ['ID']]
                common_features = [col for col in feature_cols if col in test_df.columns]
                
                if not common_features:
                    raise ValueError("공통 피처 없음")
                    
                X_train = train_df[common_features].fillna(0)
                X_test = test_df[common_features].fillna(0)
                test_ids = test_df['ID'] if 'ID' in test_df.columns else pd.Series(range(len(test_df)))
                
                X_train_split, X_val_split, y_dummy, y_val_dummy = train_test_split(
                    X_train, np.zeros(len(X_train)), test_size=val_size, random_state=42
                )
                
                return X_train_split, X_val_split, y_dummy[:len(X_train_split)], y_val_dummy, X_test, test_ids

            train_cols = set(train_df.columns)
            test_cols = set(test_df.columns)
            common_features = list((train_cols & test_cols) - {'ID', 'support_needs'})
            
            if not common_features:
                raise ValueError("공통 피처 없음")
            
            common_features = sorted(common_features)
            
            safe_train = train_df.copy()
            safe_test = test_df.copy()
            
            try:
                temporal_ids = []
                for id_val in safe_train['ID']:
                    try:
                        if '_' in str(id_val):
                            num = int(str(id_val).split('_')[1])
                            temporal_ids.append(num)
                        else:
                            temporal_ids.append(0)
                    except:
                        temporal_ids.append(0)
                
                if temporal_ids and len(set(temporal_ids)) > 10:
                    safe_train['temp_order'] = temporal_ids
                    safe_train = safe_train.sort_values('temp_order')
                    
                    gap_size = int(len(safe_train) * 0.05)
                    split_point = int(len(safe_train) * (1 - val_size)) - gap_size
                    
                    train_part = safe_train.iloc[:split_point]
                    val_part = safe_train.iloc[split_point + gap_size:]
                    
                    train_part = train_part.drop('temp_order', axis=1)
                    val_part = val_part.drop('temp_order', axis=1)
                else:
                    train_part, val_part = train_test_split(
                        safe_train, test_size=val_size, random_state=42, stratify=safe_train['support_needs']
                    )
                    
            except Exception as e:
                print(f"시간적 분할 실패, 랜덤 분할 사용: {e}")
                train_part, val_part = train_test_split(
                    safe_train, test_size=val_size, random_state=42, stratify=safe_train['support_needs']
                )
            
            X_train = train_part[common_features].fillna(0)
            y_train = train_part['support_needs']
            X_val = val_part[common_features].fillna(0)
            y_val = val_part['support_needs']
            
            X_test = safe_test[common_features].fillna(0)
            test_ids = safe_test['ID'] if 'ID' in safe_test.columns else pd.Series(range(len(safe_test)))
            
            X_train_resampled, y_train_resampled = self.apply_smote_optimized(X_train, y_train)
            
            return X_train_resampled, X_val, y_train_resampled, y_val, X_test, test_ids
            
        except Exception as e:
            print(f"데이터 분할 오류: {e}")
            return None, None, None, None, None, None

def main():
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        preprocessor = DataPreprocessor()
        train_processed, test_processed = preprocessor.process_data(train_df, test_df)
        
        if train_processed is not None and test_processed is not None:
            X_train, X_val, y_train, y_val, X_test, test_ids = preprocessor.prepare_temporal_split(
                train_processed, test_processed
            )
            
            return preprocessor, X_train, X_val, y_train, y_val, X_test, test_ids
        else:
            return preprocessor, None, None, None, None, None, None
            
    except Exception as e:
        return None, None, None, None, None, None, None

if __name__ == "__main__":
    main()