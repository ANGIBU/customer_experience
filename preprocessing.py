# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
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
        
    def safe_data_conversion(self, df):
        """안전한 데이터 변환"""
        df_clean = df.copy()
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        return df_clean
        
    def apply_temporal_filtering(self, train_df):
        """시간적 필터링"""
        if 'ID' not in train_df.columns:
            return train_df
        
        # ID에서 시간 정보 추출
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
        
        train_df = train_df.copy()
        train_df['temporal_order'] = temporal_ids
        
        # 70% 분위수를 기준으로 안전한 데이터만 사용
        threshold = np.percentile(temporal_ids, 70)
        safe_mask = train_df['temporal_order'] <= threshold
        
        filtered_data = train_df[safe_mask].copy()
        
        # temporal_order 컬럼 제거
        if 'temporal_order' in filtered_data.columns:
            filtered_data = filtered_data.drop('temporal_order', axis=1)
        
        return filtered_data
    
    def handle_missing_values(self, train_df, test_df):
        """결측치 처리"""
        train_clean = self.safe_data_conversion(train_df)
        test_clean = self.safe_data_conversion(test_df)
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        # 생성된 피처들도 포함
        for col in train_clean.columns:
            if any(keyword in col for keyword in ['activity_score', 'payment_stability', 'ratio', 'product', 'encoded']):
                if col not in numeric_cols and train_clean[col].dtype in [np.number]:
                    numeric_cols.append(col)
        
        common_numeric = [col for col in numeric_cols if col in train_clean.columns and col in test_clean.columns]
        
        if common_numeric:
            if self.imputer is None:
                self.imputer = KNNImputer(n_neighbors=5, weights='distance')
                train_imputed = self.imputer.fit_transform(train_clean[common_numeric])
            else:
                train_imputed = self.imputer.transform(train_clean[common_numeric])
            
            test_imputed = self.imputer.transform(test_clean[common_numeric])
            
            train_clean[common_numeric] = train_imputed
            test_clean[common_numeric] = test_imputed
        
        # 범주형 피처 결측치 처리
        categorical_cols = ['gender', 'subscription_type']
        for col in categorical_cols:
            if col in train_clean.columns and col in test_clean.columns:
                mode_value = train_clean[col].mode()[0] if not train_clean[col].mode().empty else 'Unknown'
                train_clean[col] = train_clean[col].fillna(mode_value)
                test_clean[col] = test_clean[col].fillna(mode_value)
        
        return train_clean, test_clean
    
    def remove_outliers(self, train_df):
        """이상치 제거"""
        train_clean = self.safe_data_conversion(train_df)
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        # 생성된 피처들도 포함
        for col in train_clean.columns:
            if any(keyword in col for keyword in ['activity_score', 'payment_stability', 'ratio', 'product']):
                if col not in numeric_cols and train_clean[col].dtype in [np.number]:
                    numeric_cols.append(col)
        
        available_cols = [col for col in numeric_cols if col in train_clean.columns]
        
        for col in available_cols:
            values = train_clean[col].dropna()
            
            if len(values) > 100:
                # IQR 방식으로 이상치 제거
                q25 = values.quantile(0.25)
                q75 = values.quantile(0.75)
                iqr = q75 - q25
                
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                
                train_clean[col] = np.clip(train_clean[col], lower_bound, upper_bound)
        
        return train_clean
    
    def select_features(self, train_df, max_features=15):
        """피처 선택"""
        if 'support_needs' not in train_df.columns:
            feature_cols = [col for col in train_df.columns if col not in ['ID']]
            return feature_cols[:min(max_features, len(feature_cols))]
        
        feature_cols = [col for col in train_df.columns if col not in ['ID', 'support_needs']]
        
        if len(feature_cols) <= max_features:
            self.selected_features = feature_cols
            return feature_cols
        
        X = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = np.clip(train_df['support_needs'], 0, 2)
        
        # 분산 기반 필터링
        variance_selector = VarianceThreshold(threshold=0.01)
        X_variance = variance_selector.fit_transform(X)
        variance_features = [feature_cols[i] for i, selected in enumerate(variance_selector.get_support()) if selected]
        
        if len(variance_features) <= max_features:
            self.selected_features = variance_features
            return variance_features
        
        # 상호 정보량 기반 선택
        X_var_df = pd.DataFrame(X_variance, columns=variance_features)
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
        mi_selector.fit(X_var_df, y)
        
        selected_features = [variance_features[i] for i, selected in enumerate(mi_selector.get_support()) if selected]
        
        self.selected_features = selected_features
        return selected_features
    
    def apply_scaling(self, train_df, test_df):
        """스케일링 적용"""
        train_clean = self.safe_data_conversion(train_df)
        test_clean = self.safe_data_conversion(test_df)
        
        numeric_cols = [col for col in train_clean.select_dtypes(include=[np.number]).columns 
                        if col not in ['ID', 'support_needs']]
        
        common_numeric = [col for col in numeric_cols if col in test_clean.columns]
        
        if not common_numeric:
            return train_clean, test_clean
        
        if self.scaler is None:
            self.scaler = RobustScaler(quantile_range=(10.0, 90.0))
            scaled_train = self.scaler.fit_transform(train_clean[common_numeric])
        else:
            scaled_train = self.scaler.transform(train_clean[common_numeric])
        
        scaled_test = self.scaler.transform(test_clean[common_numeric])
        
        train_clean[common_numeric] = scaled_train
        test_clean[common_numeric] = scaled_test
        
        return train_clean, test_clean
    
    def apply_smote_conservative(self, X_train, y_train):
        """보수적 SMOTE 적용"""
        try:
            class_counts = np.bincount(y_train.astype(int))
            min_class_count = np.min(class_counts[class_counts > 0])
            
            # 최소 클래스가 너무 적으면 SMOTE 적용 안함
            if min_class_count < 100:
                return X_train, y_train
            
            # 불균형이 심하지 않으면 SMOTE 적용 안함
            max_class_count = np.max(class_counts)
            imbalance_ratio = max_class_count / min_class_count
            
            if imbalance_ratio < 2.0:
                return X_train, y_train
            
            k_neighbors = min(5, min_class_count - 1)
            
            if k_neighbors <= 0:
                return X_train, y_train
            
            if self.smote is None:
                self.smote = SMOTE(
                    sampling_strategy='auto',
                    k_neighbors=k_neighbors,
                    random_state=42
                )
            
            X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
            
            # SMOTE 후에도 데이터가 너무 많으면 제한
            if len(X_resampled) > len(X_train) * 1.5:
                return X_train, y_train
            
            return X_resampled, y_resampled
            
        except Exception as e:
            return X_train, y_train
    
    def validate_data_quality(self, train_df, test_df):
        """데이터 품질 검증"""
        issues = []
        
        # 무한값 검사
        train_numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        test_numeric_cols = test_df.select_dtypes(include=[np.number]).columns
        
        train_inf = np.isinf(train_df[train_numeric_cols]).sum().sum()
        test_inf = np.isinf(test_df[test_numeric_cols]).sum().sum()
        
        if train_inf > 0:
            issues.append(f"train_infinite_values: {train_inf}")
        if test_inf > 0:
            issues.append(f"test_infinite_values: {test_inf}")
        
        # 결측값 검사
        train_nan = train_df[train_numeric_cols].isnull().sum().sum()
        test_nan = test_df[test_numeric_cols].isnull().sum().sum()
        
        if train_nan > 0:
            issues.append(f"train_missing_values: {train_nan}")
        if test_nan > 0:
            issues.append(f"test_missing_values: {test_nan}")
        
        # 타겟 변수 검사
        if 'support_needs' in train_df.columns:
            invalid_count = (~train_df['support_needs'].isin([0, 1, 2])).sum()
            if invalid_count > 0:
                issues.append(f"invalid_targets: {invalid_count}")
        
        return len(issues) == 0, issues
    
    def process_data(self, train_df, test_df):
        """전처리 파이프라인"""
        if train_df is None or test_df is None or train_df.empty or test_df.empty:
            return None, None
        
        # 시간적 필터링
        train_df = self.apply_temporal_filtering(train_df)
        
        # 결측치 처리
        train_df, test_df = self.handle_missing_values(train_df, test_df)
        
        # 이상치 제거
        train_df = self.remove_outliers(train_df)
        
        # 피처 선택
        if 'support_needs' in train_df.columns:
            selected_features = self.select_features(train_df, max_features=15)
            
            keep_cols_train = ['ID', 'support_needs'] + selected_features
            keep_cols_test = ['ID'] + [f for f in selected_features if f in test_df.columns]
            
            available_train_cols = [col for col in keep_cols_train if col in train_df.columns]
            available_test_cols = [col for col in keep_cols_test if col in test_df.columns]
            
            train_df = train_df[available_train_cols]
            test_df = test_df[available_test_cols]
        
        # 스케일링
        train_df, test_df = self.apply_scaling(train_df, test_df)
        
        # 데이터 품질 검증
        quality_ok, issues = self.validate_data_quality(train_df, test_df)
        
        return train_df, test_df
    
    def prepare_temporal_split(self, train_df, test_df, val_size=0.20):
        """시간적 분할"""
        if train_df is None or test_df is None:
            return None, None, None, None, None, None
            
        if 'support_needs' not in train_df.columns:
            feature_cols = [col for col in train_df.columns if col not in ['ID']]
            common_features = [col for col in feature_cols if col in test_df.columns]
            
            if not common_features:
                raise ValueError("공통 피처 없음")
                
            X_train = train_df[common_features].fillna(0).replace([np.inf, -np.inf], 0)
            X_test = test_df[common_features].fillna(0).replace([np.inf, -np.inf], 0)
            test_ids = test_df['ID'] if 'ID' in test_df.columns else pd.Series(range(len(test_df)))
            
            X_train_split, X_val_split, y_dummy, y_val_dummy = train_test_split(
                X_train, np.zeros(len(X_train)), test_size=val_size, random_state=42
            )
            
            return X_train_split, X_val_split, y_dummy[:len(X_train_split)], y_val_dummy, X_test, test_ids
        
        # 시간적 분할을 위한 ID 기반 정렬
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
        
        train_df_temp = train_df.copy()
        train_df_temp['temporal_order'] = temporal_ids
        
        # 시간 순으로 정렬
        train_df_temp = train_df_temp.sort_values('temporal_order')
        
        # 공통 피처 추출
        train_cols = set(train_df_temp.columns)
        test_cols = set(test_df.columns)
        common_features = list((train_cols & test_cols) - {'ID', 'support_needs', 'temporal_order'})
        
        if not common_features:
            raise ValueError("공통 피처 없음")
        
        common_features = sorted(common_features)
        
        # 시간적 분할 (앞의 80%를 훈련, 뒤의 20%를 검증)
        split_point = int(len(train_df_temp) * (1 - val_size))
        
        train_part = train_df_temp.iloc[:split_point]
        val_part = train_df_temp.iloc[split_point:]
        
        # 피처와 타겟 분리
        X_train = train_part[common_features].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_part['support_needs']
        X_val = val_part[common_features].fillna(0).replace([np.inf, -np.inf], 0)
        y_val = val_part['support_needs']
        
        X_test = test_df[common_features].fillna(0).replace([np.inf, -np.inf], 0)
        test_ids = test_df['ID'] if 'ID' in test_df.columns else pd.Series(range(len(test_df)))
        
        # SMOTE 보수적 적용
        X_train_resampled, y_train_resampled = self.apply_smote_conservative(X_train, y_train)
        
        return X_train_resampled, X_val, y_train_resampled, y_val, X_test, test_ids

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