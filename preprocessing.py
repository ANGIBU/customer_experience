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
        
        # 범주형 컬럼 먼저 처리
        categorical_cols = ['gender', 'subscription_type']
        for col in categorical_cols:
            if col in df_clean.columns:
                # 범주형 데이터는 문자열로 유지
                df_clean[col] = df_clean[col].astype(str).fillna('Unknown')
        
        # 숫자형 컬럼만 안전하게 변환
        for col in df_clean.columns:
            if col not in categorical_cols and col not in ['ID']:
                try:
                    # 숫자형으로 변환 가능한지 확인
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                except:
                    # 변환할 수 없으면 문자열로 유지
                    df_clean[col] = df_clean[col].astype(str).fillna('Unknown')
        
        return df_clean
        
    def apply_temporal_filtering(self, train_df):
        """시간적 필터링 (temporal 피처 생성 없이)"""
        if 'ID' not in train_df.columns:
            return train_df
        
        try:
            # ID에서 시간 정보 추출 (임시로만 사용)
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
            
            # 60% 분위수를 기준으로 안전한 데이터만 사용
            if temporal_ids and len(temporal_ids) > 100:
                threshold = np.percentile(temporal_ids, 60)
                safe_indices = [i for i, tid in enumerate(temporal_ids) if tid <= threshold]
                
                if len(safe_indices) > len(temporal_ids) * 0.4:  # 최소 40% 데이터 보장
                    filtered_data = train_df.iloc[safe_indices].copy()
                else:
                    filtered_data = train_df.copy()
            else:
                filtered_data = train_df.copy()
            
            return filtered_data
            
        except Exception as e:
            print(f"시간적 필터링 오류: {e}")
            return train_df
    
    def handle_missing_values(self, train_df, test_df):
        """결측치 처리"""
        train_clean = self.safe_data_conversion(train_df)
        test_clean = self.safe_data_conversion(test_df)
        
        # 숫자형 피처만 선택해서 처리
        numeric_cols = []
        for col in train_clean.columns:
            if col not in ['ID', 'gender', 'subscription_type', 'support_needs']:
                try:
                    # 숫자형 변환이 가능한지 확인
                    pd.to_numeric(train_clean[col], errors='raise')
                    numeric_cols.append(col)
                except:
                    # 숫자형이 아니면 제외
                    continue
        
        # 공통 숫자형 피처만 처리
        common_numeric = [col for col in numeric_cols if col in train_clean.columns and col in test_clean.columns]
        
        if common_numeric:
            try:
                if self.imputer is None:
                    self.imputer = KNNImputer(n_neighbors=5, weights='distance')
                    train_imputed = self.imputer.fit_transform(train_clean[common_numeric])
                else:
                    train_imputed = self.imputer.transform(train_clean[common_numeric])
                
                test_imputed = self.imputer.transform(test_clean[common_numeric])
                
                train_clean[common_numeric] = train_imputed
                test_clean[common_numeric] = test_imputed
            except Exception as e:
                print(f"KNN imputer 오류, 기본값으로 대체: {e}")
                # KNN imputer 실패 시 중앙값으로 대체
                for col in common_numeric:
                    median_val = train_clean[col].median()
                    train_clean[col] = train_clean[col].fillna(median_val)
                    test_clean[col] = test_clean[col].fillna(median_val)
        
        # 범주형 피처 결측치 처리
        categorical_cols = ['gender', 'subscription_type']
        for col in categorical_cols:
            if col in train_clean.columns and col in test_clean.columns:
                try:
                    mode_value = train_clean[col].mode()[0] if not train_clean[col].mode().empty else 'Unknown'
                    train_clean[col] = train_clean[col].fillna(mode_value)
                    test_clean[col] = test_clean[col].fillna(mode_value)
                except:
                    train_clean[col] = train_clean[col].fillna('Unknown')
                    test_clean[col] = test_clean[col].fillna('Unknown')
        
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
    
    def prepare_temporal_split(self, train_df, test_df, val_size=0.25):
        """시간적 분할 (안전한 버전)"""
        if train_df is None or test_df is None:
            return None, None, None, None, None, None
            
        try:
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

            # 공통 피처 추출 (안전하게)
            train_cols = set(train_df.columns)
            test_cols = set(test_df.columns)
            common_features = list((train_cols & test_cols) - {'ID', 'support_needs'})
            
            if not common_features:
                raise ValueError("공통 피처 없음")
            
            common_features = sorted(common_features)
            
            # 범주형 데이터 안전 처리
            safe_train = train_df.copy()
            categorical_cols = ['gender', 'subscription_type']
            
            for col in categorical_cols:
                if col in safe_train.columns:
                    if col in common_features:
                        # 범주형 데이터를 숫자로 안전하게 변환
                        le_mapping = {'M': 0, 'F': 1, 'Unknown': 2}
                        if col == 'gender':
                            safe_train[col] = safe_train[col].map(le_mapping).fillna(2)
                        else:
                            # subscription_type 처리
                            unique_vals = safe_train[col].unique()
                            mapping = {val: i for i, val in enumerate(unique_vals)}
                            safe_train[col] = safe_train[col].map(mapping).fillna(0)
            
            # 시간적 분할 시도
            try:
                # ID에서 시간 정보 추출
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
                    # 시간 순으로 정렬
                    safe_train['temp_order'] = temporal_ids
                    safe_train = safe_train.sort_values('temp_order')
                    
                    # 시간적 분할
                    split_point = int(len(safe_train) * (1 - val_size))
                    
                    train_part = safe_train.iloc[:split_point]
                    val_part = safe_train.iloc[split_point:]
                    
                    safe_train = safe_train.drop('temp_order', axis=1)
                    train_part = train_part.drop('temp_order', axis=1)
                    val_part = val_part.drop('temp_order', axis=1)
                else:
                    # 랜덤 분할로 대체
                    train_part, val_part = train_test_split(
                        safe_train, test_size=val_size, random_state=42, stratify=safe_train['support_needs']
                    )
                    
            except Exception as e:
                print(f"시간적 분할 실패, 랜덤 분할 사용: {e}")
                # 랜덤 분할로 대체
                train_part, val_part = train_test_split(
                    safe_train, test_size=val_size, random_state=42, stratify=safe_train['support_needs']
                )
            
            # 테스트 데이터도 동일하게 처리
            safe_test = test_df.copy()
            for col in categorical_cols:
                if col in safe_test.columns:
                    if col in common_features:
                        if col == 'gender':
                            safe_test[col] = safe_test[col].map(le_mapping).fillna(2)
                        else:
                            unique_vals = train_df[col].unique() if col in train_df.columns else safe_test[col].unique()
                            mapping = {val: i for i, val in enumerate(unique_vals)}
                            safe_test[col] = safe_test[col].map(mapping).fillna(0)
            
            # 피처와 타겟 분리
            X_train = train_part[common_features].fillna(0).replace([np.inf, -np.inf], 0)
            y_train = train_part['support_needs']
            X_val = val_part[common_features].fillna(0).replace([np.inf, -np.inf], 0)
            y_val = val_part['support_needs']
            
            X_test = safe_test[common_features].fillna(0).replace([np.inf, -np.inf], 0)
            test_ids = safe_test['ID'] if 'ID' in safe_test.columns else pd.Series(range(len(safe_test)))
            
            # SMOTE 보수적 적용
            X_train_resampled, y_train_resampled = self.apply_smote_conservative(X_train, y_train)
            
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