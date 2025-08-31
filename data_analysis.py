# data_analysis.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.analysis_results = {}
        self.temporal_cutoff = None
        
    def load_data(self):
        """데이터 로드"""
        try:
            self.train_df = pd.read_csv('train.csv')
            self.test_df = pd.read_csv('test.csv')
            
            print(f"훈련 데이터: {self.train_df.shape}")
            print(f"테스트 데이터: {self.test_df.shape}")
            
            return self.train_df, self.test_df
            
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            return None, None
    
    def analyze_temporal_structure(self):
        """시간적 구조 분석"""
        print("시간적 구조 분석")
        
        def extract_id_numbers(id_series):
            numbers = []
            for id_val in id_series:
                try:
                    if '_' in str(id_val):
                        num = int(str(id_val).split('_')[1])
                        numbers.append(num)
                except:
                    continue
            return numbers
        
        train_id_nums = extract_id_numbers(self.train_df['ID'])
        test_id_nums = extract_id_numbers(self.test_df['ID'])
        
        if train_id_nums and test_id_nums:
            train_range = [min(train_id_nums), max(train_id_nums)]
            test_range = [min(test_id_nums), max(test_id_nums)]
            
            print(f"훈련 ID 범위: {train_range}")
            print(f"테스트 ID 범위: {test_range}")
            
            # 시간적 분할점 계산
            all_ids = sorted(train_id_nums + test_id_nums)
            self.temporal_cutoff = max(test_id_nums)
            
            print(f"시간적 분할점: {self.temporal_cutoff}")
            
            # 누수 위험 계산
            temporal_overlap = any(tid <= self.temporal_cutoff for tid in train_id_nums)
            
            if temporal_overlap:
                overlap_count = sum(1 for tid in train_id_nums if tid <= self.temporal_cutoff)
                overlap_ratio = overlap_count / len(train_id_nums)
                print(f"시간적 겹침 비율: {overlap_ratio:.3f}")
                
                return {
                    'train_range': train_range,
                    'test_range': test_range,
                    'temporal_cutoff': self.temporal_cutoff,
                    'temporal_overlap': temporal_overlap,
                    'overlap_ratio': overlap_ratio,
                    'safe_train_mask': [tid > self.temporal_cutoff for tid in train_id_nums]
                }
            
        return {}
    
    def analyze_feature_patterns(self):
        """피처 패턴 분석"""
        print("피처 패턴 분석")
        
        if 'support_needs' not in self.train_df.columns:
            return {}
        
        patterns = {}
        
        # 수치형 피처 분석
        numeric_features = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        for feature in numeric_features:
            if feature in self.train_df.columns:
                feature_data = self.train_df[feature].dropna()
                
                patterns[feature] = {
                    'mean': feature_data.mean(),
                    'std': feature_data.std(),
                    'median': feature_data.median(),
                    'q25': feature_data.quantile(0.25),
                    'q75': feature_data.quantile(0.75),
                    'skewness': feature_data.skew(),
                    'kurtosis': feature_data.kurtosis(),
                    'outlier_ratio': self.calculate_outlier_ratio(feature_data)
                }
        
        # 범주형 피처 분석
        categorical_features = ['gender', 'subscription_type']
        
        for feature in categorical_features:
            if feature in self.train_df.columns:
                value_counts = self.train_df[feature].value_counts()
                patterns[feature] = {
                    'categories': value_counts.index.tolist(),
                    'frequencies': value_counts.values.tolist(),
                    'entropy': self.calculate_entropy(value_counts.values)
                }
        
        return patterns
    
    def calculate_outlier_ratio(self, data):
        """이상치 비율 계산"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((data < lower) | (data > upper)).sum()
            return outliers / len(data)
        
        return 0.0
    
    def calculate_entropy(self, frequencies):
        """엔트로피 계산"""
        frequencies = np.array(frequencies)
        probabilities = frequencies / frequencies.sum()
        probabilities = probabilities[probabilities > 0]
        
        if len(probabilities) <= 1:
            return 0.0
        
        return -np.sum(probabilities * np.log2(probabilities))
    
    def analyze_class_relationships(self):
        """클래스 관계 분석"""
        print("클래스 관계 분석")
        
        if 'support_needs' not in self.train_df.columns:
            return {}
        
        class_relationships = {}
        
        # 수치형 피처별 클래스 분리도
        numeric_features = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        for feature in numeric_features:
            if feature in self.train_df.columns:
                class_groups = []
                class_stats = {}
                
                for cls in [0, 1, 2]:
                    class_data = self.train_df[self.train_df['support_needs'] == cls][feature].dropna()
                    class_groups.append(class_data)
                    
                    if len(class_data) > 0:
                        class_stats[cls] = {
                            'mean': class_data.mean(),
                            'std': class_data.std(),
                            'median': class_data.median()
                        }
                
                # ANOVA F-통계량 계산
                if all(len(group) > 10 for group in class_groups):
                    try:
                        f_stat, p_val = stats.f_oneway(*class_groups)
                        class_relationships[feature] = {
                            'f_statistic': f_stat,
                            'p_value': p_val,
                            'class_stats': class_stats,
                            'separability': p_val < 0.01
                        }
                    except:
                        pass
        
        return class_relationships
    
    def detect_leakage_features(self):
        """누수 피처 탐지"""
        print("누수 피처 탐지")
        
        if 'support_needs' not in self.train_df.columns:
            return {}
        
        leakage_features = {}
        
        # after_interaction 피처 분석
        if 'after_interaction' in self.train_df.columns:
            class_means = {}
            
            for cls in [0, 1, 2]:
                class_data = self.train_df[self.train_df['support_needs'] == cls]['after_interaction'].dropna()
                if len(class_data) > 0:
                    class_means[cls] = class_data.mean()
            
            if len(class_means) >= 2:
                mean_values = list(class_means.values())
                max_diff = max(mean_values) - min(mean_values)
                
                # 상관관계 계산
                correlation = self.train_df[['after_interaction', 'support_needs']].corr().iloc[0, 1]
                
                leakage_features['after_interaction'] = {
                    'class_means': class_means,
                    'max_difference': max_diff,
                    'correlation': correlation,
                    'leakage_risk': max_diff > 1.0 or abs(correlation) > 0.05
                }
                
                if max_diff > 1.0:
                    print(f"after_interaction 누수 위험: 클래스 간 차이 {max_diff:.3f}")
        
        return leakage_features
    
    def analyze_feature_importance_safe(self):
        """안전한 피처 중요도 분석"""
        print("피처 중요도 분석")
        
        if 'support_needs' not in self.train_df.columns:
            return {}
        
        try:
            # after_interaction 제외한 피처들만 사용
            safe_features = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
            
            # 범주형 변수 인코딩
            train_encoded = self.train_df.copy()
            le = LabelEncoder()
            
            categorical_cols = ['gender', 'subscription_type']
            for col in categorical_cols:
                if col in train_encoded.columns:
                    train_encoded[col] = le.fit_transform(train_encoded[col].fillna('Unknown'))
                    safe_features.append(col)
            
            # 사용 가능한 피처만 선택
            available_features = [f for f in safe_features if f in train_encoded.columns]
            
            if not available_features:
                return {}
            
            X = train_encoded[available_features].fillna(0)
            y = train_encoded['support_needs']
            
            # 상호정보량 계산
            mi_scores = mutual_info_classif(X, y, random_state=42)
            importance_dict = dict(zip(available_features, mi_scores))
            
            # 정렬
            importance_sorted = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            print("피처 중요도 (상호정보량):")
            for feature, score in importance_sorted:
                print(f"  {feature}: {score:.4f}")
            
            return importance_dict
            
        except Exception as e:
            print(f"중요도 분석 오류: {e}")
            return {}
    
    def validate_data_integrity(self):
        """데이터 무결성 검증"""
        print("데이터 무결성 검증")
        
        integrity_issues = []
        
        # ID 중복 확인
        train_id_duplicates = self.train_df['ID'].duplicated().sum()
        test_id_duplicates = self.test_df['ID'].duplicated().sum()
        
        if train_id_duplicates > 0:
            integrity_issues.append(f"훈련 데이터 ID 중복: {train_id_duplicates}개")
        
        if test_id_duplicates > 0:
            integrity_issues.append(f"테스트 데이터 ID 중복: {test_id_duplicates}개")
        
        # 타겟 변수 유효성 확인
        if 'support_needs' in self.train_df.columns:
            invalid_targets = self.train_df['support_needs'].isin([0, 1, 2]).sum()
            total_targets = len(self.train_df)
            
            if invalid_targets != total_targets:
                integrity_issues.append(f"잘못된 타겟 값: {total_targets - invalid_targets}개")
        
        # 결측치 패턴 분석
        missing_analysis = {}
        for col in self.train_df.columns:
            if col != 'ID':
                missing_count = self.train_df[col].isnull().sum()
                missing_ratio = missing_count / len(self.train_df)
                
                if missing_ratio > 0.1:
                    missing_analysis[col] = missing_ratio
                    integrity_issues.append(f"{col} 높은 결측률: {missing_ratio:.1%}")
        
        self.analysis_results['integrity'] = {
            'issues': integrity_issues,
            'missing_analysis': missing_analysis
        }
        
        return len(integrity_issues) == 0
    
    def create_temporal_split_strategy(self):
        """시간적 분할 전략 생성"""
        print("시간적 분할 전략 생성")
        
        temporal_analysis = self.analyze_temporal_structure()
        
        if not temporal_analysis:
            return None
        
        # 안전한 훈련 데이터 마스크 생성
        train_id_nums = []
        for id_val in self.train_df['ID']:
            try:
                if '_' in str(id_val):
                    num = int(str(id_val).split('_')[1])
                    train_id_nums.append(num)
                else:
                    train_id_nums.append(99999)
            except:
                train_id_nums.append(99999)
        
        # 시간적 누수 방지를 위한 분할
        safe_mask = [tid > self.temporal_cutoff for tid in train_id_nums]
        
        split_strategy = {
            'temporal_cutoff': self.temporal_cutoff,
            'safe_train_indices': [i for i, safe in enumerate(safe_mask) if safe],
            'leakage_indices': [i for i, safe in enumerate(safe_mask) if not safe],
            'safe_ratio': sum(safe_mask) / len(safe_mask)
        }
        
        print(f"안전한 훈련 데이터 비율: {split_strategy['safe_ratio']:.3f}")
        
        if split_strategy['safe_ratio'] < 0.3:
            print("경고: 안전한 훈련 데이터 부족")
        
        return split_strategy
    
    def analyze_target_distribution(self):
        """타겟 분포 분석"""
        print("타겟 분포 분석")
        
        if 'support_needs' not in self.train_df.columns:
            return None
        
        target_counts = self.train_df['support_needs'].value_counts().sort_index()
        total = len(self.train_df)
        
        print("클래스 분포:")
        distribution_info = {}
        
        for cls, count in target_counts.items():
            pct = count / total * 100
            distribution_info[cls] = {'count': count, 'percentage': pct}
            print(f"  클래스 {cls}: {count:,}개 ({pct:.1f}%)")
        
        # 불균형 비율 계산
        max_count = target_counts.max()
        min_count = target_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else 0
        
        print(f"불균형 비율: {imbalance_ratio:.2f}")
        
        self.analysis_results['target'] = {
            'distribution': distribution_info,
            'imbalance_ratio': imbalance_ratio,
            'class_weights_needed': imbalance_ratio > 2.0
        }
        
        return target_counts, imbalance_ratio
    
    def analyze_correlation_structure(self):
        """상관관계 구조 분석"""
        print("상관관계 구조 분석")
        
        if 'support_needs' not in self.train_df.columns:
            return {}
        
        # 수치형 피처만 사용 (after_interaction 제외)
        safe_numeric = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_numeric = [f for f in safe_numeric if f in self.train_df.columns]
        
        correlation_analysis = {}
        
        if available_numeric:
            # 피처 간 상관관계
            feature_corr_matrix = self.train_df[available_numeric].corr()
            
            # 높은 상관관계 피처 쌍 찾기
            high_corr_pairs = []
            for i in range(len(available_numeric)):
                for j in range(i+1, len(available_numeric)):
                    corr_val = feature_corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append({
                            'feature1': available_numeric[i],
                            'feature2': available_numeric[j],
                            'correlation': corr_val
                        })
            
            # 타겟과의 상관관계
            target_correlations = {}
            for feature in available_numeric:
                corr = self.train_df[[feature, 'support_needs']].corr().iloc[0, 1]
                target_correlations[feature] = corr
            
            correlation_analysis = {
                'feature_correlations': feature_corr_matrix.to_dict(),
                'high_corr_pairs': high_corr_pairs,
                'target_correlations': target_correlations
            }
            
            if high_corr_pairs:
                print("높은 상관관계 피처 쌍:")
                for pair in high_corr_pairs:
                    print(f"  {pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}")
        
        return correlation_analysis
    
    def run_analysis(self):
        """전체 분석 실행"""
        print("데이터 분석 시작")
        print("=" * 40)
        
        if self.load_data() is None:
            print("데이터 로드 실패")
            return {}
        
        # 기본 분석
        self.validate_data_integrity()
        
        # 시간적 구조 분석
        temporal_info = self.analyze_temporal_structure()
        self.analysis_results['temporal'] = temporal_info
        
        # 타겟 분석
        target_info = self.analyze_target_distribution()
        
        # 피처 패턴 분석
        pattern_info = self.analyze_feature_patterns()
        self.analysis_results['patterns'] = pattern_info
        
        # 클래스 관계 분석
        class_info = self.analyze_class_relationships()
        self.analysis_results['class_relationships'] = class_info
        
        # 상관관계 분석
        correlation_info = self.analyze_correlation_structure()
        self.analysis_results['correlations'] = correlation_info
        
        # 누수 탐지
        leakage_info = self.detect_leakage_features()
        self.analysis_results['leakage'] = leakage_info
        
        # 시간적 분할 전략
        split_strategy = self.create_temporal_split_strategy()
        self.analysis_results['split_strategy'] = split_strategy
        
        print("데이터 분석 완료")
        return self.analysis_results

def main():
    analyzer = DataAnalyzer()
    results = analyzer.run_analysis()
    return analyzer, results

if __name__ == "__main__":
    main()