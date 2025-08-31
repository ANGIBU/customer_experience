# data_analysis.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
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
            self.temporal_cutoff = max(test_id_nums)
            print(f"시간적 분할점: {self.temporal_cutoff}")
            
            # 누수 위험 계산
            overlap_count = sum(1 for tid in train_id_nums if tid <= self.temporal_cutoff)
            overlap_ratio = overlap_count / len(train_id_nums)
            
            print(f"시간적 겹침 비율: {overlap_ratio:.3f}")
            
            return {
                'train_range': train_range,
                'test_range': test_range,
                'temporal_cutoff': self.temporal_cutoff,
                'overlap_ratio': overlap_ratio,
                'safe_train_indices': [i for i, tid in enumerate(train_id_nums) if tid > self.temporal_cutoff]
            }
            
        return {}
    
    def detect_target_leakage(self):
        """타겟 누수 탐지"""
        print("타겟 누수 탐지")
        
        leakage_features = {}
        
        if 'after_interaction' in self.train_df.columns and 'support_needs' in self.train_df.columns:
            # 클래스별 평균 계산
            class_means = {}
            for cls in [0, 1, 2]:
                class_data = self.train_df[self.train_df['support_needs'] == cls]['after_interaction'].dropna()
                if len(class_data) > 0:
                    class_means[cls] = class_data.mean()
            
            # 상관관계 계산
            correlation = self.train_df[['after_interaction', 'support_needs']].corr().iloc[0, 1]
            
            # 상호정보량 계산
            after_clean = self.train_df['after_interaction'].fillna(0)
            target_clean = self.train_df['support_needs']
            
            mi_score = mutual_info_classif(after_clean.values.reshape(-1, 1), target_clean, random_state=42)[0]
            
            # 클래스 분리도 측정
            if len(class_means) >= 2:
                mean_values = list(class_means.values())
                max_diff = max(mean_values) - min(mean_values)
                
                leakage_features['after_interaction'] = {
                    'correlation': correlation,
                    'mutual_info': mi_score,
                    'class_separation': max_diff,
                    'is_leakage': abs(correlation) > 0.1 or mi_score > 0.5 or max_diff > 2.0
                }
                
                print(f"after_interaction 상관관계: {correlation:.4f}")
                print(f"after_interaction 상호정보량: {mi_score:.4f}")
                print(f"after_interaction 클래스 분리도: {max_diff:.4f}")
                
                if leakage_features['after_interaction']['is_leakage']:
                    print("경고: after_interaction 피처 누수 위험 탐지")
        
        return leakage_features
    
    def analyze_feature_distributions(self):
        """피처 분포 분석"""
        print("피처 분포 분석")
        
        distribution_analysis = {}
        numeric_features = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        for feature in numeric_features:
            if feature in self.train_df.columns and feature in self.test_df.columns:
                train_vals = self.train_df[feature].dropna()
                test_vals = self.test_df[feature].dropna()
                
                # KS 테스트
                ks_stat, ks_p = ks_2samp(train_vals, test_vals)
                
                # 기본 통계량
                train_stats = {
                    'mean': train_vals.mean(),
                    'std': train_vals.std(),
                    'skew': train_vals.skew(),
                    'kurtosis': train_vals.kurtosis()
                }
                
                test_stats = {
                    'mean': test_vals.mean(),
                    'std': test_vals.std(),
                    'skew': test_vals.skew(),
                    'kurtosis': test_vals.kurtosis()
                }
                
                # PSI 계산
                psi_score = self.calculate_psi(train_vals, test_vals)
                
                distribution_analysis[feature] = {
                    'train_stats': train_stats,
                    'test_stats': test_stats,
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p,
                    'psi_score': psi_score,
                    'distribution_shift': ks_stat > 0.1 or psi_score > 0.2
                }
                
                if distribution_analysis[feature]['distribution_shift']:
                    print(f"{feature} 분포 이동 탐지: KS={ks_stat:.3f}, PSI={psi_score:.3f}")
        
        return distribution_analysis
    
    def calculate_psi(self, train_data, test_data, bins=10):
        """PSI 계산"""
        try:
            # 동일한 구간으로 분할
            min_val = min(train_data.min(), test_data.min())
            max_val = max(train_data.max(), test_data.max())
            
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            train_hist, _ = np.histogram(train_data, bins=bin_edges)
            test_hist, _ = np.histogram(test_data, bins=bin_edges)
            
            # 비율로 변환
            train_pct = train_hist / len(train_data)
            test_pct = test_hist / len(test_data)
            
            # PSI 계산
            psi = 0
            for i in range(len(train_pct)):
                if train_pct[i] > 0 and test_pct[i] > 0:
                    psi += (test_pct[i] - train_pct[i]) * np.log(test_pct[i] / train_pct[i])
            
            return psi
            
        except Exception:
            return 0.0
    
    def analyze_class_distribution(self):
        """클래스 분포 분석"""
        print("클래스 분포 분석")
        
        if 'support_needs' not in self.train_df.columns:
            return {}
        
        target_counts = self.train_df['support_needs'].value_counts().sort_index()
        total = len(self.train_df)
        
        distribution_info = {}
        for cls, count in target_counts.items():
            pct = count / total
            distribution_info[cls] = {'count': count, 'percentage': pct}
            print(f"클래스 {cls}: {count:,}개 ({pct:.3f})")
        
        # 불균형 비율
        max_count = target_counts.max()
        min_count = target_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else 0
        
        print(f"불균형 비율: {imbalance_ratio:.2f}")
        
        return {
            'distribution': distribution_info,
            'imbalance_ratio': imbalance_ratio,
            'needs_rebalancing': imbalance_ratio > 2.0
        }
    
    def analyze_feature_correlations(self):
        """피처 상관관계 분석"""
        print("피처 상관관계 분석")
        
        numeric_features = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_features = [f for f in numeric_features if f in self.train_df.columns]
        
        if not available_features or 'support_needs' not in self.train_df.columns:
            return {}
        
        # 피처 간 상관관계
        feature_corr = self.train_df[available_features].corr()
        
        # 높은 상관관계 쌍
        high_corr_pairs = []
        for i in range(len(available_features)):
            for j in range(i+1, len(available_features)):
                corr_val = feature_corr.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append({
                        'feature1': available_features[i],
                        'feature2': available_features[j],
                        'correlation': corr_val
                    })
        
        # 타겟과의 상관관계
        target_corr = {}
        for feature in available_features:
            corr = self.train_df[[feature, 'support_needs']].corr().iloc[0, 1]
            target_corr[feature] = corr
        
        if high_corr_pairs:
            print("높은 상관관계:")
            for pair in high_corr_pairs:
                print(f"{pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}")
        
        return {
            'feature_correlations': feature_corr.to_dict(),
            'high_corr_pairs': high_corr_pairs,
            'target_correlations': target_corr
        }
    
    def compute_feature_importance(self):
        """피처 중요도 계산"""
        print("피처 중요도 계산")
        
        if 'support_needs' not in self.train_df.columns:
            return {}
        
        # 수치형 피처
        numeric_features = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_numeric = [f for f in numeric_features if f in self.train_df.columns]
        
        # 범주형 피처 인코딩
        categorical_features = ['gender', 'subscription_type']
        train_encoded = self.train_df.copy()
        
        for col in categorical_features:
            if col in self.train_df.columns:
                le = LabelEncoder()
                train_encoded[col] = le.fit_transform(train_encoded[col].fillna('Unknown'))
                available_numeric.append(col)
        
        if not available_numeric:
            return {}
        
        # 상호정보량 계산
        X = train_encoded[available_numeric].fillna(0)
        y = train_encoded['support_needs']
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        importance_dict = dict(zip(available_numeric, mi_scores))
        
        # 정렬
        importance_sorted = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        print("피처 중요도:")
        for feature, score in importance_sorted:
            print(f"{feature}: {score:.4f}")
        
        return importance_dict
    
    def validate_data_integrity(self):
        """데이터 무결성 검증"""
        print("데이터 무결성 검증")
        
        issues = []
        
        # ID 중복 확인
        train_duplicates = self.train_df['ID'].duplicated().sum()
        test_duplicates = self.test_df['ID'].duplicated().sum()
        
        if train_duplicates > 0:
            issues.append(f"훈련 ID 중복: {train_duplicates}")
        if test_duplicates > 0:
            issues.append(f"테스트 ID 중복: {test_duplicates}")
        
        # 타겟 유효성
        if 'support_needs' in self.train_df.columns:
            invalid_targets = ~self.train_df['support_needs'].isin([0, 1, 2])
            invalid_count = invalid_targets.sum()
            if invalid_count > 0:
                issues.append(f"잘못된 타겟: {invalid_count}")
        
        # 결측치 패턴
        missing_info = {}
        for col in self.train_df.columns:
            if col != 'ID':
                missing_ratio = self.train_df[col].isnull().mean()
                if missing_ratio > 0.1:
                    missing_info[col] = missing_ratio
                    issues.append(f"{col} 결측률: {missing_ratio:.1%}")
        
        if issues:
            print("무결성 문제:")
            for issue in issues:
                print(f"- {issue}")
        else:
            print("무결성 검증 통과")
        
        return len(issues) == 0, issues, missing_info
    
    def analyze_categorical_distributions(self):
        """범주형 변수 분포 분석"""
        print("범주형 분포 분석")
        
        categorical_cols = ['gender', 'subscription_type']
        categorical_analysis = {}
        
        for col in categorical_cols:
            if col in self.train_df.columns and col in self.test_df.columns:
                train_dist = self.train_df[col].value_counts(normalize=True)
                test_dist = self.test_df[col].value_counts(normalize=True)
                
                # 공통 카테고리
                common_cats = set(train_dist.index) & set(test_dist.index)
                
                if len(common_cats) > 1:
                    # 카이제곱 검정을 위한 데이터 준비
                    train_counts = []
                    test_counts = []
                    
                    for cat in sorted(common_cats):
                        train_counts.append(self.train_df[col].eq(cat).sum())
                        test_counts.append(self.test_df[col].eq(cat).sum())
                    
                    try:
                        chi2_stat, chi2_p = chi2_contingency([train_counts, test_counts])[:2]
                    except:
                        chi2_stat, chi2_p = 0, 1
                    
                    categorical_analysis[col] = {
                        'train_dist': train_dist.to_dict(),
                        'test_dist': test_dist.to_dict(),
                        'chi2_statistic': chi2_stat,
                        'chi2_p_value': chi2_p,
                        'distribution_shift': chi2_p < 0.01
                    }
                    
                    if chi2_p < 0.01:
                        print(f"{col} 분포 이동: p={chi2_p:.4f}")
        
        return categorical_analysis
    
    def compute_data_quality_score(self):
        """데이터 품질 점수 계산"""
        print("데이터 품질 점수 계산")
        
        quality_metrics = {}
        
        # 완전성 점수
        completeness_scores = []
        for col in self.train_df.columns:
            if col != 'ID':
                completeness = 1 - self.train_df[col].isnull().mean()
                completeness_scores.append(completeness)
        
        quality_metrics['completeness'] = np.mean(completeness_scores)
        
        # 일관성 점수
        consistency_scores = []
        numeric_features = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        for feature in numeric_features:
            if feature in self.train_df.columns and feature in self.test_df.columns:
                train_vals = self.train_df[feature].dropna()
                test_vals = self.test_df[feature].dropna()
                
                if len(train_vals) > 0 and len(test_vals) > 0:
                    ks_stat, _ = ks_2samp(train_vals, test_vals)
                    consistency = 1 - min(ks_stat, 1.0)
                    consistency_scores.append(consistency)
        
        quality_metrics['consistency'] = np.mean(consistency_scores) if consistency_scores else 0
        
        # 유효성 점수
        validity_score = 1.0
        if 'support_needs' in self.train_df.columns:
            invalid_ratio = ~self.train_df['support_needs'].isin([0, 1, 2]).mean()
            validity_score = 1 - invalid_ratio
        
        quality_metrics['validity'] = validity_score
        
        # 종합 점수
        overall_quality = (
            quality_metrics['completeness'] * 0.4 +
            quality_metrics['consistency'] * 0.4 +
            quality_metrics['validity'] * 0.2
        )
        
        quality_metrics['overall'] = overall_quality
        
        print(f"완전성: {quality_metrics['completeness']:.3f}")
        print(f"일관성: {quality_metrics['consistency']:.3f}")
        print(f"유효성: {quality_metrics['validity']:.3f}")
        print(f"종합 품질: {overall_quality:.3f}")
        
        return quality_metrics
    
    def run_analysis(self):
        """전체 분석 실행"""
        print("데이터 분석 시작")
        print("=" * 40)
        
        if self.load_data() is None:
            return {}
        
        # 무결성 검증
        integrity_ok, integrity_issues, missing_info = self.validate_data_integrity()
        self.analysis_results['integrity'] = {
            'passed': integrity_ok,
            'issues': integrity_issues,
            'missing_info': missing_info
        }
        
        # 시간적 구조
        temporal_info = self.analyze_temporal_structure()
        self.analysis_results['temporal'] = temporal_info
        
        # 타겟 누수 탐지
        leakage_info = self.detect_target_leakage()
        self.analysis_results['leakage'] = leakage_info
        
        # 분포 분석
        distribution_info = self.analyze_feature_distributions()
        self.analysis_results['distributions'] = distribution_info
        
        # 범주형 분포
        categorical_info = self.analyze_categorical_distributions()
        self.analysis_results['categorical'] = categorical_info
        
        # 클래스 분포
        class_info = self.analyze_class_distribution()
        self.analysis_results['class_distribution'] = class_info
        
        # 상관관계
        correlation_info = self.analyze_feature_correlations()
        self.analysis_results['correlations'] = correlation_info
        
        # 피처 중요도
        importance_info = self.compute_feature_importance()
        self.analysis_results['feature_importance'] = importance_info
        
        # 데이터 품질
        quality_info = self.compute_data_quality_score()
        self.analysis_results['data_quality'] = quality_info
        
        print("데이터 분석 완료")
        return self.analysis_results

def main():
    analyzer = DataAnalyzer()
    results = analyzer.run_analysis()
    return analyzer, results

if __name__ == "__main__":
    main()