# data_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """데이터 분석 클래스"""
    
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.analysis_results = {}
        
    def load_data(self):
        """데이터 로드"""
        self.train_df = pd.read_csv('train.csv')
        self.test_df = pd.read_csv('test.csv')
        
        print(f"훈련 데이터: {self.train_df.shape}")
        print(f"테스트 데이터: {self.test_df.shape}")
        
        return self.train_df, self.test_df
    
    def analyze_target_distribution(self):
        """타겟 분포 분석"""
        print("\n=== 타겟 분포 세밀 분석 ===")
        
        target_counts = self.train_df['support_needs'].value_counts().sort_index()
        total = len(self.train_df)
        
        print("클래스 분포:")
        for cls, count in target_counts.items():
            pct = count / total * 100
            print(f"  클래스 {cls}: {count:,}개 ({pct:.2f}%)")
        
        imbalance_ratio = target_counts.max() / target_counts.min()
        print(f"불균형 비율: {imbalance_ratio:.2f}")
        
        self.analysis_results['target_distribution'] = {
            'counts': target_counts.to_dict(),
            'imbalance_ratio': imbalance_ratio
        }
        
        return target_counts
    
    def analyze_feature_distributions(self):
        """피처 분포 분석"""
        print("\n=== 피처 분포 분석 ===")
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 
                       'contract_length', 'after_interaction']
        
        distribution_stats = {}
        
        for col in numeric_cols:
            stats = {
                'mean': self.train_df[col].mean(),
                'std': self.train_df[col].std(),
                'skew': self.train_df[col].skew(),
                'kurtosis': self.train_df[col].kurtosis(),
                'min': self.train_df[col].min(),
                'max': self.train_df[col].max()
            }
            distribution_stats[col] = stats
            
            print(f"{col}:")
            print(f"  평균: {stats['mean']:.2f}, 표준편차: {stats['std']:.2f}")
            print(f"  왜도: {stats['skew']:.3f}, 첨도: {stats['kurtosis']:.3f}")
        
        self.analysis_results['feature_distributions'] = distribution_stats
        return distribution_stats
    
    def analyze_feature_importance(self):
        """피처 중요도 분석"""
        print("\n=== 피처 중요도 분석 ===")
        
        train_encoded = self.train_df.copy()
        le = LabelEncoder()
        
        categorical_cols = ['gender', 'subscription_type']
        for col in categorical_cols:
            train_encoded[col] = le.fit_transform(train_encoded[col])
        
        feature_cols = [col for col in train_encoded.columns if col not in ['ID', 'support_needs']]
        X = train_encoded[feature_cols]
        y = train_encoded['support_needs']
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_results = dict(zip(feature_cols, mi_scores))
        mi_sorted = sorted(mi_results.items(), key=lambda x: x[1], reverse=True)
        
        print("상호정보량 기반 피처 중요도:")
        for feature, score in mi_sorted:
            print(f"  {feature}: {score:.4f}")
        
        self.analysis_results['feature_importance'] = mi_results
        return mi_results
    
    def analyze_correlations(self):
        """고급 상관관계 분석"""
        print("\n=== 고급 상관관계 분석 ===")
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 
                       'contract_length', 'after_interaction', 'support_needs']
        
        corr_matrix = self.train_df[numeric_cols].corr()
        
        target_corr = corr_matrix['support_needs'].abs().sort_values(ascending=False)
        print("타겟과의 상관관계 (절댓값):")
        for feature, corr in target_corr[:-1].items():
            print(f"  {feature}: {corr:.4f}")
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            print("\n다중공선성 위험 피처 쌍:")
            for feat1, feat2, corr in high_corr_pairs:
                print(f"  {feat1} - {feat2}: {corr:.4f}")
        
        self.analysis_results['correlations'] = {
            'target_correlations': target_corr.to_dict(),
            'high_correlations': high_corr_pairs
        }
        
        return corr_matrix
    
    def analyze_data_quality(self):
        """데이터 품질 분석"""
        print("\n=== 데이터 품질 분석 ===")
        
        quality_report = {}
        
        missing_train = self.train_df.isnull().sum()
        missing_test = self.test_df.isnull().sum()
        
        print("결측치 현황:")
        print("  훈련 데이터:", missing_train[missing_train > 0].to_dict() if missing_train.sum() > 0 else "없음")
        print("  테스트 데이터:", missing_test[missing_test > 0].to_dict() if missing_test.sum() > 0 else "없음")
        
        train_duplicates = self.train_df.duplicated().sum()
        test_duplicates = self.test_df.duplicated().sum()
        
        print(f"중복 행: 훈련 {train_duplicates}개, 테스트 {test_duplicates}개")
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 
                       'contract_length', 'after_interaction']
        
        outlier_counts = {}
        for col in numeric_cols:
            Q1 = self.train_df[col].quantile(0.25)
            Q3 = self.train_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((self.train_df[col] < lower_bound) | 
                       (self.train_df[col] > upper_bound)).sum()
            outlier_counts[col] = outliers
        
        print("IQR 기준 이상치 개수:")
        for col, count in outlier_counts.items():
            pct = count / len(self.train_df) * 100
            print(f"  {col}: {count}개 ({pct:.2f}%)")
        
        quality_report = {
            'missing_values': {'train': missing_train.to_dict(), 'test': missing_test.to_dict()},
            'duplicates': {'train': train_duplicates, 'test': test_duplicates},
            'outliers': outlier_counts
        }
        
        self.analysis_results['data_quality'] = quality_report
        return quality_report
    
    def analyze_train_test_distribution(self):
        """훈련-테스트 분포 차이 분석"""
        print("\n=== 훈련-테스트 분포 차이 분석 ===")
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 
                       'contract_length', 'after_interaction']
        
        distribution_shifts = {}
        
        for col in numeric_cols:
            train_mean = self.train_df[col].mean()
            test_mean = self.test_df[col].mean()
            train_std = self.train_df[col].std()
            test_std = self.test_df[col].std()
            
            mean_shift = abs((test_mean - train_mean) / train_mean) * 100
            std_shift = abs((test_std - train_std) / train_std) * 100
            
            distribution_shifts[col] = {
                'mean_shift_pct': mean_shift,
                'std_shift_pct': std_shift
            }
            
            print(f"{col}:")
            print(f"  평균 변화: {mean_shift:.2f}%")
            print(f"  표준편차 변화: {std_shift:.2f}%")
        
        categorical_cols = ['gender', 'subscription_type']
        for col in categorical_cols:
            train_dist = self.train_df[col].value_counts(normalize=True)
            test_dist = self.test_df[col].value_counts(normalize=True)
            
            print(f"\n{col} 분포 비교:")
            for category in train_dist.index:
                train_pct = train_dist.get(category, 0) * 100
                test_pct = test_dist.get(category, 0) * 100
                diff = abs(train_pct - test_pct)
                print(f"  {category}: 훈련 {train_pct:.1f}% vs 테스트 {test_pct:.1f}% (차이: {diff:.1f}%)")
        
        self.analysis_results['distribution_shifts'] = distribution_shifts
        return distribution_shifts
    
    def detect_potential_leakage(self):
        """데이터 누수 탐지"""
        print("\n=== 데이터 누수 탐지 ===")
        
        train_ids = self.train_df['ID'].tolist()
        test_ids = self.test_df['ID'].tolist()
        
        common_ids = set(train_ids) & set(test_ids)
        if common_ids:
            print(f"경고: 훈련-테스트 간 공통 ID {len(common_ids)}개 발견!")
        else:
            print("ID 중복: 없음")
        
        train_id_numbers = [int(id.split('_')[1]) for id in train_ids]
        test_id_numbers = [int(id.split('_')[1]) for id in test_ids]
        
        print(f"훈련 ID 범위: {min(train_id_numbers)} - {max(train_id_numbers)}")
        print(f"테스트 ID 범위: {min(test_id_numbers)} - {max(test_id_numbers)}")
        
        common_features = [col for col in self.train_df.columns if col in self.test_df.columns and col != 'ID']
        
        train_features = self.train_df[common_features]
        test_features = self.test_df[common_features]
        
        train_hash = pd.util.hash_pandas_object(train_features, index=False)
        test_hash = pd.util.hash_pandas_object(test_features, index=False)
        
        common_patterns = len(set(train_hash) & set(test_hash))
        print(f"동일한 피처 패턴: {common_patterns}개")
        
        self.analysis_results['leakage_detection'] = {
            'common_ids': len(common_ids),
            'common_patterns': common_patterns
        }
    
    def generate_comprehensive_report(self):
        """종합 분석 보고서 생성"""
        print("\n" + "="*60)
        print("종합 데이터 분석 보고서")
        print("="*60)
        
        print("\n핵심 발견사항:")
        
        imbalance = self.analysis_results.get('target_distribution', {}).get('imbalance_ratio', 0)
        if imbalance > 2:
            print(f"- 심각한 클래스 불균형: {imbalance:.2f}")
        
        shifts = self.analysis_results.get('distribution_shifts', {})
        high_shift_features = [feat for feat, info in shifts.items() 
                              if info.get('mean_shift_pct', 0) > 5]
        if high_shift_features:
            print(f"- 분포 변화 피처: {high_shift_features}")
        
        outliers = self.analysis_results.get('data_quality', {}).get('outliers', {})
        high_outlier_features = [feat for feat, count in outliers.items() 
                                if count > len(self.train_df) * 0.05]
        if high_outlier_features:
            print(f"- 이상치 많은 피처: {high_outlier_features}")
        
        print("\n권장사항:")
        print("- 고급 피처 엔지니어링 (다항식, 타겟인코딩) 필수")
        print("- 강력한 교차검증 전략 필요")
        print("- 클래스 불균형 처리 방법 적용")
        print("- 앙상블 모델 다양성 확보")
        print("- 예측 후처리 및 보정 적용")
    
    def run_complete_analysis(self):
        """전체 분석 실행"""
        print("고급 데이터 분석 시작")
        print("="*50)
        
        self.load_data()
        
        self.analyze_target_distribution()
        self.analyze_feature_distributions()
        self.analyze_feature_importance()
        self.analyze_correlations()
        self.analyze_data_quality()
        self.analyze_train_test_distribution()
        self.detect_potential_leakage()
        
        self.generate_comprehensive_report()
        
        return self.analysis_results

def main():
    """메인 실행 함수"""
    analyzer = DataAnalyzer()
    results = analyzer.run_complete_analysis()
    return analyzer, results

if __name__ == "__main__":
    main()