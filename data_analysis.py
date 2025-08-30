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
        
    def load_data(self):
        """데이터 로드"""
        self.train_df = pd.read_csv('train.csv')
        self.test_df = pd.read_csv('test.csv')
        
        print(f"훈련 데이터: {self.train_df.shape}")
        print(f"테스트 데이터: {self.test_df.shape}")
        
        return self.train_df, self.test_df
    
    def analyze_target(self):
        """타겟 변수 분석"""
        print("타겟 분포 분석")
        
        target_counts = self.train_df['support_needs'].value_counts().sort_index()
        total = len(self.train_df)
        
        print("클래스 분포:")
        for cls, count in target_counts.items():
            pct = count / total * 100
            print(f"  클래스 {cls}: {count:,}개 ({pct:.1f}%)")
        
        imbalance_ratio = target_counts.max() / target_counts.min()
        
        self.analysis_results['target'] = {
            'distribution': target_counts.to_dict(),
            'imbalance_ratio': imbalance_ratio
        }
        
        return target_counts, imbalance_ratio
    
    def analyze_features(self):
        """피처 특성 분석"""
        print("피처 특성 분석")
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 
                       'contract_length', 'after_interaction']
        
        feature_stats = {}
        
        for col in numeric_cols:
            stats_dict = {
                'mean': self.train_df[col].mean(),
                'std': self.train_df[col].std(),
                'skew': self.train_df[col].skew(),
                'min': self.train_df[col].min(),
                'max': self.train_df[col].max(),
                'missing': self.train_df[col].isnull().sum()
            }
            feature_stats[col] = stats_dict
            
        self.analysis_results['features'] = feature_stats
        return feature_stats
    
    def analyze_correlations(self):
        """상관관계 분석"""
        print("상관관계 분석")
        
        train_encoded = self.train_df.copy()
        le = LabelEncoder()
        
        categorical_cols = ['gender', 'subscription_type']
        for col in categorical_cols:
            train_encoded[col] = le.fit_transform(train_encoded[col])
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 
                       'contract_length', 'after_interaction']
        all_cols = numeric_cols + categorical_cols + ['support_needs']
        
        corr_matrix = train_encoded[all_cols].corr()
        target_corr = corr_matrix['support_needs'].abs().sort_values(ascending=False)
        
        print("타겟 상관관계:")
        for feature, corr in target_corr[:-1].items():
            print(f"  {feature}: {corr:.3f}")
        
        self.analysis_results['correlations'] = target_corr.to_dict()
        return corr_matrix, target_corr
    
    def analyze_distribution_shifts(self):
        """분포 변화 분석"""
        print("훈련-테스트 분포 분석")
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 
                       'contract_length', 'after_interaction']
        
        shifts = {}
        
        for col in numeric_cols:
            train_mean = self.train_df[col].mean()
            test_mean = self.test_df[col].mean()
            train_std = self.train_df[col].std()
            test_std = self.test_df[col].std()
            
            mean_shift = abs((test_mean - train_mean) / train_mean) * 100
            std_shift = abs((test_std - train_std) / train_std) * 100
            
            shifts[col] = {
                'mean_shift': mean_shift,
                'std_shift': std_shift
            }
            
            if mean_shift > 5 or std_shift > 10:
                print(f"  {col}: 평균 {mean_shift:.1f}%, 표준편차 {std_shift:.1f}% 변화")
        
        self.analysis_results['distribution_shifts'] = shifts
        return shifts
    
    def detect_outliers(self):
        """이상치 탐지"""
        print("이상치 분석")
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 
                       'contract_length', 'after_interaction']
        
        outlier_counts = {}
        
        for col in numeric_cols:
            Q1 = self.train_df[col].quantile(0.25)
            Q3 = self.train_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outliers = ((self.train_df[col] < lower) | (self.train_df[col] > upper)).sum()
            outlier_pct = outliers / len(self.train_df) * 100
            
            outlier_counts[col] = {
                'count': outliers,
                'percentage': outlier_pct
            }
            
            if outlier_pct > 5:
                print(f"  {col}: {outliers}개 ({outlier_pct:.1f}%)")
        
        self.analysis_results['outliers'] = outlier_counts
        return outlier_counts
    
    def analyze_feature_importance(self):
        """피처 중요도 분석"""
        print("피처 중요도 분석")
        
        train_encoded = self.train_df.copy()
        le = LabelEncoder()
        
        categorical_cols = ['gender', 'subscription_type']
        for col in categorical_cols:
            train_encoded[col] = le.fit_transform(train_encoded[col])
        
        feature_cols = [col for col in train_encoded.columns 
                       if col not in ['ID', 'support_needs']]
        
        X = train_encoded[feature_cols]
        y = train_encoded['support_needs']
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        importance_dict = dict(zip(feature_cols, mi_scores))
        importance_sorted = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        print("상호정보량 중요도:")
        for feature, score in importance_sorted:
            print(f"  {feature}: {score:.4f}")
        
        self.analysis_results['importance'] = importance_dict
        return importance_dict
    
    def check_data_leakage(self):
        """데이터 누수 확인"""
        print("데이터 누수 확인")
        
        issues = []
        
        # ID 중복 확인
        train_ids = set(self.train_df['ID'])
        test_ids = set(self.test_df['ID'])
        common_ids = train_ids & test_ids
        
        if common_ids:
            issues.append(f"공통 ID {len(common_ids)}개")
        
        # 시간적 순서 확인
        train_id_nums = [int(id.split('_')[1]) for id in self.train_df['ID']]
        test_id_nums = [int(id.split('_')[1]) for id in self.test_df['ID']]
        
        train_max = max(train_id_nums)
        test_min = min(test_id_nums)
        
        if train_max >= test_min:
            issues.append("시간적 순서 위반")
        
        # after_interaction 피처 누수 위험
        if 'after_interaction' in self.train_df.columns:
            after_corr = self.train_df[['after_interaction', 'support_needs']].corr().iloc[0, 1]
            if abs(after_corr) > 0.3:
                issues.append(f"after_interaction 높은 상관관계: {after_corr:.3f}")
        
        self.analysis_results['leakage'] = issues
        
        if issues:
            print("누수 위험 발견:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("누수 위험 없음")
        
        return len(issues) == 0
    
    def generate_insights(self):
        """분석 결과 요약"""
        print("\n분석 결과 요약")
        print("=" * 30)
        
        target_info = self.analysis_results.get('target', {})
        imbalance = target_info.get('imbalance_ratio', 0)
        
        if imbalance > 2:
            print(f"클래스 불균형 심각: {imbalance:.1f}")
        
        shifts = self.analysis_results.get('distribution_shifts', {})
        high_shift_features = [feat for feat, info in shifts.items() 
                              if info.get('mean_shift', 0) > 5]
        
        if high_shift_features:
            print(f"분포 변화 피처: {high_shift_features}")
        
        leakage_issues = self.analysis_results.get('leakage', [])
        if leakage_issues:
            print(f"데이터 누수 위험: {len(leakage_issues)}개")
        
        importance = self.analysis_results.get('importance', {})
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print("핵심 피처:")
        for feat, score in top_features:
            print(f"  {feat}: {score:.3f}")
    
    def run_analysis(self):
        """전체 분석 실행"""
        print("데이터 분석 시작")
        print("=" * 40)
        
        self.load_data()
        self.analyze_target()
        self.analyze_features()
        self.analyze_correlations()
        self.analyze_distribution_shifts()
        self.detect_outliers()
        self.analyze_feature_importance()
        self.check_data_leakage()
        self.generate_insights()
        
        return self.analysis_results

def main():
    analyzer = DataAnalyzer()
    results = analyzer.run_analysis()
    return analyzer, results

if __name__ == "__main__":
    main()