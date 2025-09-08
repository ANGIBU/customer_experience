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
        self.temporal_threshold = None
        
    def load_data(self):
        """데이터 로드"""
        try:
            self.train_df = pd.read_csv('train.csv')
            self.test_df = pd.read_csv('test.csv')
            return self.train_df, self.test_df
        except Exception as e:
            return None, None
    
    def analyze_target_distribution(self):
        """타겟 분포 분석"""
        if 'support_needs' not in self.train_df.columns:
            return {}
        
        target_counts = self.train_df['support_needs'].value_counts().sort_index()
        total = len(self.train_df)
        
        distribution_info = {}
        for cls, count in target_counts.items():
            pct = count / total
            distribution_info[cls] = {'count': count, 'percentage': pct}
        
        max_count = target_counts.max()
        min_count = target_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else 0
        
        return {
            'distribution': distribution_info,
            'imbalance_ratio': imbalance_ratio,
            'total_samples': total
        }
    
    def analyze_temporal_patterns(self):
        """시간적 패턴 분석"""
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
            
            # 시간적 분할점 계산 (85% 퍼센타일로 조정)
            overlap_threshold = int(np.percentile(train_id_nums, 85))
            
            self.temporal_threshold = overlap_threshold
            
            # 안전 구간 계산
            safe_indices = [i for i, tid in enumerate(train_id_nums) if tid > overlap_threshold]
            safe_ratio = len(safe_indices) / len(train_id_nums)
            
            return {
                'train_range': train_range,
                'test_range': test_range,
                'temporal_threshold': self.temporal_threshold,
                'safe_ratio': safe_ratio,
                'safe_indices': safe_indices
            }
            
        return {}
    
    def detect_data_leakage(self):
        """데이터 누수 탐지"""
        leakage_features = {}
        
        if 'after_interaction' in self.train_df.columns and 'support_needs' in self.train_df.columns:
            # 클래스별 통계
            class_stats = {}
            for cls in [0, 1, 2]:
                class_data = self.train_df[self.train_df['support_needs'] == cls]['after_interaction'].dropna()
                if len(class_data) > 0:
                    class_stats[cls] = {
                        'mean': class_data.mean(),
                        'std': class_data.std(),
                        'count': len(class_data)
                    }
            
            # 상관관계 분석
            correlation = self.train_df[['after_interaction', 'support_needs']].corr().iloc[0, 1]
            
            # 상호정보량
            after_clean = self.train_df['after_interaction'].fillna(0)
            target_clean = self.train_df['support_needs']
            mi_score = mutual_info_classif(after_clean.values.reshape(-1, 1), target_clean, random_state=42)[0]
            
            # 분리도 측정
            if len(class_stats) >= 2:
                means = [stats['mean'] for stats in class_stats.values()]
                separation = max(means) - min(means)
                
                # 통계적 검정
                groups = []
                for cls in [0, 1, 2]:
                    if cls in class_stats:
                        group_data = self.train_df[self.train_df['support_needs'] == cls]['after_interaction'].dropna()
                        groups.append(group_data)
                
                f_stat, p_value = stats.f_oneway(*groups) if len(groups) >= 2 else (0, 1)
                
                # 누수 기준 (0.18, 0.35, 0.001)
                leakage_features['after_interaction'] = {
                    'correlation': correlation,
                    'mutual_info': mi_score,
                    'class_separation': separation,
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'class_stats': class_stats,
                    'is_leakage': abs(correlation) > 0.18 or mi_score > 0.35 or p_value < 0.001
                }
        
        return leakage_features
    
    def analyze_feature_stability(self):
        """피처 안정성 분석"""
        numeric_features = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length', 'after_interaction']
        stability_results = {}
        
        for feature in numeric_features:
            if feature in self.train_df.columns and feature in self.test_df.columns:
                train_vals = self.train_df[feature].dropna()
                test_vals = self.test_df[feature].dropna()
                
                if len(train_vals) > 100 and len(test_vals) > 100:
                    # KS 테스트
                    ks_stat, ks_p = ks_2samp(train_vals, test_vals)
                    
                    # PSI 계산
                    psi_score = self.calculate_psi(train_vals, test_vals)
                    
                    # 기본 통계량 비교
                    train_stats = {
                        'mean': train_vals.mean(),
                        'std': train_vals.std(),
                        'median': train_vals.median(),
                        'q25': train_vals.quantile(0.25),
                        'q75': train_vals.quantile(0.75)
                    }
                    
                    test_stats = {
                        'mean': test_vals.mean(),
                        'std': test_vals.std(),
                        'median': test_vals.median(),
                        'q25': test_vals.quantile(0.25),
                        'q75': test_vals.quantile(0.75)
                    }
                    
                    stability_results[feature] = {
                        'ks_statistic': ks_stat,
                        'ks_p_value': ks_p,
                        'psi_score': psi_score,
                        'train_stats': train_stats,
                        'test_stats': test_stats,
                        'is_stable': ks_stat < 0.1 and psi_score < 0.2
                    }
        
        return stability_results
    
    def calculate_psi(self, train_data, test_data, bins=10):
        """PSI 계산"""
        try:
            min_val = min(train_data.min(), test_data.min())
            max_val = max(train_data.max(), test_data.max())
            
            if min_val == max_val:
                return 0.0
            
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            bin_edges[0] -= 1e-6
            bin_edges[-1] += 1e-6
            
            train_hist, _ = np.histogram(train_data, bins=bin_edges)
            test_hist, _ = np.histogram(test_data, bins=bin_edges)
            
            train_pct = (train_hist + 1) / (len(train_data) + bins)
            test_pct = (test_hist + 1) / (len(test_data) + bins)
            
            psi = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))
            
            return psi
            
        except Exception:
            return 0.0
    
    def analyze_categorical_features(self):
        """범주형 피처 분석"""
        categorical_cols = ['gender', 'subscription_type']
        categorical_analysis = {}
        
        for col in categorical_cols:
            if col in self.train_df.columns and col in self.test_df.columns:
                train_counts = self.train_df[col].value_counts()
                test_counts = self.test_df[col].value_counts()
                
                # 공통 카테고리
                common_cats = set(train_counts.index) & set(test_counts.index)
                
                if len(common_cats) > 1:
                    # 분포 비교
                    train_dist = train_counts / len(self.train_df)
                    test_dist = test_counts / len(self.test_df)
                    
                    # 카이제곱 검정
                    contingency_table = []
                    for cat in sorted(common_cats):
                        contingency_table.append([
                            train_counts.get(cat, 0),
                            test_counts.get(cat, 0)
                        ])
                    
                    try:
                        chi2_stat, chi2_p = chi2_contingency(np.array(contingency_table).T)[:2]
                    except:
                        chi2_stat, chi2_p = 0, 1
                    
                    categorical_analysis[col] = {
                        'train_distribution': train_dist.to_dict(),
                        'test_distribution': test_dist.to_dict(),
                        'chi2_statistic': chi2_stat,
                        'chi2_p_value': chi2_p,
                        'common_categories': list(common_cats),
                        'is_stable': chi2_p > 0.05
                    }
        
        return categorical_analysis
    
    def compute_feature_importance_baseline(self):
        """기본 피처 중요도"""
        if 'support_needs' not in self.train_df.columns:
            return {}
        
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
        
        # 정규화
        max_score = max(importance_dict.values()) if importance_dict else 1
        if max_score > 0:
            importance_dict = {k: v/max_score for k, v in importance_dict.items()}
        
        return importance_dict
    
    def validate_data_integrity(self):
        """데이터 무결성 검증"""
        issues = []
        
        # ID 중복 확인
        train_duplicates = self.train_df['ID'].duplicated().sum()
        test_duplicates = self.test_df['ID'].duplicated().sum()
        
        if train_duplicates > 0:
            issues.append(f"train_id_duplicates: {train_duplicates}")
        if test_duplicates > 0:
            issues.append(f"test_id_duplicates: {test_duplicates}")
        
        # 타겟 유효성
        if 'support_needs' in self.train_df.columns:
            invalid_targets = ~self.train_df['support_needs'].isin([0, 1, 2])
            invalid_count = invalid_targets.sum()
            if invalid_count > 0:
                issues.append(f"invalid_targets: {invalid_count}")
        
        # 결측치 패턴
        missing_info = {}
        for col in self.train_df.columns:
            if col != 'ID':
                missing_ratio = self.train_df[col].isnull().mean()
                missing_info[col] = missing_ratio
                if missing_ratio > 0.4:
                    issues.append(f"{col}_high_missing: {missing_ratio:.3f}")
        
        return len(issues) == 0, issues, missing_info
    
    def run_analysis(self):
        """분석 실행"""
        if self.load_data() is None:
            return {}
        
        # 무결성 검증
        integrity_ok, integrity_issues, missing_info = self.validate_data_integrity()
        self.analysis_results['integrity'] = {
            'passed': integrity_ok,
            'issues': integrity_issues,
            'missing_info': missing_info
        }
        
        # 타겟 분포
        target_info = self.analyze_target_distribution()
        self.analysis_results['target_distribution'] = target_info
        
        # 시간적 패턴
        temporal_info = self.analyze_temporal_patterns()
        self.analysis_results['temporal'] = temporal_info
        
        # 데이터 누수 탐지
        leakage_info = self.detect_data_leakage()
        self.analysis_results['leakage'] = leakage_info
        
        # 피처 안정성
        stability_info = self.analyze_feature_stability()
        self.analysis_results['stability'] = stability_info
        
        # 범주형 분석
        categorical_info = self.analyze_categorical_features()
        self.analysis_results['categorical'] = categorical_info
        
        # 피처 중요도
        importance_info = self.compute_feature_importance_baseline()
        self.analysis_results['feature_importance'] = importance_info
        
        return self.analysis_results

def main():
    try:
        analyzer = DataAnalyzer()
        results = analyzer.run_analysis()
        
        if results:
            print("데이터 분석 완료")
            return analyzer, results
        else:
            print("데이터 분석 실패")
            return None, {}
            
    except Exception as e:
        print(f"데이터 분석 오류: {e}")
        return None, {}

if __name__ == "__main__":
    main()