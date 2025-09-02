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
        try:
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
            
            if train_id_nums and test_id_nums and len(train_id_nums) > 100 and len(test_id_nums) > 100:
                train_range = [min(train_id_nums), max(train_id_nums)]
                test_range = [min(test_id_nums), max(test_id_nums)]
                
                train_max = max(train_id_nums)
                test_min = min(test_id_nums)
                
                overlap_threshold = min(train_max, test_min - 1000)
                
                self.temporal_threshold = overlap_threshold
                
                safe_indices = [i for i, tid in enumerate(train_id_nums) if tid <= overlap_threshold]
                safe_ratio = len(safe_indices) / len(train_id_nums) if train_id_nums else 0.5
                
                temporal_gap = test_min - train_max if train_max < test_min else 0
                overlap_samples = len([x for x in train_id_nums if x >= test_min])
                overlap_ratio = overlap_samples / len(train_id_nums) if train_id_nums else 0
                
                return {
                    'train_range': train_range,
                    'test_range': test_range,
                    'temporal_threshold': self.temporal_threshold,
                    'safe_ratio': safe_ratio,
                    'safe_indices': safe_indices,
                    'temporal_gap': temporal_gap,
                    'overlap_ratio': overlap_ratio,
                    'has_temporal_leak': overlap_ratio > 0.001,
                    'can_use_after_interaction': overlap_ratio == 0.0 and temporal_gap > 500
                }
            
            return {
                'safe_ratio': 0.5,
                'has_temporal_leak': True,
                'temporal_threshold': None,
                'can_use_after_interaction': False
            }
            
        except Exception as e:
            return {
                'safe_ratio': 0.5,
                'has_temporal_leak': True,
                'temporal_threshold': None,
                'can_use_after_interaction': False
            }
    
    def detect_data_leakage(self):
        """데이터 누수 탐지"""
        leakage_features = {}
        
        if 'after_interaction' in self.train_df.columns:
            temporal_info = self.analyze_temporal_patterns()
            can_use_safely = temporal_info.get('can_use_after_interaction', False)
            
            if can_use_safely:
                leakage_features['after_interaction'] = {
                    'should_remove': False,
                    'reason': 'safe_with_temporal_split',
                    'risk_level': 'LOW'
                }
            else:
                leakage_features['after_interaction'] = {
                    'should_remove': True,
                    'reason': 'temporal_leak_detected',
                    'risk_level': 'CRITICAL'
                }
        
        if 'support_needs' not in self.train_df.columns:
            return leakage_features
        
        feature_cols = [col for col in self.train_df.columns 
                       if col not in ['ID', 'support_needs']]
        
        target = self.train_df['support_needs']
        
        for col in feature_cols:
            if col in self.train_df.columns and col != 'after_interaction':
                if self.train_df[col].dtype in [np.number]:
                    correlation = abs(self.train_df[col].corr(target))
                    
                    if correlation > 0.95:
                        leakage_features[col] = {
                            'should_remove': True,
                            'correlation': correlation,
                            'reason': 'extreme_correlation',
                            'risk_level': 'CRITICAL'
                        }
                    
                    variance = self.train_df[col].var()
                    if variance < 0.0001:
                        leakage_features[col] = {
                            'should_remove': True,
                            'variance': variance,
                            'reason': 'quasi_constant',
                            'risk_level': 'HIGH'
                        }
                
                unique_ratio = self.train_df[col].nunique() / len(self.train_df)
                if unique_ratio > 0.98:
                    leakage_features[col] = {
                        'should_remove': True,
                        'unique_ratio': unique_ratio,
                        'reason': 'high_cardinality',
                        'risk_level': 'HIGH'
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
                    ks_stat, ks_p = ks_2samp(train_vals, test_vals)
                    psi_score = self.calculate_psi(train_vals, test_vals)
                    
                    stability_results[feature] = {
                        'ks_statistic': ks_stat,
                        'ks_p_value': ks_p,
                        'psi_score': psi_score,
                        'is_stable': ks_stat < 0.15 and psi_score < 0.25
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
            
            return abs(psi)
            
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
                
                common_cats = set(train_counts.index) & set(test_counts.index)
                
                if len(common_cats) > 1:
                    train_dist = train_counts / len(self.train_df)
                    test_dist = test_counts / len(self.test_df)
                    
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
                        'is_stable': chi2_p > 0.01
                    }
        
        return categorical_analysis
    
    def compute_feature_importance_baseline(self):
        """기본 피처 중요도"""
        if 'support_needs' not in self.train_df.columns:
            return {}
        
        numeric_features = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        temporal_info = self.analyze_temporal_patterns()
        
        if temporal_info.get('can_use_after_interaction', False):
            numeric_features.append('after_interaction')
        
        available_numeric = [f for f in numeric_features if f in self.train_df.columns]
        
        categorical_features = ['gender', 'subscription_type']
        train_encoded = self.train_df.copy()
        
        for col in categorical_features:
            if col in self.train_df.columns:
                le = LabelEncoder()
                train_encoded[col] = le.fit_transform(train_encoded[col].fillna('Unknown'))
                available_numeric.append(col)
        
        if not available_numeric:
            return {}
        
        X = train_encoded[available_numeric].fillna(0)
        y = train_encoded['support_needs']
        
        try:
            mi_scores = mutual_info_classif(X, y, random_state=42)
            importance_dict = dict(zip(available_numeric, mi_scores))
            
            max_score = max(importance_dict.values()) if importance_dict else 1
            if max_score > 0:
                importance_dict = {k: v/max_score for k, v in importance_dict.items()}
            
            return importance_dict
        except:
            return {}
    
    def validate_data_integrity(self):
        """데이터 무결성 검증"""
        issues = []
        
        train_duplicates = self.train_df['ID'].duplicated().sum()
        test_duplicates = self.test_df['ID'].duplicated().sum()
        
        if train_duplicates > 0:
            issues.append(f"train_id_duplicates: {train_duplicates}")
        if test_duplicates > 0:
            issues.append(f"test_id_duplicates: {test_duplicates}")
        
        if 'support_needs' in self.train_df.columns:
            invalid_targets = ~self.train_df['support_needs'].isin([0, 1, 2])
            invalid_count = invalid_targets.sum()
            if invalid_count > 0:
                issues.append(f"invalid_targets: {invalid_count}")
        
        missing_info = {}
        for col in self.train_df.columns:
            if col != 'ID':
                missing_ratio = self.train_df[col].isnull().mean()
                missing_info[col] = missing_ratio
                if missing_ratio > 0.4:
                    issues.append(f"{col}_high_missing: {missing_ratio:.3f}")
        
        return len(issues) == 0, issues, missing_info
    
    def analyze_correlation_matrix(self):
        """상관관계 분석"""
        numeric_cols = [col for col in self.train_df.select_dtypes(include=[np.number]).columns 
                       if col not in ['support_needs']]
        
        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = self.train_df[numeric_cols].corr()
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.85:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlation_pairs': high_corr_pairs,
            'max_correlation': corr_matrix.abs().max().max() if not corr_matrix.empty else 0
        }
    
    def analyze_after_interaction_safety(self):
        """after_interaction 피처 안전성 분석"""
        if 'after_interaction' not in self.train_df.columns:
            return {'can_use': False, 'reason': 'feature_not_found'}
        
        temporal_info = self.analyze_temporal_patterns()
        overlap_ratio = temporal_info.get('overlap_ratio', 1.0)
        
        if overlap_ratio <= 0.001:
            correlation = abs(self.train_df['after_interaction'].corr(self.train_df['support_needs']))
            
            return {
                'can_use': True,
                'overlap_ratio': overlap_ratio,
                'correlation_with_target': correlation,
                'safety_level': 'SAFE',
                'recommendation': 'use_with_careful_validation'
            }
        else:
            return {
                'can_use': False,
                'overlap_ratio': overlap_ratio,
                'safety_level': 'RISKY',
                'recommendation': 'remove_feature'
            }
    
    def run_analysis(self):
        """분석 실행"""
        train_data, test_data = self.load_data()
        if train_data is None or test_data is None:
            return {}
        
        if self.train_df.empty or self.test_df.empty:
            return {}
        
        integrity_ok, integrity_issues, missing_info = self.validate_data_integrity()
        self.analysis_results['integrity'] = {
            'passed': integrity_ok,
            'issues': integrity_issues,
            'missing_info': missing_info
        }
        
        target_info = self.analyze_target_distribution()
        self.analysis_results['target_distribution'] = target_info
        
        temporal_info = self.analyze_temporal_patterns()
        self.analysis_results['temporal'] = temporal_info
        
        leakage_info = self.detect_data_leakage()
        self.analysis_results['leakage'] = leakage_info
        
        stability_info = self.analyze_feature_stability()
        self.analysis_results['stability'] = stability_info
        
        categorical_info = self.analyze_categorical_features()
        self.analysis_results['categorical'] = categorical_info
        
        importance_info = self.compute_feature_importance_baseline()
        self.analysis_results['feature_importance'] = importance_info
        
        correlation_info = self.analyze_correlation_matrix()
        self.analysis_results['correlation'] = correlation_info
        
        after_interaction_info = self.analyze_after_interaction_safety()
        self.analysis_results['after_interaction_analysis'] = after_interaction_info
        
        return self.analysis_results

def main():
    try:
        analyzer = DataAnalyzer()
        results = analyzer.run_analysis()
        
        if results:
            return analyzer, results
        else:
            return None, {}
            
    except Exception as e:
        return None, {}

if __name__ == "__main__":
    main()