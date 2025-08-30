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
        try:
            self.train_df = pd.read_csv('train.csv')
            self.test_df = pd.read_csv('test.csv')
            
            print(f"훈련 데이터: {self.train_df.shape}")
            print(f"테스트 데이터: {self.test_df.shape}")
            
            return self.train_df, self.test_df
            
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            return None, None
    
    def analyze_temporal_order(self):
        """시간적 순서 분석"""
        print("시간적 순서 분석")
        
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
            
            temporal_overlap = train_range[1] >= test_range[0]
            overlap_ratio = 0
            
            if temporal_overlap:
                overlap_count = len([x for x in train_id_nums if x >= test_range[0]])
                overlap_ratio = overlap_count / len(train_id_nums)
                print(f"시간적 겹침: {overlap_ratio:.3f}")
            
            return {
                'train_range': train_range,
                'test_range': test_range,
                'temporal_overlap': temporal_overlap,
                'overlap_ratio': overlap_ratio
            }
        
        return {}
    
    def analyze_class_patterns(self):
        """클래스별 패턴 분석"""
        print("클래스별 패턴 분석")
        
        if 'support_needs' not in self.train_df.columns:
            return {}
        
        class_patterns = {}
        
        for cls in [0, 1, 2]:
            class_data = self.train_df[self.train_df['support_needs'] == cls]
            
            if len(class_data) > 0:
                patterns = {}
                
                numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
                for col in numeric_cols:
                    if col in class_data.columns:
                        patterns[col] = {
                            'mean': class_data[col].mean(),
                            'median': class_data[col].median(),
                            'std': class_data[col].std()
                        }
                
                categorical_cols = ['gender', 'subscription_type']
                for col in categorical_cols:
                    if col in class_data.columns:
                        patterns[col] = class_data[col].value_counts(normalize=True).to_dict()
                
                class_patterns[cls] = patterns
        
        return class_patterns
    
    def analyze_target(self):
        """타겟 변수 분석"""
        print("타겟 분포 분석")
        
        if 'support_needs' not in self.train_df.columns:
            print("타겟 변수 없음")
            return None, 0
        
        try:
            target_counts = self.train_df['support_needs'].value_counts().sort_index()
            total = len(self.train_df)
            
            print("클래스 분포:")
            for cls, count in target_counts.items():
                pct = count / total * 100
                print(f"  클래스 {cls}: {count:,}개 ({pct:.1f}%)")
            
            imbalance_ratio = target_counts.max() / target_counts.min() if target_counts.min() > 0 else 0
            
            self.analysis_results['target'] = {
                'distribution': target_counts.to_dict(),
                'imbalance_ratio': imbalance_ratio
            }
            
            return target_counts, imbalance_ratio
            
        except Exception as e:
            print(f"타겟 분석 오류: {e}")
            return None, 0
    
    def analyze_features(self):
        """피처 특성 분석"""
        print("피처 특성 분석")
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 
                       'contract_length', 'after_interaction']
        
        available_cols = [col for col in numeric_cols if col in self.train_df.columns]
        
        feature_stats = {}
        
        for col in available_cols:
            try:
                stats_dict = {
                    'mean': self.train_df[col].mean(),
                    'std': self.train_df[col].std(),
                    'skew': self.train_df[col].skew(),
                    'min': self.train_df[col].min(),
                    'max': self.train_df[col].max(),
                    'missing': self.train_df[col].isnull().sum()
                }
                feature_stats[col] = stats_dict
                
            except Exception as e:
                print(f"  {col} 분석 오류: {e}")
                continue
                
        self.analysis_results['features'] = feature_stats
        return feature_stats
    
    def analyze_correlations(self):
        """상관관계 분석"""
        print("상관관계 분석")
        
        if 'support_needs' not in self.train_df.columns:
            print("타겟 변수 없어 상관관계 분석 불가")
            return None, None
        
        try:
            train_encoded = self.train_df.copy()
            le = LabelEncoder()
            
            categorical_cols = ['gender', 'subscription_type']
            for col in categorical_cols:
                if col in train_encoded.columns:
                    train_encoded[col] = le.fit_transform(train_encoded[col])
            
            numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 
                           'contract_length', 'after_interaction']
            
            available_cols = [col for col in numeric_cols + categorical_cols 
                             if col in train_encoded.columns]
            
            all_cols = available_cols + ['support_needs']
            
            corr_matrix = train_encoded[all_cols].corr()
            target_corr = corr_matrix['support_needs'].abs().sort_values(ascending=False)
            
            print("타겟 상관관계:")
            for feature, corr in target_corr[:-1].items():
                print(f"  {feature}: {corr:.3f}")
            
            self.analysis_results['correlations'] = target_corr.to_dict()
            return corr_matrix, target_corr
            
        except Exception as e:
            print(f"상관관계 분석 오류: {e}")
            return None, None
    
    def analyze_distribution_shifts(self):
        """분포 변화 분석"""
        print("훈련-테스트 분포 분석")
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 
                       'contract_length', 'after_interaction']
        
        common_cols = [col for col in numeric_cols 
                      if col in self.train_df.columns and col in self.test_df.columns]
        
        shifts = {}
        
        for col in common_cols:
            try:
                train_mean = self.train_df[col].mean()
                test_mean = self.test_df[col].mean()
                train_std = self.train_df[col].std()
                test_std = self.test_df[col].std()
                
                if train_mean != 0:
                    mean_shift = abs((test_mean - train_mean) / train_mean) * 100
                else:
                    mean_shift = 0
                    
                if train_std != 0:
                    std_shift = abs((test_std - train_std) / train_std) * 100
                else:
                    std_shift = 0
                
                shifts[col] = {
                    'mean_shift': mean_shift,
                    'std_shift': std_shift
                }
                
                if mean_shift > 5 or std_shift > 10:
                    print(f"  {col}: 평균 {mean_shift:.1f}%, 표준편차 {std_shift:.1f}% 변화")
                    
            except Exception as e:
                print(f"  {col} 분포 분석 오류: {e}")
                continue
        
        self.analysis_results['distribution_shifts'] = shifts
        return shifts
    
    def detect_outliers(self):
        """이상치 탐지"""
        print("이상치 분석")
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 
                       'contract_length', 'after_interaction']
        
        available_cols = [col for col in numeric_cols if col in self.train_df.columns]
        
        outlier_counts = {}
        
        for col in available_cols:
            try:
                Q1 = self.train_df[col].quantile(0.25)
                Q3 = self.train_df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
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
                else:
                    print(f"  {col}: IQR=0, 이상치 탐지 불가")
                    
            except Exception as e:
                print(f"  {col} 이상치 분석 오류: {e}")
                continue
        
        self.analysis_results['outliers'] = outlier_counts
        return outlier_counts
    
    def analyze_feature_importance(self):
        """피처 중요도 분석"""
        print("피처 중요도 분석")
        
        if 'support_needs' not in self.train_df.columns:
            print("타겟 변수 없어 중요도 분석 불가")
            return {}
        
        try:
            train_encoded = self.train_df.copy()
            le = LabelEncoder()
            
            categorical_cols = ['gender', 'subscription_type']
            for col in categorical_cols:
                if col in train_encoded.columns:
                    train_encoded[col] = le.fit_transform(train_encoded[col])
            
            feature_cols = [col for col in train_encoded.columns 
                           if col not in ['ID', 'support_needs']]
            
            if not feature_cols:
                print("분석할 피처 없음")
                return {}
            
            X = train_encoded[feature_cols]
            y = train_encoded['support_needs']
            
            X = X.fillna(0)
            
            mi_scores = mutual_info_classif(X, y, random_state=42)
            importance_dict = dict(zip(feature_cols, mi_scores))
            importance_sorted = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            print("상호정보량 중요도:")
            for feature, score in importance_sorted:
                print(f"  {feature}: {score:.4f}")
            
            self.analysis_results['importance'] = importance_dict
            return importance_dict
            
        except Exception as e:
            print(f"중요도 분석 오류: {e}")
            return {}
    
    def check_data_leakage(self):
        """데이터 누수 확인"""
        print("데이터 누수 확인")
        
        issues = []
        
        try:
            if 'ID' in self.train_df.columns and 'ID' in self.test_df.columns:
                train_ids = set(self.train_df['ID'])
                test_ids = set(self.test_df['ID'])
                common_ids = train_ids & test_ids
                
                if common_ids:
                    issues.append(f"공통 ID {len(common_ids)}개")
            
            temporal_analysis = self.analyze_temporal_order()
            if temporal_analysis.get('temporal_overlap', False):
                overlap_ratio = temporal_analysis.get('overlap_ratio', 0)
                if overlap_ratio > 0.1:
                    issues.append(f"시간적 순서 위반: 겹침 비율 {overlap_ratio:.1%}")
            
            if 'after_interaction' in self.train_df.columns and 'support_needs' in self.train_df.columns:
                try:
                    correlation = self.train_df[['after_interaction', 'support_needs']].corr().iloc[0, 1]
                    if abs(correlation) > 0.05:
                        issues.append(f"after_interaction 상관관계: {correlation:.3f}")
                except:
                    pass
        
            self.analysis_results['leakage'] = issues
            
            if issues:
                print("누수 위험:")
                for issue in issues:
                    print(f"  - {issue}")
                return False
            else:
                print("누수 위험 없음")
                return True
                
        except Exception as e:
            print(f"누수 확인 오류: {e}")
            return True
    
    def calculate_feature_stability(self):
        """피처 안정성 계산"""
        print("피처 안정성 분석")
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        common_cols = [col for col in numeric_cols 
                      if col in self.train_df.columns and col in self.test_df.columns]
        
        stability_scores = {}
        
        for col in common_cols:
            try:
                train_vals = self.train_df[col].dropna()
                test_vals = self.test_df[col].dropna()
                
                if len(train_vals) > 100 and len(test_vals) > 100:
                    from scipy.stats import ks_2samp
                    statistic, p_value = ks_2samp(train_vals, test_vals)
                    
                    stability_score = 1 - statistic
                    stability_scores[col] = {
                        'ks_statistic': statistic,
                        'p_value': p_value,
                        'stability_score': stability_score
                    }
                    
                    if stability_score < 0.95:
                        print(f"  {col}: 안정성 {stability_score:.3f}")
                        
            except Exception as e:
                print(f"  {col} 안정성 분석 오류: {e}")
                continue
        
        return stability_scores
    
    def analyze_class_separability(self):
        """클래스 분리도 분석"""
        print("클래스 분리도 분석")
        
        if 'support_needs' not in self.train_df.columns:
            return {}
        
        separability = {}
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_cols = [col for col in numeric_cols if col in self.train_df.columns]
        
        for col in available_cols:
            try:
                class_groups = []
                for cls in [0, 1, 2]:
                    class_data = self.train_df[self.train_df['support_needs'] == cls][col]
                    class_groups.append(class_data.dropna())
                
                if all(len(group) > 10 for group in class_groups):
                    from scipy.stats import f_oneway
                    f_stat, p_val = f_oneway(*class_groups)
                    
                    separability[col] = {
                        'f_statistic': f_stat,
                        'p_value': p_val,
                        'separable': p_val < 0.05
                    }
                    
                    if p_val < 0.01:
                        print(f"  {col}: 높은 분리도 (p={p_val:.4f})")
                        
            except Exception as e:
                print(f"  {col} 분리도 분석 오류: {e}")
                continue
        
        return separability
    
    def detect_feature_interactions(self):
        """피처 상호작용 탐지"""
        print("피처 상호작용 탐지")
        
        if 'support_needs' not in self.train_df.columns:
            return {}
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_cols = [col for col in numeric_cols if col in self.train_df.columns]
        
        interactions = {}
        
        for i, col1 in enumerate(available_cols):
            for col2 in available_cols[i+1:]:
                try:
                    interaction_col = self.train_df[col1] * self.train_df[col2]
                    
                    correlation = np.corrcoef(interaction_col.fillna(0), self.train_df['support_needs'])[0, 1]
                    
                    if abs(correlation) > 0.1:
                        interactions[f"{col1}_{col2}"] = {
                            'correlation': correlation,
                            'significant': abs(correlation) > 0.15
                        }
                        
                        if abs(correlation) > 0.15:
                            print(f"  {col1} × {col2}: 상관관계 {correlation:.3f}")
                            
                except Exception as e:
                    continue
        
        return interactions
    
    def run_analysis(self):
        """전체 분석 실행"""
        print("데이터 분석 시작")
        print("=" * 40)
        
        if self.load_data() is None:
            print("데이터 로드 실패")
            return {}
        
        try:
            self.analyze_target()
        except Exception as e:
            print(f"타겟 분석 실패: {e}")
        
        try:
            self.analyze_features()
        except Exception as e:
            print(f"피처 분석 실패: {e}")
        
        try:
            self.analyze_correlations()
        except Exception as e:
            print(f"상관관계 분석 실패: {e}")
        
        try:
            self.analyze_distribution_shifts()
        except Exception as e:
            print(f"분포 변화 분석 실패: {e}")
        
        try:
            self.detect_outliers()
        except Exception as e:
            print(f"이상치 분석 실패: {e}")
        
        try:
            self.analyze_feature_importance()
        except Exception as e:
            print(f"중요도 분석 실패: {e}")
        
        try:
            self.check_data_leakage()
        except Exception as e:
            print(f"누수 확인 실패: {e}")
        
        try:
            class_patterns = self.analyze_class_patterns()
            self.analysis_results['class_patterns'] = class_patterns
        except Exception as e:
            print(f"클래스 패턴 분석 실패: {e}")
        
        try:
            stability = self.calculate_feature_stability()
            self.analysis_results['stability'] = stability
        except Exception as e:
            print(f"안정성 분석 실패: {e}")
        
        try:
            separability = self.analyze_class_separability()
            self.analysis_results['separability'] = separability
        except Exception as e:
            print(f"분리도 분석 실패: {e}")
        
        try:
            interactions = self.detect_feature_interactions()
            self.analysis_results['interactions'] = interactions
        except Exception as e:
            print(f"상호작용 분석 실패: {e}")
        
        print("데이터 분석 완료")
        return self.analysis_results

def main():
    analyzer = DataAnalyzer()
    results = analyzer.run_analysis()
    return analyzer, results

if __name__ == "__main__":
    main()