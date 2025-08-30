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
        
        # 실제 존재하는 컬럼만 선택
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
            
            # 실제 존재하는 컬럼만 선택
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
        
        # 공통 컬럼만 선택
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
        
        # 실제 존재하는 컬럼만 선택
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
            
            # 결측치 처리
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
            # ID 중복 확인
            if 'ID' in self.train_df.columns and 'ID' in self.test_df.columns:
                train_ids = set(self.train_df['ID'])
                test_ids = set(self.test_df['ID'])
                common_ids = train_ids & test_ids
                
                if common_ids:
                    issues.append(f"공통 ID {len(common_ids)}개")
            
            # 시간적 순서 확인
            if 'ID' in self.train_df.columns and 'ID' in self.test_df.columns:
                try:
                    train_id_nums = []
                    test_id_nums = []
                    
                    for train_id in self.train_df['ID']:
                        if '_' in str(train_id):
                            train_id_nums.append(int(str(train_id).split('_')[1]))
                    
                    for test_id in self.test_df['ID']:
                        if '_' in str(test_id):
                            test_id_nums.append(int(str(test_id).split('_')[1]))
                    
                    if train_id_nums and test_id_nums:
                        train_max = max(train_id_nums)
                        test_min = min(test_id_nums)
                        
                        if train_max >= test_min:
                            issues.append("시간적 순서 위반")
                            
                except Exception as e:
                    print(f"ID 시간 순서 분석 오류: {e}")
            
            # after_interaction 피처 누수 위험
            if 'after_interaction' in self.train_df.columns and 'support_needs' in self.train_df.columns:
                try:
                    after_corr = self.train_df[['after_interaction', 'support_needs']].corr().iloc[0, 1]
                    if abs(after_corr) > 0.3:
                        issues.append(f"after_interaction 높은 상관관계: {after_corr:.3f}")
                except Exception as e:
                    print(f"상관관계 분석 오류: {e}")
            
            self.analysis_results['leakage'] = issues
            
            if issues:
                print("누수 위험 발견:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("누수 위험 없음")
            
            return len(issues) == 0
            
        except Exception as e:
            print(f"누수 확인 오류: {e}")
            return True
    
    def generate_insights(self):
        """분석 결과 요약"""
        print("\n분석 결과 요약")
        print("=" * 30)
        
        try:
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
            if importance:
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
                
                print("핵심 피처:")
                for feat, score in top_features:
                    print(f"  {feat}: {score:.3f}")
                    
        except Exception as e:
            print(f"인사이트 생성 오류: {e}")
    
    def validate_data_structure(self):
        """데이터 구조 검증"""
        print("데이터 구조 검증")
        
        issues = []
        
        try:
            # 필수 컬럼 확인
            required_train_cols = ['ID', 'age', 'gender', 'subscription_type', 
                                 'tenure', 'frequent', 'payment_interval', 
                                 'contract_length', 'after_interaction', 'support_needs']
            
            required_test_cols = ['ID', 'age', 'gender', 'subscription_type', 
                                'tenure', 'frequent', 'payment_interval', 
                                'contract_length', 'after_interaction']
            
            missing_train = [col for col in required_train_cols if col not in self.train_df.columns]
            missing_test = [col for col in required_test_cols if col not in self.test_df.columns]
            
            if missing_train:
                issues.append(f"훈련 데이터 누락 컬럼: {missing_train}")
            
            if missing_test:
                issues.append(f"테스트 데이터 누락 컬럼: {missing_test}")
            
            # 데이터 타입 확인
            if 'support_needs' in self.train_df.columns:
                if self.train_df['support_needs'].dtype not in ['int64', 'int32']:
                    issues.append("support_needs가 정수형이 아님")
                
                # 타겟 값 범위 확인
                unique_targets = set(self.train_df['support_needs'].unique())
                expected_targets = {0, 1, 2}
                
                if not unique_targets.issubset(expected_targets):
                    issues.append(f"잘못된 타겟 값: {unique_targets - expected_targets}")
            
            if issues:
                print("구조 문제:")
                for issue in issues:
                    print(f"  - {issue}")
                return False
            else:
                print("데이터 구조 정상")
                return True
                
        except Exception as e:
            print(f"구조 검증 오류: {e}")
            return False
    
    def run_analysis(self):
        """전체 분석 실행"""
        print("데이터 분석 시작")
        print("=" * 40)
        
        # 데이터 로드
        if self.load_data() is None:
            print("데이터 로드 실패")
            return {}
        
        # 데이터 구조 검증
        if not self.validate_data_structure():
            print("데이터 구조 문제 발견")
        
        # 개별 분석 수행 (오류가 있어도 계속 진행)
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
            self.generate_insights()
        except Exception as e:
            print(f"인사이트 생성 실패: {e}")
        
        print("데이터 분석 완료")
        return self.analysis_results

def main():
    analyzer = DataAnalyzer()
    results = analyzer.run_analysis()
    return analyzer, results

if __name__ == "__main__":
    main()