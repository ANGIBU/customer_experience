# main.py

"""
고객 지원 필요 수준 예측 AI 알고리즘
목표 성능: 0.60+ 정확도, 다중 모델 앙상블, 피처 엔지니어링 활용

개발 환경:
- OS: Windows/Linux
- Python: 3.11.9
- 주요 라이브러리: pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, hyperopt

데이터 경로: 현재 디렉토리
- train.csv: 훈련 데이터
- test.csv: 테스트 데이터  
- sample_submission.csv: 제출 형식
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 사용자 정의 모듈 임포트
from data_analysis import DataAnalyzer
from feature_engineering import FeatureEngineer
from preprocessing import DataPreprocessor
from model_training import ModelTrainer
from validation import ValidationSystem
from prediction import PredictionSystem

class AISystem:
    """AI 시스템 전체 파이프라인 클래스"""
    
    def __init__(self):
        self.start_time = None
        self.results = {}
        self.target_accuracy = 0.60
        
    def setup_environment(self):
        """환경 설정"""
        print("=" * 60)
        print("고객 지원 필요 수준 예측 AI 시스템")
        print("=" * 60)
        print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python 버전: {sys.version}")
        print(f"작업 디렉토리: {os.getcwd()}")
        print(f"목표 성능: {self.target_accuracy:.1%} 이상")
        
        # 필수 파일 확인
        required_files = ['train.csv', 'test.csv', 'sample_submission.csv']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"필수 파일 없음: {missing_files}")
            return False
        
        print(f"필수 데이터 파일 확인 완료: {required_files}")
        
        # 결과 디렉토리 생성
        os.makedirs('models/pkl', exist_ok=True)
        os.makedirs('models/json', exist_ok=True)
        
        self.start_time = time.time()
        return True
    
    def step1_data_analysis(self):
        """1단계: 데이터 분석"""
        print("\n" + "="*50)
        print("1단계: 데이터 분석 및 패턴 탐지")
        print("="*50)
        
        try:
            analyzer = DataAnalyzer()
            analysis_results = analyzer.run_complete_analysis()
            
            self.results['data_analysis'] = {
                'target_imbalance': analysis_results.get('target_distribution', {}).get('imbalance_ratio', 0),
                'feature_importance': analysis_results.get('feature_importance', {}),
                'data_quality': analysis_results.get('data_quality', {}),
                'distribution_shifts': analysis_results.get('distribution_shifts', {})
            }
            
            print("데이터 분석 완료!")
            return True, analyzer
            
        except Exception as e:
            print(f"데이터 분석 중 오류: {e}")
            return False, None
    
    def step2_feature_engineering(self):
        """2단계: 피처 엔지니어링"""
        print("\n" + "="*50)
        print("2단계: 피처 엔지니어링")
        print("="*50)
        
        try:
            # 데이터 로드
            train_df = pd.read_csv('train.csv')
            test_df = pd.read_csv('test.csv')
            
            # 피처 엔지니어링
            engineer = FeatureEngineer()
            train_processed, test_processed = engineer.process_all_features(train_df, test_df)
            
            self.results['feature_engineering'] = {
                'original_features': train_df.shape[1],
                'engineered_features': train_processed.shape[1],
                'created_features': train_processed.shape[1] - train_df.shape[1]
            }
            
            print("피처 엔지니어링 완료!")
            return True, engineer, train_processed, test_processed
            
        except Exception as e:
            print(f"피처 엔지니어링 중 오류: {e}")
            return False, None, None, None
    
    def step3_preprocessing(self, train_df, test_df):
        """3단계: 데이터 전처리"""
        print("\n" + "="*50)
        print("3단계: 데이터 전처리")
        print("="*50)
        
        try:
            preprocessor = DataPreprocessor()
            train_final, test_final = preprocessor.process_complete_pipeline(train_df, test_df)
            
            # 모델링용 데이터 준비
            X_train, X_val, y_train, y_val, X_test, test_ids = preprocessor.prepare_model_data(
                train_final, test_final
            )
            
            self.results['preprocessing'] = {
                'train_shape': X_train.shape,
                'val_shape': X_val.shape,
                'test_shape': X_test.shape,
                'final_features': X_train.shape[1]
            }
            
            print("데이터 전처리 완료!")
            return True, preprocessor, X_train, X_val, y_train, y_val, X_test, test_ids
            
        except Exception as e:
            print(f"데이터 전처리 중 오류: {e}")
            return False, None, None, None, None, None, None, None
    
    def step4_validation(self, train_df, test_df):
        """4단계: 검증 시스템"""
        print("\n" + "="*50)
        print("4단계: 검증 및 리키지 탐지")
        print("="*50)
        
        try:
            validator = ValidationSystem()
            
            # 데이터 누수 탐지
            leakage_free = validator.detect_data_leakage(train_df, test_df)
            
            self.results['validation'] = {
                'leakage_free': leakage_free,
                'leakage_issues': validator.leakage_checks.get('issues', [])
            }
            
            if not leakage_free:
                print("경고: 데이터 누수 위험 탐지됨!")
            
            print("검증 시스템 완료!")
            return True, validator
            
        except Exception as e:
            print(f"검증 시스템 중 오류: {e}")
            return False, None
    
    def step5_model_training(self, feature_engineer, preprocessor):
        """5단계: 모델 학습"""
        print("\n" + "="*50)
        print("5단계: 모델 학습 및 최적화")
        print("="*50)
        
        try:
            trainer = ModelTrainer()
            
            # 데이터 준비 (다시 로드하여 일관성 유지)
            X_train, X_val, y_train, y_val, X_test, test_ids, _, _ = trainer.prepare_data()
            
            # 모든 모델 학습
            trainer.train_all_models()
            
            # 성능 확인
            best_cv_score = 0
            if trainer.cv_scores:
                best_cv_score = max([result['mean'] for result in trainer.cv_scores.values()])
            
            self.results['model_training'] = {
                'models_trained': len(trainer.models),
                'best_cv_score': best_cv_score,
                'target_achieved': best_cv_score >= self.target_accuracy
            }
            
            print(f"최고 교차 검증 점수: {best_cv_score:.4f}")
            
            if best_cv_score >= self.target_accuracy:
                print(f"✓ 목표 성능 ({self.target_accuracy:.1%}) 달성!")
            else:
                print(f"⚠ 목표 성능 미달 (목표: {self.target_accuracy:.1%}, 달성: {best_cv_score:.1%})")
            
            print("모델 학습 완료!")
            return True, trainer
            
        except Exception as e:
            print(f"모델 학습 중 오류: {e}")
            return False, None
    
    def step6_prediction(self):
        """6단계: 예측 및 제출"""
        print("\n" + "="*50)
        print("6단계: 예측 및 제출 파일 생성")
        print("="*50)
        
        try:
            predictor = PredictionSystem()
            submission_df = predictor.predict_test_data()
            
            if submission_df is not None:
                self.results['prediction'] = {
                    'submission_shape': submission_df.shape,
                    'prediction_distribution': submission_df['support_needs'].value_counts().to_dict()
                }
                
                print("예측 및 제출 파일 생성 완료!")
                return True, submission_df
            else:
                print("예측 실패!")
                return False, None
                
        except Exception as e:
            print(f"예측 중 오류: {e}")
            return False, None
    
    def generate_performance_report(self):
        """성능 보고서 생성"""
        print("\n" + "="*60)
        print("성능 분석 보고서")
        print("="*60)
        
        total_time = time.time() - self.start_time
        
        print(f"총 실행 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
        print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 주요 성과 지표
        print("\n주요 성과 지표:")
        
        # 피처 엔지니어링 효과
        if 'feature_engineering' in self.results:
            fe_result = self.results['feature_engineering']
            feature_expansion = fe_result['engineered_features'] / fe_result['original_features']
            print(f"  피처 확장률: {feature_expansion:.1f}x ({fe_result['original_features']} → {fe_result['engineered_features']})")
        
        # 모델 성능
        if 'model_training' in self.results:
            mt_result = self.results['model_training']
            print(f"  학습된 모델 수: {mt_result['models_trained']}개")
            print(f"  최고 CV 점수: {mt_result['best_cv_score']:.4f}")
            print(f"  목표 달성: {'✓' if mt_result['target_achieved'] else '✗'}")
        
        # 데이터 품질
        if 'validation' in self.results:
            val_result = self.results['validation']
            print(f"  데이터 누수: {'없음' if val_result['leakage_free'] else '위험'}")
        
        # 최종 예측 품질
        if 'prediction' in self.results:
            pred_result = self.results['prediction']
            distribution = pred_result['prediction_distribution']
            total_pred = sum(distribution.values())
            
            print("  예측 분포:")
            for cls in sorted(distribution.keys()):
                pct = distribution[cls] / total_pred * 100
                print(f"    클래스 {cls}: {pct:.1f}%")
    
    def generate_technical_report(self):
        """기술적 상세 보고서"""
        print("\n기술적 구현 세부사항:")
        
        print("  피처 엔지니어링:")
        print("    - 다항식 특성 (2차 교호작용)")
        print("    - 타겟 인코딩 (평균, 표준편차)")
        print("    - 클러스터링 기반 거리 피처")
        print("    - PCA 차원 축소 피처")
        print("    - 통계적 집계 피처")
        
        print("  전처리 기법:")
        print("    - Isolation Forest 이상치 탐지")
        print("    - 다중 스케일링 (Standard, Robust, Quantile)")
        print("    - 상호정보량 기반 피처 선택")
        print("    - 데이터 품질 검증")
        
        print("  모델링 전략:")
        print("    - 6개 다양한 알고리즘 앙상블")
        print("    - 하이퍼파라미터 베이지안 최적화")
        print("    - 스태킹 메타 학습")
        print("    - 예측 확률 보정")
        print("    - 클래스 균형 조정")
        
        print("  검증 방법:")
        print("    - 계층화 K-Fold 교차 검증")
        print("    - 데이터 누수 탐지 시스템")
        print("    - 안정성 검증")
        print("    - 분포 유사성 검증")
    
    def run_complete_system(self):
        """전체 시스템 실행"""
        try:
            # 환경 설정
            if not self.setup_environment():
                return False
            
            # 1단계: 데이터 분석
            success, analyzer = self.step1_data_analysis()
            if not success:
                return False
            
            # 2단계: 피처 엔지니어링
            success, engineer, train_df, test_df = self.step2_feature_engineering()
            if not success:
                return False
            
            # 3단계: 전처리
            success, preprocessor, X_train, X_val, y_train, y_val, X_test, test_ids = self.step3_preprocessing(train_df, test_df)
            if not success:
                return False
            
            # 4단계: 검증
            success, validator = self.step4_validation(train_df, test_df)
            if not success:
                return False
            
            # 5단계: 모델 학습
            success, trainer = self.step5_model_training(engineer, preprocessor)
            if not success:
                return False
            
            # 6단계: 예측
            success, submission_df = self.step6_prediction()
            if not success:
                return False
            
            # 성능 보고서
            self.generate_performance_report()
            self.generate_technical_report()
            
            print("\n" + "="*60)
            
            # 최종 결과 확인
            best_score = self.results.get('model_training', {}).get('best_cv_score', 0)
            
            if best_score >= self.target_accuracy:
                print("✓ AI 시스템 구축 성공!")
                print(f"✓ 목표 성능 ({self.target_accuracy:.1%}) 달성: {best_score:.1%}")
                print("✓ 제출 파일: submission.csv")
            else:
                print("⚠ AI 시스템 구축 완료 (성능 목표 미달)")
                print(f"⚠ 달성 성능: {best_score:.1%} (목표: {self.target_accuracy:.1%})")
                print("⚠ 추가 최적화 필요")
            
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"시스템 실행 중 치명적 오류: {e}")
            return False

def main():
    """메인 실행 함수"""
    # AI 시스템 인스턴스 생성
    ai_system = AISystem()
    
    # 전체 시스템 실행
    success = ai_system.run_complete_system()
    
    if success:
        print("\n프로그램이 성공적으로 완료되었습니다.")
        return 0
    else:
        print("\n프로그램 실행 중 오류가 발생했습니다.")
        return 1

if __name__ == "__main__":
    exit_code = main()