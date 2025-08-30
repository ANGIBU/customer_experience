# main.py

import os
import sys
import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data_analysis import DataAnalyzer
from feature_engineering import FeatureEngineer
from preprocessing import DataPreprocessor
from model_training import ModelTrainer
from validation import ValidationSystem
from prediction import PredictionSystem

class AISystem:
    def __init__(self):
        self.start_time = None
        self.results = {}
        self.target_accuracy = 0.55
        
    def setup_environment(self):
        print("AI 시스템 초기화")
        print("=" * 40)
        print(f"Python 버전: {sys.version}")
        print(f"작업 디렉토리: {os.getcwd()}")
        print(f"목표 정확도: {self.target_accuracy}")
        
        required_files = ['train.csv', 'test.csv', 'sample_submission.csv']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"필수 파일 누락: {missing_files}")
            return False
        
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        self.start_time = time.time()
        return True
    
    def step1_data_analysis(self):
        print("\n1단계: 데이터 분석")
        print("=" * 30)
        
        try:
            analyzer = DataAnalyzer()
            analysis_results = analyzer.run_analysis()
            
            self.results['data_analysis'] = analysis_results
            print("데이터 분석 완료")
            return True, analyzer
            
        except Exception as e:
            print(f"데이터 분석 오류: {e}")
            return False, None
    
    def step2_feature_engineering(self):
        print("\n2단계: 피처 생성")
        print("=" * 30)
        
        try:
            train_df = pd.read_csv('train.csv')
            test_df = pd.read_csv('test.csv')
            
            engineer = FeatureEngineer()
            train_processed, test_processed = engineer.create_features(train_df, test_df)
            
            self.results['feature_engineering'] = {
                'original_features': train_df.shape[1] - 1,
                'final_features': train_processed.shape[1] - 2,
                'created_features': train_processed.shape[1] - train_df.shape[1]
            }
            
            print("피처 생성 완료")
            return True, engineer, train_processed, test_processed
            
        except Exception as e:
            print(f"피처 생성 오류: {e}")
            return False, None, None, None
    
    def step3_preprocessing(self, train_df, test_df):
        print("\n3단계: 데이터 전처리")
        print("=" * 30)
        
        try:
            preprocessor = DataPreprocessor()
            train_final, test_final = preprocessor.process_data(train_df, test_df)
            
            X_train, X_val, y_train, y_val, X_test, test_ids = preprocessor.prepare_data(
                train_final, test_final
            )
            
            self.results['preprocessing'] = {
                'train_shape': X_train.shape,
                'val_shape': X_val.shape,
                'test_shape': X_test.shape
            }
            
            print("데이터 전처리 완료")
            return True, preprocessor, X_train, X_val, y_train, y_val, X_test, test_ids
            
        except Exception as e:
            print(f"데이터 전처리 오류: {e}")
            return False, None, None, None, None, None, None, None
    
    def step4_validation(self, X_train, y_train):
        print("\n4단계: 검증 시스템")
        print("=" * 30)
        
        try:
            validator = ValidationSystem()
            validation_results = validator.validate_system(X_train, y_train)
            
            self.results['validation'] = validation_results
            print("검증 시스템 완료")
            return True, validator
            
        except Exception as e:
            print(f"검증 시스템 오류: {e}")
            return False, None
    
    def step5_model_training(self, X_train, X_val, y_train, y_val):
        print("\n5단계: 모델 학습")
        print("=" * 30)
        
        try:
            trainer = ModelTrainer()
            trainer.train_models(X_train, X_val, y_train, y_val)
            
            best_score = max([score['mean'] for score in trainer.cv_scores.values()])
            
            self.results['model_training'] = {
                'models_count': len(trainer.models),
                'best_cv_score': best_score,
                'target_achieved': best_score >= self.target_accuracy
            }
            
            print(f"최고 CV 점수: {best_score:.4f}")
            print("모델 학습 완료")
            return True, trainer
            
        except Exception as e:
            print(f"모델 학습 오류: {e}")
            return False, None
    
    def step6_prediction(self):
        print("\n6단계: 예측 생성")
        print("=" * 30)
        
        try:
            predictor = PredictionSystem()
            submission_df = predictor.generate_predictions()
            
            if submission_df is not None:
                self.results['prediction'] = {
                    'submission_shape': submission_df.shape,
                    'prediction_counts': submission_df['support_needs'].value_counts().to_dict()
                }
                
                print("예측 생성 완료")
                return True, submission_df
            else:
                print("예측 생성 실패")
                return False, None
                
        except Exception as e:
            print(f"예측 생성 오류: {e}")
            return False, None
    
    def generate_report(self):
        print("\n" + "=" * 50)
        print("최종 성과 보고서")
        print("=" * 50)
        
        total_time = time.time() - self.start_time
        print(f"총 실행 시간: {total_time:.1f}초")
        
        if 'feature_engineering' in self.results:
            fe = self.results['feature_engineering']
            print(f"피처 확장: {fe['original_features']} → {fe['final_features']}")
        
        if 'model_training' in self.results:
            mt = self.results['model_training']
            print(f"학습 모델 수: {mt['models_count']}")
            print(f"최고 CV 점수: {mt['best_cv_score']:.4f}")
            
            if mt['target_achieved']:
                print("목표 성능 달성")
            else:
                print("목표 성능 미달")
        
        if 'prediction' in self.results:
            pred = self.results['prediction']
            print("예측 분포:")
            for cls, count in pred['prediction_counts'].items():
                pct = count / sum(pred['prediction_counts'].values()) * 100
                print(f"  클래스 {cls}: {pct:.1f}%")
    
    def run_system(self):
        try:
            if not self.setup_environment():
                return False
            
            success, analyzer = self.step1_data_analysis()
            if not success:
                return False
            
            success, engineer, train_df, test_df = self.step2_feature_engineering()
            if not success:
                return False
            
            success, preprocessor, X_train, X_val, y_train, y_val, X_test, test_ids = self.step3_preprocessing(train_df, test_df)
            if not success:
                return False
            
            success, validator = self.step4_validation(X_train, y_train)
            if not success:
                return False
            
            success, trainer = self.step5_model_training(X_train, X_val, y_train, y_val)
            if not success:
                return False
            
            success, submission_df = self.step6_prediction()
            if not success:
                return False
            
            self.generate_report()
            
            print("\nAI 시스템 구축 완료")
            return True
            
        except Exception as e:
            print(f"시스템 실행 오류: {e}")
            return False

def main():
    ai_system = AISystem()
    success = ai_system.run_system()
    
    if success:
        print("프로그램 정상 완료")
        return 0
    else:
        print("프로그램 실행 실패")
        return 1

if __name__ == "__main__":
    exit_code = main()