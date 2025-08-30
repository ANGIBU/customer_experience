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
        """환경 설정"""
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
        """1단계: 데이터 분석"""
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
        """2단계: 피처 생성"""
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
        """3단계: 데이터 전처리"""
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
        """4단계: 검증 시스템"""
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
        """5단계: 모델 학습"""
        print("\n5단계: 모델 학습")
        print("=" * 30)
        
        try:
            trainer = ModelTrainer()
            trainer.train_models(X_train, X_val, y_train, y_val)
            
            # 학습된 모델들의 성능 확인
            if trainer.models:
                # 간단한 성능 측정
                from sklearn.metrics import accuracy_score
                best_score = 0.0
                
                for model_name, model in trainer.models.items():
                    try:
                        if model_name == 'lightgbm':
                            y_pred = model.predict(X_val)
                            y_pred_class = np.argmax(y_pred, axis=1)
                        elif model_name == 'xgboost':
                            import xgboost as xgb
                            y_pred = model.predict(xgb.DMatrix(X_val))
                            y_pred_class = np.argmax(y_pred, axis=1)
                        else:
                            y_pred_class = model.predict(X_val)
                        
                        score = accuracy_score(y_val, y_pred_class)
                        if score > best_score:
                            best_score = score
                            
                    except Exception as e:
                        print(f"{model_name} 평가 오류: {e}")
                        continue
            else:
                best_score = 0.0
            
            self.results['model_training'] = {
                'models_count': len(trainer.models),
                'best_cv_score': best_score,
                'target_achieved': best_score >= self.target_accuracy
            }
            
            print(f"최고 성능: {best_score:.4f}")
            print("모델 학습 완료")
            return True, trainer
            
        except Exception as e:
            print(f"모델 학습 오류: {e}")
            return False, None
    
    def step6_prediction(self):
        """6단계: 예측 생성"""
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
        """성과 보고서 생성"""
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
            print(f"최고 성능: {mt['best_cv_score']:.4f}")
            
            if mt['target_achieved']:
                print("✓ 목표 성능 달성")
            else:
                print("✗ 목표 성능 미달")
        
        if 'prediction' in self.results:
            pred = self.results['prediction']
            print("예측 분포:")
            for cls, count in pred['prediction_counts'].items():
                pct = count / sum(pred['prediction_counts'].values()) * 100
                print(f"  클래스 {cls}: {pct:.1f}%")
        
        # 전체 단계 성공 여부
        total_steps = 6
        completed_steps = len(self.results)
        success_rate = completed_steps / total_steps * 100
        
        print(f"\n단계별 완료율: {completed_steps}/{total_steps} ({success_rate:.1f}%)")
        
        if success_rate == 100:
            print("✓ 전체 파이프라인 성공")
        else:
            print("✗ 일부 단계 실패")
    
    def run_system(self):
        """전체 시스템 실행"""
        try:
            # 환경 설정
            if not self.setup_environment():
                print("환경 설정 실패")
                return False
            
            # 1단계: 데이터 분석
            success, analyzer = self.step1_data_analysis()
            if not success:
                print("1단계 실패 - 계속 진행")
            
            # 2단계: 피처 생성
            success, engineer, train_df, test_df = self.step2_feature_engineering()
            if not success:
                print("2단계 실패 - 시스템 종료")
                return False
            
            # 3단계: 데이터 전처리
            success, preprocessor, X_train, X_val, y_train, y_val, X_test, test_ids = self.step3_preprocessing(train_df, test_df)
            if not success:
                print("3단계 실패 - 시스템 종료")
                return False
            
            # 4단계: 검증 시스템
            success, validator = self.step4_validation(X_train, y_train)
            if not success:
                print("4단계 실패 - 계속 진행")
            
            # 5단계: 모델 학습
            success, trainer = self.step5_model_training(X_train, X_val, y_train, y_val)
            if not success:
                print("5단계 실패 - 계속 진행")
            
            # 6단계: 예측 생성
            success, submission_df = self.step6_prediction()
            if not success:
                print("6단계 실패 - 시스템 종료")
                return False
            
            # 성과 보고서
            self.generate_report()
            
            print("\nAI 시스템 구축 완료")
            return True
            
        except Exception as e:
            print(f"시스템 실행 중 예외 발생: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def emergency_prediction(self):
        """비상 예측 생성"""
        print("비상 예측 모드 실행")
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # 기본 데이터 로드
            train_df = pd.read_csv('train.csv')
            test_df = pd.read_csv('test.csv')
            
            # 간단한 전처리
            numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
            categorical_cols = ['gender', 'subscription_type']
            
            # 결측치 처리
            for col in numeric_cols:
                if col in train_df.columns:
                    train_df[col].fillna(train_df[col].median(), inplace=True)
                    if col in test_df.columns:
                        test_df[col].fillna(train_df[col].median(), inplace=True)
            
            # 범주형 인코딩
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                if col in train_df.columns and col in test_df.columns:
                    le = LabelEncoder()
                    combined = pd.concat([train_df[col], test_df[col]])
                    le.fit(combined)
                    train_df[col] = le.transform(train_df[col])
                    test_df[col] = le.transform(test_df[col])
            
            # 피처와 타겟 준비
            feature_cols = numeric_cols + categorical_cols
            feature_cols = [col for col in feature_cols if col in train_df.columns and col in test_df.columns]
            
            X = train_df[feature_cols]
            y = train_df['support_needs']
            X_test = test_df[feature_cols]
            
            # 모델 학습
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X, y)
            predictions = model.predict(X_test)
            
            # 제출 파일 생성
            submission_df = pd.DataFrame({
                'ID': test_df['ID'],
                'support_needs': predictions
            })
            
            submission_df.to_csv('emergency_submission.csv', index=False)
            
            print("비상 예측 완료: emergency_submission.csv")
            return True
            
        except Exception as e:
            print(f"비상 예측 실패: {e}")
            return False

def main():
    """메인 함수"""
    ai_system = AISystem()
    
    # 정상 시스템 실행
    success = ai_system.run_system()
    
    if success:
        print("\n프로그램 정상 완료")
        return 0
    else:
        print("\n정상 시스템 실패 - 비상 모드 시도")
        
        # 비상 예측 모드
        emergency_success = ai_system.emergency_prediction()
        
        if emergency_success:
            print("비상 모드 성공")
            return 0
        else:
            print("프로그램 실행 실패")
            return 1

if __name__ == "__main__":
    exit_code = main()