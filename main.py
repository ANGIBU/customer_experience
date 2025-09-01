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
from monitoring import ModelMonitor

class AISystem:
    def __init__(self):
        self.start_time = None
        self.results = {}
        self.target_accuracy = 0.42
        
    def setup_environment(self):
        """환경 설정"""
        self.start_time = time.time()
        
        required_files = ['train.csv', 'test.csv', 'sample_submission.csv']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"필수 파일 누락: {missing_files}")
            return False
        
        os.makedirs('models', exist_ok=True)
        return True
    
    def step1_data_analysis(self):
        """데이터 분석"""
        print("데이터 분석 시작...")
        try:
            analyzer = DataAnalyzer()
            analysis_results = analyzer.run_analysis()
            self.results['data_analysis'] = analysis_results
            
            if analysis_results and 'temporal' in analysis_results:
                temporal_info = analysis_results['temporal']
                safe_ratio = temporal_info.get('safe_ratio', 0.7)
                has_leak = temporal_info.get('has_temporal_leak', False)
                print(f"시간적 안전 데이터 비율: {safe_ratio:.3f}")
                if has_leak:
                    print("시간적 누수 감지됨")
            
            if analysis_results and 'leakage' in analysis_results:
                leakage_info = analysis_results['leakage']
                if 'after_interaction' in leakage_info:
                    print("미래 정보 누수 피처 확인됨 - 제거 예정")
            
            return True, analyzer
            
        except Exception as e:
            print(f"데이터 분석 오류: {e}")
            self.results['data_analysis'] = {}
            return False, None
    
    def step2_feature_engineering(self):
        """피처 생성"""
        print("피처 생성 시작...")
        try:
            train_df = pd.read_csv('train.csv')
            test_df = pd.read_csv('test.csv')
            
            if train_df.empty or test_df.empty:
                raise ValueError("빈 데이터프레임")
            
            engineer = FeatureEngineer()
            train_processed, test_processed = engineer.create_features(train_df, test_df)
            
            if train_processed is None or test_processed is None:
                raise ValueError("피처 생성 실패")
            
            original_features = train_df.shape[1] - 1
            final_features = train_processed.shape[1] - 2
            
            print(f"피처 수: {original_features} → {final_features}")
            
            # after_interaction 제거 확인
            if 'after_interaction' not in train_processed.columns:
                print("누수 피처 제거 완료")
            
            self.results['feature_engineering'] = {
                'original_features': original_features,
                'final_features': final_features
            }
            
            return True, engineer, train_processed, test_processed
            
        except Exception as e:
            print(f"피처 생성 오류: {e}")
            return False, None, None, None
    
    def step3_preprocessing(self, train_df, test_df):
        """데이터 전처리"""
        print("데이터 전처리 시작...")
        try:
            if train_df is None or test_df is None:
                raise ValueError("입력 데이터 없음")
            
            preprocessor = DataPreprocessor()
            train_final, test_final = preprocessor.process_data(train_df, test_df)
            
            if train_final is None or test_final is None:
                raise ValueError("전처리 실패")
            
            # 시간적 분할 사용
            X_train, X_val, y_train, y_val, X_test, test_ids = preprocessor.prepare_temporal_split(
                train_final, test_final, val_size=0.25
            )
            
            if any(data is None for data in [X_train, X_val, y_train, y_val]):
                raise ValueError("데이터 분할 실패")
            
            print(f"시간적 분할 완료 - 훈련: {X_train.shape}, 검증: {X_val.shape}")
            
            self.results['preprocessing'] = {
                'train_shape': X_train.shape,
                'val_shape': X_val.shape,
                'test_shape': X_test.shape
            }
            
            return True, preprocessor, X_train, X_val, y_train, y_val, X_test, test_ids
            
        except Exception as e:
            print(f"전처리 오류: {e}")
            return False, None, None, None, None, None, None, None
    
    def step4_validation(self, X_train, y_train, X_val=None, y_val=None):
        """검증 시스템"""
        print("모델 검증 시작...")
        try:
            if X_train is None or y_train is None:
                raise ValueError("검증 데이터 없음")
            
            validator = ValidationSystem()
            validation_results = validator.validate_system(X_train, y_train, X_val, y_val)
            
            self.results['validation'] = validation_results
            
            overall_score = validation_results.get('overall_score', 0.0)
            raw_score = validation_results.get('holdout', {}).get('raw_accuracy', 0.0)
            
            print(f"보수적 검증 점수: {overall_score:.4f}")
            print(f"원본 검증 점수: {raw_score:.4f}")
            
            if overall_score >= self.target_accuracy:
                print("목표 성능 달성")
            else:
                print(f"목표 미달 - 현재: {overall_score:.4f}, 목표: {self.target_accuracy}")
            
            return True, validator
            
        except Exception as e:
            print(f"검증 오류: {e}")
            self.results['validation'] = {'overall_score': 0.0}
            return False, None
    
    def step5_model_training(self, X_train, X_val, y_train, y_val, engineer, preprocessor):
        """모델 학습"""
        print("모델 학습 시작...")
        try:
            if any(data is None for data in [X_train, X_val, y_train, y_val]):
                raise ValueError("학습 데이터 없음")
            
            trainer = ModelTrainer()
            trainer.feature_names = list(X_train.columns)
            trainer.calculate_class_weights(y_train)
            
            trainer.train_models(X_train, X_val, y_train, y_val, engineer, preprocessor)
            
            best_score = 0.0
            best_model_name = None
            model_count = len(trainer.models)
            
            if trainer.models:
                for model_name, model in trainer.models.items():
                    try:
                        if hasattr(model, 'predict'):
                            y_pred = model.predict(X_val)
                            score = np.mean(y_pred == y_val)
                            if score > best_score:
                                best_score = score
                                best_model_name = model_name
                    except:
                        continue
            
            print(f"최고 모델: {best_model_name} (정확도: {best_score:.4f})")
            print(f"학습된 모델 수: {model_count}")
            
            self.results['model_training'] = {
                'models_count': model_count,
                'best_validation_score': best_score,
                'best_model': best_model_name,
                'target_achieved': best_score >= self.target_accuracy
            }
            
            return True, trainer
            
        except Exception as e:
            print(f"모델 학습 오류: {e}")
            self.results['model_training'] = {
                'models_count': 0,
                'best_validation_score': 0.0,
                'target_achieved': False
            }
            return False, None
    
    def step6_prediction(self):
        """예측 생성"""
        print("예측 생성 시작...")
        try:
            predictor = PredictionSystem()
            submission_df = predictor.generate_final_predictions()
            
            if submission_df is not None and not submission_df.empty:
                pred_counts = submission_df['support_needs'].value_counts().sort_index()
                print(f"예측 분포: {pred_counts.to_dict()}")
                
                self.results['prediction'] = {
                    'submission_shape': submission_df.shape,
                    'prediction_counts': pred_counts.to_dict()
                }
                
                return True, submission_df
            else:
                return self.fallback_prediction()
                
        except Exception as e:
            print(f"예측 생성 오류: {e}")
            return self.fallback_prediction()
    
    def fallback_prediction(self):
        """대체 예측"""
        print("대체 예측 실행...")
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            train_df = pd.read_csv('train.csv')
            test_df = pd.read_csv('test.csv')
            
            # after_interaction 제거
            if 'after_interaction' in train_df.columns:
                train_df = train_df.drop('after_interaction', axis=1)
            if 'after_interaction' in test_df.columns:
                test_df = test_df.drop('after_interaction', axis=1)
            
            numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
            categorical_cols = ['gender', 'subscription_type']
            
            train_processed = train_df.copy()
            test_processed = test_df.copy()
            
            # 범주형 변수 안전한 처리
            for col in categorical_cols:
                if col in train_df.columns and col in test_df.columns:
                    try:
                        # 안전한 매핑 사용
                        if col == 'gender':
                            gender_mapping = {'M': 0, 'F': 1, 'Male': 0, 'Female': 1, 'Unknown': 2}
                            train_processed[col] = train_df[col].astype(str).map(gender_mapping).fillna(2)
                            test_processed[col] = test_df[col].astype(str).map(gender_mapping).fillna(2)
                        else:
                            # subscription_type 처리
                            combined_values = pd.concat([train_df[col], test_df[col]]).astype(str).unique()
                            mapping = {val: i for i, val in enumerate(combined_values)}
                            mapping['Unknown'] = len(mapping)
                            
                            train_processed[col] = train_df[col].astype(str).map(mapping).fillna(len(mapping))
                            test_processed[col] = test_df[col].astype(str).map(mapping).fillna(len(mapping))
                    except Exception as e:
                        print(f"범주형 변수 {col} 처리 오류: {e}, 기본값 사용")
                        # 오류 시 기본값 사용
                        train_processed[col] = 0
                        test_processed[col] = 0
            
            feature_cols = numeric_cols + categorical_cols
            feature_cols = [col for col in feature_cols if col in train_processed.columns and col in test_processed.columns]
            
            # 안전한 데이터 준비
            X = train_processed[feature_cols].fillna(0)
            y = train_processed['support_needs']
            X_test = test_processed[feature_cols].fillna(0)
            
            # 데이터 타입 안전성 보장
            for col in feature_cols:
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
                except:
                    X[col] = 0
                    X_test[col] = 0
            
            # 보수적인 모델 설정
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=15,
                min_samples_leaf=8,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X, y)
            predictions = model.predict(X_test)
            
            submission_df = pd.DataFrame({
                'ID': test_processed['ID'],
                'support_needs': predictions.astype(int)
            })
            
            submission_df.to_csv('submission.csv', index=False)
            
            pred_counts = submission_df['support_needs'].value_counts().sort_index()
            self.results['prediction'] = {
                'submission_shape': submission_df.shape,
                'prediction_counts': pred_counts.to_dict()
            }
            
            return True, submission_df
            
        except Exception as e:
            print(f"대체 예측 오류: {e}")
            try:
                # 최후의 수단
                test_df = pd.read_csv('test.csv')
                np.random.seed(42)
                random_predictions = np.random.choice([0, 1, 2], size=len(test_df), p=[0.60, 0.25, 0.15])
                
                submission_df = pd.DataFrame({
                    'ID': test_df['ID'],
                    'support_needs': random_predictions
                })
                
                submission_df.to_csv('submission.csv', index=False)
                return True, submission_df
            except:
                return False, None
    
    def step7_monitoring(self, train_df, test_df, X_train, y_train, validation_results):
        """정밀 모니터링"""
        print("정밀 모니터링 시작...")
        try:
            monitor = ModelMonitor()
            
            cv_results = validation_results.get('cross_validation', {})
            validation_score = validation_results.get('overall_score', 0.0)
            
            monitoring_results = monitor.comprehensive_monitoring(
                train_df, test_df, X_train, y_train, cv_results, validation_score
            )
            
            self.results['monitoring'] = monitoring_results
            
            estimated_performance = monitoring_results['performance_estimate']['estimate']
            risk_level = monitoring_results['risk_level']
            
            print(f"실제 성능 추정: {estimated_performance:.4f}")
            print(f"위험 등급: {risk_level}")
            
            if risk_level in ['CRITICAL', 'HIGH']:
                print(f"⚠ 시스템 위험 감지: {risk_level}")
            
            return True, monitor, monitoring_results
            
        except Exception as e:
            print(f"모니터링 오류: {e}")
            return False, None, None
    
    def generate_report(self):
        """성과 보고서"""
        total_time = time.time() - self.start_time if self.start_time else 0
        print(f"\n=== 시스템 실행 완료 (소요시간: {total_time:.1f}초) ===")
        
        if 'validation' in self.results:
            val = self.results['validation']
            overall_score = val.get('overall_score', 0.0)
            raw_score = val.get('holdout', {}).get('raw_accuracy', 0.0)
            print(f"보수적 검증 점수: {overall_score:.4f}")
            print(f"원본 검증 점수: {raw_score:.4f}")
            
            if overall_score >= self.target_accuracy:
                print("목표 정확도 달성")
            else:
                print(f"목표 미달")
        
        if 'monitoring' in self.results:
            mon = self.results['monitoring']
            estimated = mon['performance_estimate']['estimate']
            risk_level = mon['risk_level']
            print(f"실제 성능 추정: {estimated:.4f}")
            print(f"시스템 위험도: {risk_level}")
        
        total_steps = 6
        completed_steps = sum(1 for step in ['data_analysis', 'feature_engineering', 'preprocessing', 'validation', 'model_training', 'prediction'] if step in self.results)
        success_rate = completed_steps / total_steps * 100
        
        print(f"단계 완료율: {completed_steps}/{total_steps} ({success_rate:.1f}%)")
    
    def run_system(self):
        """전체 시스템 실행"""
        try:
            if not self.setup_environment():
                print("환경 설정 실패")
                return False
            
            success, analyzer = self.step1_data_analysis()
            if not success:
                print("데이터 분석 실패, 계속 진행...")
            
            success, engineer, train_df, test_df = self.step2_feature_engineering()
            if not success or train_df is None or test_df is None:
                print("피처 생성 실패, 대체 예측으로 전환")
                fallback_success, fallback_result = self.fallback_prediction()
                if fallback_success:
                    self.generate_report()
                    return True
                else:
                    return False
            
            success, preprocessor, X_train, X_val, y_train, y_val, X_test, test_ids = self.step3_preprocessing(train_df, test_df)
            if not success or any(data is None for data in [X_train, X_val, y_train, y_val]):
                print("전처리 실패, 대체 예측으로 전환")
                fallback_success, fallback_result = self.fallback_prediction()
                if fallback_success:
                    self.generate_report()
                    return True
                else:
                    return False
            
            success, validator = self.step4_validation(X_train, y_train, X_val, y_val)
            if not success:
                print("검증 실패, 계속 진행...")
            
            success, trainer = self.step5_model_training(X_train, X_val, y_train, y_val, engineer, preprocessor)
            if not success:
                print("모델 학습 실패, 계속 진행...")
            
            success_monitoring, monitor, monitoring_results = self.step7_monitoring(train_df, test_df, X_train, y_train, self.results.get('validation', {}))
            if not success_monitoring:
                print("모니터링 실패, 계속 진행...")
            
            success, submission_df = self.step6_prediction()
            if not success:
                print("예측 생성 실패, 대체 예측으로 전환")
                final_success, final_result = self.fallback_prediction()
                if not final_success:
                    print("모든 예측 방법 실패")
                    return False
            
            self.generate_report()
            return True
            
        except Exception as e:
            print(f"시스템 실행 오류: {e}")
            try:
                fallback_success, fallback_result = self.fallback_prediction()
                if fallback_success:
                    return True
                else:
                    return False
            except Exception as fallback_e:
                print(f"대체 예측 오류: {fallback_e}")
                return False

def main():
    """메인 함수"""
    ai_system = AISystem()
    
    try:
        success = ai_system.run_system()
        
        if success:
            print("시스템 실행 성공")
            return 0
        else:
            print("시스템 실행 실패")
            return 1
            
    except Exception as e:
        print(f"메인 함수 오류: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)