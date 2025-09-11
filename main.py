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
        self.target_accuracy = 0.50
        
    def setup_environment(self):
        """환경 설정"""
        print("AI 시스템 시작")
        print("=" * 40)
        print(f"Python 버전: {sys.version}")
        print(f"작업 디렉토리: {os.getcwd()}")
        print(f"목표 정확도: {self.target_accuracy}")
        
        # 필수 파일 확인
        required_files = ['train.csv', 'test.csv', 'sample_submission.csv']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"필수 파일 누락: {missing_files}")
            return False
        
        # 모델 디렉토리 생성
        os.makedirs('models', exist_ok=True)
        
        self.start_time = time.time()
        return True
    
    def step1_data_analysis(self):
        """데이터 분석"""
        print("\n1단계: 데이터 분석")
        print("=" * 30)
        
        try:
            analyzer = DataAnalyzer()
            analysis_results = analyzer.run_analysis()
            
            self.results['data_analysis'] = analysis_results
            
            # 시간적 누수 확인
            if 'temporal' in analysis_results:
                temporal_info = analysis_results['temporal']
                safe_ratio = temporal_info.get('safe_ratio', 1.0)
                if safe_ratio < 0.90:
                    print(f"주의: 시간적 안전 비율 {safe_ratio:.1%}")
            
            # 타겟 누수 확인
            if 'leakage' in analysis_results:
                leakage_info = analysis_results['leakage']
                if 'after_interaction' in leakage_info:
                    if leakage_info['after_interaction'].get('is_leakage', False):
                        print("주의: after_interaction 피처 누수 위험")
            
            # 데이터 무결성
            integrity_info = analysis_results.get('integrity', {})
            if not integrity_info.get('passed', True):
                print(f"주의: 데이터 무결성 문제 {len(integrity_info.get('issues', []))}개")
            
            print("데이터 분석 완료")
            return True, analyzer
            
        except Exception as e:
            print(f"데이터 분석 오류: {e}")
            self.results['data_analysis'] = {}
            return False, None
    
    def step2_feature_engineering(self):
        """피처 생성"""
        print("\n2단계: 피처 생성")
        print("=" * 30)
        
        try:
            train_df = pd.read_csv('train.csv')
            test_df = pd.read_csv('test.csv')
            
            if train_df.empty or test_df.empty:
                print("데이터 파일 비어있음")
                return False, None, None, None
            
            # 시간적 임계값 가져오기
            temporal_threshold = None
            if 'data_analysis' in self.results:
                temporal_info = self.results['data_analysis'].get('temporal', {})
                temporal_threshold = temporal_info.get('temporal_threshold')
            
            engineer = FeatureEngineer()
            train_processed, test_processed = engineer.create_features(train_df, test_df, temporal_threshold)
            
            if train_processed is None or test_processed is None:
                print("피처 생성 실패")
                return False, None, None, None
            
            original_features = train_df.shape[1] - 1  # ID 제외
            final_features = train_processed.shape[1] - 2  # ID, support_needs 제외
            
            self.results['feature_engineering'] = {
                'original_features': original_features,
                'final_features': final_features,
                'created_features': final_features - original_features,
                'temporal_threshold': temporal_threshold
            }
            
            print(f"피처: {original_features} → {final_features} (+{final_features - original_features})")
            return True, engineer, train_processed, test_processed
            
        except Exception as e:
            print(f"피처 생성 오류: {e}")
            return False, None, None, None
    
    def step3_preprocessing(self, train_df, test_df):
        """데이터 전처리"""
        print("\n3단계: 데이터 전처리")
        print("=" * 30)
        
        try:
            if train_df is None or test_df is None:
                print("입력 데이터 None")
                return False, None, None, None, None, None, None, None
            
            preprocessor = DataPreprocessor()
            
            # 시간적 정보 전달
            temporal_info = None
            if 'data_analysis' in self.results:
                temporal_info = self.results['data_analysis'].get('temporal')
            
            train_final, test_final = preprocessor.process_data(train_df, test_df, temporal_info)
            
            if train_final is None or test_final is None:
                print("전처리 실패")
                return False, None, None, None, None, None, None, None
            
            if train_final.empty or test_final.empty:
                print("전처리 데이터 비어있음")
                return False, None, None, None, None, None, None, None
            
            # 시간 기반 분할
            X_train, X_val, y_train, y_val, X_test, test_ids = preprocessor.prepare_data_temporal_optimized(
                train_final, test_final, val_size=0.18, gap_size=0.005
            )
            
            if X_train is None or X_val is None or y_train is None or y_val is None:
                print("데이터 분할 실패")
                return False, None, None, None, None, None, None, None
            
            if len(X_train) == 0 or len(X_val) == 0:
                print("분할 데이터 비어있음")
                return False, None, None, None, None, None, None, None
            
            self.results['preprocessing'] = {
                'train_shape': X_train.shape,
                'val_shape': X_val.shape,
                'test_shape': X_test.shape,
                'selected_features': len(X_train.columns)
            }
            
            print(f"훈련: {X_train.shape}, 검증: {X_val.shape}, 테스트: {X_test.shape}")
            return True, preprocessor, X_train, X_val, y_train, y_val, X_test, test_ids
            
        except Exception as e:
            print(f"데이터 전처리 오류: {e}")
            return False, None, None, None, None, None, None, None
    
    def step4_validation(self, X_train, y_train, X_val=None, y_val=None):
        """검증 시스템"""
        print("\n4단계: 검증 시스템")
        print("=" * 30)
        
        try:
            if X_train is None or y_train is None:
                print("검증 데이터 None")
                return False, None
            
            if len(X_train) == 0 or len(y_train) == 0:
                print("검증 데이터 비어있음")
                return False, None
            
            validator = ValidationSystem()
            validation_results = validator.validate_system(X_train, y_train, X_val, y_val)
            
            self.results['validation'] = validation_results
            
            # 검증 성능 확인
            overall_score = validation_results.get('overall_score', 0.0)
            holdout_score = validation_results.get('component_scores', {}).get('holdout_score', 0.0)
            cv_score = validation_results.get('component_scores', {}).get('cv_score', 0.0)
            stability_score = validation_results.get('component_scores', {}).get('stability_score', 0.0)
            
            print(f"홀드아웃: {holdout_score:.4f}")
            print(f"교차검증: {cv_score:.4f}")
            print(f"안정성: {stability_score:.4f}")
            print(f"종합 점수: {overall_score:.4f}")
            
            if overall_score >= self.target_accuracy:
                print("목표 성능 달성")
            else:
                gap = self.target_accuracy - overall_score
                print(f"목표까지: {gap:.4f}")
            
            print("검증 시스템 완료")
            return True, validator
            
        except Exception as e:
            print(f"검증 시스템 오류: {e}")
            self.results['validation'] = {'overall_score': 0.0}
            return False, None
    
    def step5_model_training(self, X_train, X_val, y_train, y_val, engineer, preprocessor):
        """모델 학습"""
        print("\n5단계: 모델 학습")
        print("=" * 30)
        
        try:
            if any(data is None for data in [X_train, X_val, y_train, y_val]):
                print("모델 학습 데이터 None")
                return False, None
            
            if any(len(data) == 0 for data in [X_train, X_val, y_train, y_val]):
                print("모델 학습 데이터 비어있음")
                return False, None
            
            trainer = ModelTrainer()
            trainer.feature_names = list(X_train.columns)
            trainer.calculate_class_weights(y_train)
            
            trainer.train_models(X_train, X_val, y_train, y_val, engineer, preprocessor)
            
            # 성능 확인
            best_score = 0.0
            best_model_name = None
            model_count = len(trainer.models)
            
            if trainer.models and model_count > 0:
                from sklearn.metrics import accuracy_score
                
                for model_name, model in trainer.models.items():
                    try:
                        X_val_clean = trainer.safe_data_conversion(X_val)
                        y_val_clean = trainer.safe_data_conversion(y_val)
                        
                        if model_name == 'lightgbm':
                            y_pred = model.predict(X_val_clean)
                            if y_pred.ndim == 2 and y_pred.shape[1] == 3:
                                y_pred_class = np.argmax(y_pred, axis=1)
                            else:
                                y_pred_class = np.clip(np.round(y_pred).astype(int), 0, 2)
                                
                        elif model_name == 'xgboost':
                            import xgboost as xgb
                            if trainer.feature_names:
                                xgb_test = xgb.DMatrix(X_val_clean, feature_names=trainer.feature_names)
                            else:
                                xgb_test = xgb.DMatrix(X_val_clean)
                            y_pred = model.predict(xgb_test)
                            if y_pred.ndim == 2 and y_pred.shape[1] == 3:
                                y_pred_class = np.argmax(y_pred, axis=1)
                            else:
                                y_pred_class = np.clip(np.round(y_pred).astype(int), 0, 2)
                                
                        elif model_name == 'catboost':
                            y_pred_class = model.predict(X_val_clean)
                            y_pred_class = np.clip(y_pred_class, 0, 2)
                            
                        elif model_name == 'stacking':
                            continue
                            
                        else:
                            y_pred_class = model.predict(X_val_clean)
                            y_pred_class = np.clip(y_pred_class, 0, 2)
                        
                        score = accuracy_score(y_val_clean, y_pred_class)
                        if score > best_score:
                            best_score = score
                            best_model_name = model_name
                            
                    except Exception as e:
                        continue
            
            self.results['model_training'] = {
                'models_count': model_count,
                'best_validation_score': best_score,
                'best_model': best_model_name,
                'target_achieved': best_score >= self.target_accuracy,
                'ensemble_weights': getattr(trainer, 'ensemble_weights', {})
            }
            
            if best_model_name:
                print(f"최고 성능: {best_score:.4f} ({best_model_name})")
                print(f"학습된 모델: {model_count}개")
            
            print("모델 학습 완료")
            return True, trainer
            
        except Exception as e:
            print(f"모델 학습 오류: {e}")
            self.results['model_training'] = {
                'models_count': 0,
                'best_validation_score': 0.0,
                'best_model': None,
                'target_achieved': False
            }
            return False, None
    
    def step6_prediction(self):
        """예측 생성"""
        print("\n6단계: 예측 생성")
        print("=" * 30)
        
        try:
            predictor = PredictionSystem()
            submission_df = predictor.generate_final_predictions()
            
            if submission_df is not None and not submission_df.empty:
                unique_classes = submission_df['support_needs'].unique()
                pred_counts = submission_df['support_needs'].value_counts().sort_index()
                
                print("예측 분포:")
                total_preds = len(submission_df)
                for cls in [0, 1, 2]:
                    count = pred_counts.get(cls, 0)
                    pct = count / total_preds * 100
                    print(f"클래스 {cls}: {count:,}개 ({pct:.1f}%)")
                
                if len(unique_classes) >= 2:
                    self.results['prediction'] = {
                        'submission_shape': submission_df.shape,
                        'prediction_counts': pred_counts.to_dict(),
                        'unique_classes': len(unique_classes),
                        'diversity_score': len(unique_classes) / 3.0,
                        'method': 'weighted_ensemble'
                    }
                    
                    print("예측 생성 완료")
                    return True, submission_df
                else:
                    print("예측 다양성 부족")
                    return self.fallback_prediction()
            else:
                print("예측 생성 실패")
                return self.fallback_prediction()
                
        except Exception as e:
            print(f"예측 생성 오류: {e}")
            return self.fallback_prediction()
    
    def fallback_prediction(self):
        """대체 예측"""
        print("대체 예측 실행")
        
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import cross_val_score
            
            train_df = pd.read_csv('train.csv')
            test_df = pd.read_csv('test.csv')
            
            # 피처 준비
            numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
            categorical_cols = ['gender', 'subscription_type']
            
            train_processed = train_df.copy()
            test_processed = test_df.copy()
            
            # 범주형 인코딩
            le = LabelEncoder()
            for col in categorical_cols:
                if col in train_df.columns and col in test_df.columns:
                    combined = pd.concat([train_df[col], test_df[col]])
                    le.fit(combined.fillna('Unknown'))
                    train_processed[col] = le.transform(train_df[col].fillna('Unknown'))
                    test_processed[col] = le.transform(test_df[col].fillna('Unknown'))
            
            # 추가 피처 생성
            if all(col in train_processed.columns for col in ['age', 'tenure']):
                train_processed['age_tenure_ratio'] = train_processed['age'] / (train_processed['tenure'] + 1)
                test_processed['age_tenure_ratio'] = test_processed['age'] / (test_processed['tenure'] + 1)
            
            if all(col in train_processed.columns for col in ['frequent', 'payment_interval']):
                train_processed['frequency_payment_ratio'] = train_processed['frequent'] / (train_processed['payment_interval'] + 1)
                test_processed['frequency_payment_ratio'] = test_processed['frequent'] / (test_processed['payment_interval'] + 1)
            
            # 피처 선택
            feature_cols = numeric_cols + categorical_cols + ['age_tenure_ratio', 'frequency_payment_ratio']
            feature_cols = [col for col in feature_cols if col in train_processed.columns and col in test_processed.columns]
            
            X = train_processed[feature_cols].fillna(0)
            y = train_processed['support_needs']
            X_test = test_processed[feature_cols].fillna(0)
            
            # 클래스 가중치 계산
            class_counts = np.bincount(y)
            total_samples = len(y)
            class_weights = {}
            
            for i, count in enumerate(class_counts):
                if count > 0:
                    class_weights[i] = total_samples / (len(class_counts) * count)
                else:
                    class_weights[i] = 1.0
            
            # 클래스 1 보정
            class_weights[1] *= 1.1
            class_weights[2] *= 1.05
            
            # 앙상블 모델 학습
            models = []
            
            # Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=4,
                max_features=0.8,
                class_weight=class_weights,
                random_state=42,
                n_jobs=-1
            )
            models.append(('rf', rf_model))
            
            # Gradient Boosting
            gb_model = GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.06,
                max_depth=7,
                min_samples_split=15,
                min_samples_leaf=8,
                subsample=0.85,
                random_state=42
            )
            models.append(('gb', gb_model))
            
            # 앙상블 예측
            ensemble_predictions = []
            model_weights = [0.6, 0.4]  # RF에 더 높은 가중치
            
            for i, (name, model) in enumerate(models):
                model.fit(X, y)
                pred_proba = model.predict_proba(X_test)
                ensemble_predictions.append(pred_proba * model_weights[i])
            
            # 가중 평균
            final_proba = np.sum(ensemble_predictions, axis=0)
            
            # 클래스 균형 조정
            class_adjustments = np.array([1.0, 1.05, 1.02])
            adjusted_proba = final_proba * class_adjustments[np.newaxis, :]
            normalized_proba = adjusted_proba / adjusted_proba.sum(axis=1, keepdims=True)
            
            predictions = np.argmax(normalized_proba, axis=1)
            
            # 분포 후처리
            pred_counts = np.bincount(predictions, minlength=3)
            total_preds = len(predictions)
            
            # 클래스 1이 너무 적으면 조정
            if pred_counts[1] < total_preds * 0.08:
                class_1_proba = normalized_proba[:, 1]
                top_indices = np.argsort(class_1_proba)[-int(total_preds * 0.08):]
                predictions[top_indices] = 1
            
            # 제출 파일
            submission_df = pd.DataFrame({
                'ID': test_processed['ID'],
                'support_needs': predictions.astype(int)
            })
            
            submission_df.to_csv('submission.csv', index=False)
            
            # 분포 출력
            final_counts = submission_df['support_needs'].value_counts().sort_index()
            print("대체 예측 분포:")
            for cls in [0, 1, 2]:
                count = final_counts.get(cls, 0)
                pct = count / len(submission_df) * 100
                print(f"클래스 {cls}: {count:,}개 ({pct:.1f}%)")
            
            self.results['prediction'] = {
                'submission_shape': submission_df.shape,
                'prediction_counts': final_counts.to_dict(),
                'method': 'fallback_ensemble'
            }
            
            print("대체 예측 완료")
            return True, submission_df
            
        except Exception as e:
            print(f"대체 예측 실패: {e}")
            return False, None
    
    def generate_report(self):
        """성과 보고서"""
        print("\n" + "=" * 50)
        print("최종 성과 보고서")
        print("=" * 50)
        
        try:
            total_time = time.time() - self.start_time if self.start_time else 0
            print(f"총 실행 시간: {total_time:.1f}초")
            
            # 데이터 분석 결과
            if 'data_analysis' in self.results:
                da = self.results['data_analysis']
                integrity = da.get('integrity', {})
                if integrity.get('passed', True):
                    print("데이터 무결성: 통과")
                else:
                    print(f"데이터 무결성: 문제 {len(integrity.get('issues', []))}개")
            
            # 피처 생성 결과
            if 'feature_engineering' in self.results:
                fe = self.results['feature_engineering']
                print(f"피처 확장: {fe['original_features']} → {fe['final_features']}")
                
                if fe.get('temporal_threshold'):
                    print(f"시간적 임계값: {fe['temporal_threshold']}")
            
            # 전처리 결과
            if 'preprocessing' in self.results:
                pp = self.results['preprocessing']
                print(f"최종 피처: {pp.get('selected_features', 0)}개")
            
            # 검증 결과
            if 'validation' in self.results:
                val = self.results['validation']
                overall_score = val.get('overall_score', 0.0)
                print(f"검증 점수: {overall_score:.4f}")
                
                if overall_score >= self.target_accuracy:
                    print("✓ 목표 정확도 달성")
                else:
                    gap = self.target_accuracy - overall_score
                    print(f"목표까지: {gap:.4f}")
            
            # 모델 학습 결과
            if 'model_training' in self.results:
                mt = self.results['model_training']
                print(f"학습 모델: {mt['models_count']}개")
                print(f"최고 검증 성능: {mt['best_validation_score']:.4f}")
                
                if mt.get('best_model'):
                    print(f"최고 모델: {mt['best_model']}")
                
                if mt['target_achieved']:
                    print("✓ 모델 목표 달성")
            
            # 예측 결과
            if 'prediction' in self.results:
                pred = self.results['prediction']
                print("예측 분포:")
                total_predictions = sum(pred['prediction_counts'].values())
                for cls in [0, 1, 2]:
                    count = pred['prediction_counts'].get(cls, 0)
                    pct = count / total_predictions * 100 if total_predictions > 0 else 0
                    print(f"  클래스 {cls}: {pct:.1f}%")
                
                diversity_score = pred.get('diversity_score', 0)
                print(f"예측 다양성: {diversity_score:.2f}")
                
                if 'method' in pred:
                    print(f"예측 방법: {pred['method']}")
            
            # 전체 성공률
            total_steps = 6
            completed_steps = sum(1 for step in ['data_analysis', 'feature_engineering', 'preprocessing', 'validation', 'model_training', 'prediction'] if step in self.results)
            success_rate = completed_steps / total_steps * 100
            
            print(f"\n단계 완료율: {completed_steps}/{total_steps} ({success_rate:.1f}%)")
            
            # 성능 등급
            if 'validation' in self.results and 'model_training' in self.results:
                val_score = self.results['validation'].get('overall_score', 0.0)
                model_achieved = self.results['model_training'].get('target_achieved', False)
                
                if val_score >= self.target_accuracy and model_achieved:
                    print("✓ 목표 성능 달성")
                elif val_score >= self.target_accuracy * 0.95:
                    print("→ 목표 근접")
                elif success_rate >= 83:
                    print("→ 파이프라인 안정")
                else:
                    print("→ 부분 성공")
            
        except Exception as e:
            print(f"보고서 생성 오류: {e}")
    
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
            if not success or train_df is None or test_df is None:
                print("2단계 실패 - 대체 예측")
                fallback_success, fallback_result = self.fallback_prediction()
                if fallback_success:
                    self.generate_report()
                    return True
                else:
                    print("대체 예측 실패")
                    return False
            
            # 3단계: 전처리
            success, preprocessor, X_train, X_val, y_train, y_val, X_test, test_ids = self.step3_preprocessing(train_df, test_df)
            if not success or any(data is None for data in [X_train, X_val, y_train, y_val]):
                print("3단계 실패 - 대체 예측")
                fallback_success, fallback_result = self.fallback_prediction()
                if fallback_success:
                    self.generate_report()
                    return True
                else:
                    print("대체 예측 실패")
                    return False
            
            # 4단계: 검증
            success, validator = self.step4_validation(X_train, y_train, X_val, y_val)
            if not success:
                print("4단계 실패 - 계속 진행")
            
            # 5단계: 모델 학습
            success, trainer = self.step5_model_training(X_train, X_val, y_train, y_val, engineer, preprocessor)
            if not success:
                print("5단계 실패 - 계속 진행")
            
            # 6단계: 예측 생성
            success, submission_df = self.step6_prediction()
            if not success:
                print("6단계 실패 - 최종 대체")
                final_success, final_result = self.fallback_prediction()
                if not final_success:
                    print("최종 대체 실패")
                    return False
            
            # 보고서 생성
            self.generate_report()
            
            print(f"\n{'='*50}")
            print("AI 시스템 구축 완료")
            print(f"{'='*50}")
            return True
            
        except Exception as e:
            print(f"시스템 실행 예외: {e}")
            
            try:
                fallback_success, fallback_result = self.fallback_prediction()
                if fallback_success:
                    print("대체 모드 성공")
                    return True
                else:
                    print("대체 모드 실패")
                    return False
            except Exception as fallback_e:
                print(f"대체 모드 예외: {fallback_e}")
                return False

def main():
    """메인 함수"""
    ai_system = AISystem()
    
    try:
        success = ai_system.run_system()
        
        if success:
            print("\n프로그램 정상 완료")
            return 0
        else:
            print("\n프로그램 실행 실패")
            return 1
            
    except Exception as e:
        print(f"\n메인 함수 예외: {e}")
        print("프로그램 실행 실패")
        return 1

if __name__ == "__main__":
    exit_code = main()