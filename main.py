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
            best_score = 0.0
            best_model_name = None
            
            if trainer.models:
                from sklearn.metrics import accuracy_score
                
                for model_name, model in trainer.models.items():
                    try:
                        # 데이터 전처리
                        X_val_clean = X_val.fillna(0).replace([np.inf, -np.inf], 0)
                        
                        if model_name == 'lightgbm':
                            y_pred = model.predict(X_val_clean.values)
                            y_pred_class = np.argmax(y_pred, axis=1)
                        elif model_name == 'xgboost':
                            import xgboost as xgb
                            if trainer.feature_names:
                                xgb_test = xgb.DMatrix(X_val_clean.values, feature_names=trainer.feature_names)
                            else:
                                xgb_test = xgb.DMatrix(X_val_clean.values)
                            y_pred = model.predict(xgb_test)
                            y_pred_class = np.argmax(y_pred, axis=1)
                        elif model_name == 'catboost':
                            y_pred_class = model.predict(X_val_clean.values)
                        elif model_name in ['stacking', 'voting']:
                            # 앙상블 모델은 건너뛰기 (기본 모델 의존)
                            continue
                        else:
                            # sklearn 모델들
                            y_pred_class = model.predict(X_val_clean.values)
                        
                        score = accuracy_score(y_val, y_pred_class)
                        if score > best_score:
                            best_score = score
                            best_model_name = model_name
                            
                    except Exception as e:
                        print(f"{model_name} 평가 오류: {e}")
                        continue
            else:
                best_score = 0.0
            
            self.results['model_training'] = {
                'models_count': len(trainer.models),
                'best_cv_score': best_score,
                'best_model': best_model_name,
                'target_achieved': best_score >= self.target_accuracy
            }
            
            print(f"최고 성능: {best_score:.4f} ({best_model_name})")
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
            
            if mt.get('best_model'):
                print(f"최고 모델: {mt['best_model']}")
            
            if mt['target_achieved']:
                print("✓ 목표 성능 달성")
            else:
                print("✗ 목표 성능 미달")
                gap = self.target_accuracy - mt['best_cv_score']
                print(f"  성능 격차: {gap:.4f}")
        
        if 'validation' in self.results:
            val = self.results['validation']
            if 'overall_score' in val:
                print(f"검증 점수: {val['overall_score']:.4f}")
        
        if 'prediction' in self.results:
            pred = self.results['prediction']
            print("예측 분포:")
            total_predictions = sum(pred['prediction_counts'].values())
            for cls, count in sorted(pred['prediction_counts'].items()):
                pct = count / total_predictions * 100
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
        
        # 개선 제안
        self.suggest_improvements()
    
    def suggest_improvements(self):
        """개선 제안"""
        print("\n개선 제안:")
        
        suggestions = []
        
        # 성능 기반 제안
        if 'model_training' in self.results:
            mt = self.results['model_training']
            if not mt['target_achieved']:
                gap = self.target_accuracy - mt['best_cv_score']
                if gap > 0.02:
                    suggestions.append("하이퍼파라미터 튜닝 필요")
                    suggestions.append("앙상블 가중치 조정")
                if gap > 0.05:
                    suggestions.append("추가 피처 생성")
                    suggestions.append("다른 모델 아키텍처 시도")
        
        # 검증 기반 제안
        if 'validation' in self.results:
            val = self.results['validation']
            if val.get('overall_score', 0) < 0.5:
                suggestions.append("검증 전략 재검토")
            
            if 'stability' in val and val['stability'].get('stability_score', 0) < 0.95:
                suggestions.append("모델 안정성 개선")
        
        # 데이터 기반 제안
        if 'data_analysis' in self.results:
            da = self.results['data_analysis']
            if 'leakage' in da and da['leakage']:
                suggestions.append("데이터 누수 해결")
            
            if 'target' in da and da['target'].get('imbalance_ratio', 0) > 3:
                suggestions.append("클래스 불균형 해결")
        
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
        else:
            print("  추가 개선사항 없음")
    
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
                print("6단계 실패 - 비상 모드 시도")
                emergency_success = self.emergency_prediction()
                if not emergency_success:
                    print("비상 모드도 실패 - 시스템 종료")
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
            from sklearn.preprocessing import LabelEncoder
            
            # 기본 데이터 로드
            train_df = pd.read_csv('train.csv')
            test_df = pd.read_csv('test.csv')
            
            # 간단한 전처리
            numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
            categorical_cols = ['gender', 'subscription_type']
            
            # 결측치 처리
            for col in numeric_cols:
                if col in train_df.columns:
                    median_val = train_df[col].median()
                    train_df[col].fillna(median_val, inplace=True)
                    if col in test_df.columns:
                        test_df[col].fillna(median_val, inplace=True)
            
            # 범주형 인코딩
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
            
            # 개선된 RandomForest 모델
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced_subsample',
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
            
            # 분포 확인
            pred_dist = submission_df['support_needs'].value_counts().sort_index()
            print("비상 예측 분포:")
            for cls, count in pred_dist.items():
                pct = count / len(submission_df) * 100
                print(f"  클래스 {cls}: {count:,}개 ({pct:.1f}%)")
            
            print("비상 예측 완료: emergency_submission.csv")
            return True
            
        except Exception as e:
            print(f"비상 예측 실패: {e}")
            return False
    
    def optimize_for_target(self):
        """목표 성능 달성을 위한 최적화"""
        print("성능 최적화 시도")
        
        try:
            # 더 공격적인 피처 생성
            train_df = pd.read_csv('train.csv')
            test_df = pd.read_csv('test.csv')
            
            # 추가 피처 생성
            engineer = FeatureEngineer()
            train_processed, test_processed = engineer.create_features(train_df, test_df)
            
            # 더 많은 피처 사용
            preprocessor = DataPreprocessor()
            train_final, test_final = preprocessor.process_data(train_processed, test_processed)
            
            # 피처 선택을 더 관대하게
            feature_cols = [col for col in train_final.columns if col not in ['ID', 'support_needs']]
            
            if len(feature_cols) > 80:
                # 상위 80개 피처 사용
                from sklearn.feature_selection import SelectKBest, mutual_info_classif
                
                X = train_final[feature_cols].fillna(0)
                y = train_final['support_needs']
                
                selector = SelectKBest(score_func=mutual_info_classif, k=80)
                X_selected = selector.fit_transform(X, y)
                
                selected_mask = selector.get_support()
                selected_features = [feature_cols[i] for i, selected in enumerate(selected_mask) if selected]
                
                print(f"최적화: {len(selected_features)}개 피처 사용")
                
                # 최적화된 모델 학습
                X_train_opt = train_final[selected_features].fillna(0)
                X_test_opt = test_final[selected_features].fillna(0)
                y_train_opt = train_final['support_needs']
                
                # LightGBM으로 빠른 최적화
                import lightgbm as lgb
                
                train_data = lgb.Dataset(X_train_opt, label=y_train_opt)
                
                lgb_params = {
                    'objective': 'multiclass',
                    'num_class': 3,
                    'metric': 'multi_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 50,
                    'learning_rate': 0.03,
                    'feature_fraction': 0.95,
                    'bagging_fraction': 0.85,
                    'bagging_freq': 5,
                    'min_child_weight': 3,
                    'reg_alpha': 0.05,
                    'reg_lambda': 0.05,
                    'verbose': -1,
                    'random_state': 42
                }
                
                model_opt = lgb.train(
                    lgb_params,
                    train_data,
                    num_boost_round=1500,
                    callbacks=[lgb.early_stopping(100)]
                )
                
                predictions_opt = model_opt.predict(X_test_opt)
                predictions_opt_class = np.argmax(predictions_opt, axis=1)
                
                # 최적화된 제출 파일
                submission_opt = pd.DataFrame({
                    'ID': test_final['ID'],
                    'support_needs': predictions_opt_class
                })
                
                submission_opt.to_csv('optimized_submission.csv', index=False)
                print("최적화된 제출 파일 생성: optimized_submission.csv")
                
                return True
            
        except Exception as e:
            print(f"최적화 실패: {e}")
            return False
    
    def run_optimized_system(self):
        """최적화된 시스템 실행"""
        success = self.run_system()
        
        # 목표 성능에 미달했다면 최적화 시도
        if success and 'model_training' in self.results:
            mt = self.results['model_training']
            if not mt['target_achieved']:
                print(f"\n목표 성능 미달 ({mt['best_cv_score']:.4f} < {self.target_accuracy})")
                print("추가 최적화 시도")
                
                self.optimize_for_target()
        
        return success

def main():
    """메인 함수"""
    ai_system = AISystem()
    
    # 최적화된 시스템 실행
    success = ai_system.run_optimized_system()
    
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