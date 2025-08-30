# prediction.py

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class PredictionSystem:
    """예측 시스템 클래스"""
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = None
        self.preprocessor = None
        self.calibration_params = {}
        
    def load_models(self):
        """저장된 모델 로드"""
        print("=== 저장된 모델 로드 ===")
        
        try:
            pkl_dir = 'models/pkl'
            json_dir = 'models/json'
            
            self.preprocessor = joblib.load(os.path.join(pkl_dir, 'preprocessor.pkl'))
            self.feature_engineer = joblib.load(os.path.join(pkl_dir, 'feature_engineer.pkl'))
            print("전처리기 및 피처 엔지니어 로드 완료")
            
            model_configs = [
                ('lightgbm', json_dir, 'lightgbm_model.txt', 'lgb'),
                ('xgboost', json_dir, 'xgboost_model.json', 'xgb'),
                ('catboost', pkl_dir, 'catboost_model.pkl', 'pkl'),
                ('random_forest', pkl_dir, 'random_forest_model.pkl', 'pkl'),
                ('extra_trees', pkl_dir, 'extra_trees_model.pkl', 'pkl'),
                ('logistic', pkl_dir, 'logistic_model.pkl', 'pkl'),
                ('stacking', pkl_dir, 'stacking_model.pkl', 'pkl')
            ]
            
            for name, folder, filename, model_type in model_configs:
                model_path = os.path.join(folder, filename)
                
                if os.path.exists(model_path):
                    if model_type == 'lgb':
                        self.models[name] = lgb.Booster(model_file=model_path)
                    elif model_type == 'xgb':
                        model = xgb.Booster()
                        model.load_model(model_path)
                        self.models[name] = model
                    else:
                        self.models[name] = joblib.load(model_path)
                    
                    print(f"{name} 모델 로드 완료: {model_path}")
            
            print(f"총 {len(self.models)}개 모델 로드 완료")
            return True
            
        except Exception as e:
            print(f"모델 로드 중 오류: {e}")
            return False
    
    def load_and_process_test_data(self):
        """테스트 데이터 로드 및 처리"""
        print("=== 테스트 데이터 처리 ===")
        
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        print(f"원본 테스트 데이터: {test_df.shape}")
        
        train_processed, test_processed = self.feature_engineer.process_all_features(train_df, test_df)
        
        train_final, test_final = self.preprocessor.process_complete_pipeline(train_processed, test_processed)
        
        feature_cols = [col for col in test_final.columns if col not in ['ID', 'support_needs']]
        
        X_test = test_final[feature_cols]
        test_ids = test_final['ID']
        
        print(f"처리된 테스트 데이터: {X_test.shape}")
        print(f"피처 수: {len(feature_cols)}")
        
        return X_test, test_ids
    
    def predict_with_models(self, X_test):
        """개별 모델 예측"""
        print("=== 개별 모델 예측 ===")
        
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if name == 'lightgbm':
                    pred_proba = model.predict(X_test)
                    if pred_proba.ndim == 1:
                        pred_proba = pred_proba.reshape(-1, 1)
                    if pred_proba.shape[1] == 1:
                        pred_proba = np.column_stack([
                            1 - pred_proba.flatten(),
                            pred_proba.flatten(),
                            np.zeros(len(pred_proba))
                        ])
                    predictions[name] = pred_proba
                    
                elif name == 'xgboost':
                    xgb_test = xgb.DMatrix(X_test)
                    pred_proba = model.predict(xgb_test)
                    if pred_proba.ndim == 1:
                        pred_proba = pred_proba.reshape(-1, 1)
                    if pred_proba.shape[1] == 1:
                        pred_proba = np.column_stack([
                            1 - pred_proba.flatten(),
                            pred_proba.flatten(),
                            np.zeros(len(pred_proba))
                        ])
                    predictions[name] = pred_proba
                    
                elif name == 'stacking':
                    base_predictions = []
                    for base_name in ['lightgbm', 'xgboost', 'catboost', 'random_forest']:
                        if base_name in predictions:
                            base_predictions.append(predictions[base_name])
                    
                    if base_predictions:
                        stacking_input = np.concatenate(base_predictions, axis=1)
                        pred_proba = model.predict_proba(stacking_input)
                        predictions[name] = pred_proba
                    
                else:
                    pred_proba = model.predict_proba(X_test)
                    predictions[name] = pred_proba
                
                print(f"{name} 예측 완료: {predictions[name].shape}")
                
            except Exception as e:
                print(f"{name} 모델 예측 중 오류: {e}")
        
        return predictions
    
    def calibrate_predictions(self, predictions, method='simple'):
        """예측 확률 보정"""
        print(f"=== 예측 확률 보정 ({method}) ===")
        
        calibrated_predictions = {}
        
        for name, pred_proba in predictions.items():
            try:
                if method == 'simple' and pred_proba.shape[1] == 3:
                    calibrated_proba = np.zeros_like(pred_proba)
                    
                    for class_idx in range(3):
                        proba_class = pred_proba[:, class_idx]
                        
                        proba_class = np.clip(proba_class, 0.001, 0.999)
                        
                        calibrated_proba[:, class_idx] = proba_class
                    
                    row_sums = calibrated_proba.sum(axis=1, keepdims=True)
                    calibrated_proba = calibrated_proba / row_sums
                    
                    calibrated_predictions[name] = calibrated_proba
                else:
                    calibrated_predictions[name] = pred_proba
                    
            except Exception as e:
                print(f"{name} 보정 중 오류: {e}")
                calibrated_predictions[name] = pred_proba
        
        print(f"{len(calibrated_predictions)}개 모델 보정 완료")
        return calibrated_predictions
    
    def create_ensemble_prediction(self, predictions):
        """앙상블 예측 생성"""
        print("=== 앙상블 예측 생성 ===")
        
        weights = {
            'lightgbm': 0.25,
            'xgboost': 0.20,
            'catboost': 0.20,
            'random_forest': 0.15,
            'extra_trees': 0.10,
            'logistic': 0.05,
            'stacking': 0.05
        }
        
        print("모델별 가중치:")
        available_models = []
        total_weight = 0
        
        for name in predictions.keys():
            if name in weights:
                weight = weights[name]
                available_models.append((name, weight))
                total_weight += weight
                print(f"  {name}: {weight}")
        
        if total_weight > 0:
            available_models = [(name, weight/total_weight) for name, weight in available_models]
        
        ensemble_pred = np.zeros((list(predictions.values())[0].shape[0], 3))
        
        for name, weight in available_models:
            if name in predictions:
                pred_proba = predictions[name]
                if pred_proba.shape[1] == 3:
                    ensemble_pred += weight * pred_proba
        
        ensemble_pred_class = np.argmax(ensemble_pred, axis=1)
        
        unique, counts = np.unique(ensemble_pred_class, return_counts=True)
        print("\n앙상블 예측 분포:")
        for cls, count in zip(unique, counts):
            percentage = count / len(ensemble_pred_class) * 100
            print(f"  클래스 {cls}: {count}개 ({percentage:.1f}%)")
        
        return ensemble_pred, ensemble_pred_class
    
    def optimize_threshold(self, predictions, ensemble_pred):
        """임계값 최적화"""
        print("=== 임계값 최적화 ===")
        
        optimized_pred = ensemble_pred.copy()
        
        for class_idx in range(3):
            class_proba = optimized_pred[:, class_idx]
            
            q75 = np.percentile(class_proba, 75)
            q25 = np.percentile(class_proba, 25)
            
            high_confidence = class_proba > q75
            low_confidence = class_proba < q25
            
            optimized_pred[high_confidence, class_idx] *= 1.1
            optimized_pred[low_confidence, class_idx] *= 0.9
        
        row_sums = optimized_pred.sum(axis=1, keepdims=True)
        optimized_pred = optimized_pred / row_sums
        
        optimized_pred_class = np.argmax(optimized_pred, axis=1)
        
        original_dist = np.bincount(np.argmax(ensemble_pred, axis=1), minlength=3)
        optimized_dist = np.bincount(optimized_pred_class, minlength=3)
        
        print("임계값 최적화 결과:")
        for cls in range(3):
            print(f"  클래스 {cls}: {original_dist[cls]} → {optimized_dist[cls]}")
        
        return optimized_pred, optimized_pred_class
    
    def apply_class_balancing(self, pred_proba, pred_class):
        """클래스 균형 조정"""
        print("=== 클래스 균형 조정 ===")
        
        current_dist = np.bincount(pred_class, minlength=3)
        total_samples = len(pred_class)
        
        target_dist = np.array([0.46, 0.27, 0.27])
        target_counts = (target_dist * total_samples).astype(int)
        
        print("분포 조정:")
        for cls in range(3):
            current_pct = current_dist[cls] / total_samples * 100
            target_pct = target_dist[cls] * 100
            print(f"  클래스 {cls}: {current_pct:.1f}% → {target_pct:.1f}%")
        
        adjusted_proba = pred_proba.copy()
        
        for cls in range(3):
            if current_dist[cls] > target_counts[cls]:
                class_mask = pred_class == cls
                adjustment_factor = target_counts[cls] / current_dist[cls]
                adjusted_proba[class_mask, cls] *= adjustment_factor
            elif current_dist[cls] < target_counts[cls]:
                class_mask = pred_class == cls
                adjustment_factor = target_counts[cls] / current_dist[cls]
                adjusted_proba[class_mask, cls] *= min(adjustment_factor, 1.5)
        
        row_sums = adjusted_proba.sum(axis=1, keepdims=True)
        adjusted_proba = adjusted_proba / row_sums
        
        adjusted_pred_class = np.argmax(adjusted_proba, axis=1)
        
        return adjusted_proba, adjusted_pred_class
    
    def create_submission_file(self, test_ids, predictions, filename='submission.csv'):
        """제출 파일 생성"""
        print(f"=== 제출 파일 생성: {filename} ===")
        
        submission_df = pd.DataFrame({
            'ID': test_ids,
            'support_needs': predictions
        })
        
        submission_df.to_csv(filename, index=False)
        
        print(f"제출 파일 저장 완료: {submission_df.shape}")
        
        final_dist = submission_df['support_needs'].value_counts().sort_index()
        print("최종 예측 분포:")
        for cls, count in final_dist.items():
            pct = count / len(submission_df) * 100
            print(f"  클래스 {cls}: {count}개 ({pct:.1f}%)")
        
        print("\n제출 파일 샘플:")
        print(submission_df.head(10))
        
        return submission_df
    
    def predict_test_data(self):
        """전체 예측 파이프라인"""
        print("테스트 데이터 예측 시작")
        print("="*40)
        
        if not self.load_models():
            print("모델 로드 실패")
            return None
        
        X_test, test_ids = self.load_and_process_test_data()
        
        predictions = self.predict_with_models(X_test)
        
        if not predictions:
            print("예측 실패")
            return None
        
        calibrated_predictions = self.calibrate_predictions(predictions)
        
        ensemble_proba, ensemble_pred = self.create_ensemble_prediction(calibrated_predictions)
        
        optimized_proba, optimized_pred = self.optimize_threshold(calibrated_predictions, ensemble_proba)
        
        final_proba, final_pred = self.apply_class_balancing(optimized_proba, optimized_pred)
        
        submission_df = self.create_submission_file(test_ids, final_pred)
        
        print("\n예측 완료!")
        return submission_df

def main():
    """메인 실행 함수"""
    predictor = PredictionSystem()
    submission_df = predictor.predict_test_data()
    
    return predictor, submission_df

if __name__ == "__main__":
    main()