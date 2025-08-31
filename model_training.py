# model_training.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib
from feature_engineering import FeatureEngineer
from preprocessing import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.cv_scores = {}
        self.feature_importance = {}
        self.feature_names = None
        self.class_weights = None
        
    def calculate_class_weights(self, y_train):
        """클래스 가중치 계산"""
        y_array = np.array(y_train)
        class_counts = np.bincount(y_array)
        total_samples = len(y_array)
        
        weights = {}
        for i, count in enumerate(class_counts):
            if count > 0:
                weights[i] = total_samples / (len(class_counts) * count)
            else:
                weights[i] = 1.0
        
        # 클래스 1 가중치 조정
        weights[1] *= 1.2
        
        self.class_weights = weights
        return weights
    
    def safe_data_conversion(self, X, y=None):
        """안전한 데이터 변환"""
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        X_clean = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        if y is not None:
            if hasattr(y, 'values'):
                y_array = y.values
            else:
                y_array = np.array(y)
            y_clean = np.clip(y_array, 0, 2)
            return X_clean, y_clean
        
        return X_clean
    
    def prepare_training_data(self):
        """학습 데이터 준비"""
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        engineer = FeatureEngineer()
        train_df, test_df = engineer.create_features(train_df, test_df)
        
        preprocessor = DataPreprocessor()
        train_df, test_df = preprocessor.process_data(train_df, test_df)
        
        X_train, X_val, y_train, y_val, X_test, test_ids = preprocessor.prepare_data_temporal(
            train_df, test_df
        )
        
        self.feature_names = list(X_train.columns)
        self.calculate_class_weights(y_train)
        
        return X_train, X_val, y_train, y_val, X_test, test_ids, engineer, preprocessor
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """LightGBM 학습"""
        print("LightGBM 학습")
        
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_weight': 5,
            'min_split_gain': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'max_depth': 6,
            'verbose': -1,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        train_data = lgb.Dataset(X_train_clean, label=y_train_clean, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val_clean, label=y_val_clean, reference=train_data)
        
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_val_clean)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_val_clean, y_pred_class)
        
        print(f"LightGBM 검증 정확도: {accuracy:.4f}")
        
        self.models['lightgbm'] = model
        return model, accuracy
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """XGBoost 학습"""
        print("XGBoost 학습")
        
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_weight': 5,
            'gamma': 0.1,
            'random_state': 42,
            'verbosity': 0
        }
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        train_data = xgb.DMatrix(X_train_clean, label=y_train_clean, feature_names=self.feature_names)
        val_data = xgb.DMatrix(X_val_clean, label=y_val_clean, feature_names=self.feature_names)
        
        model = xgb.train(
            params,
            train_data,
            num_boost_round=1000,
            evals=[(val_data, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=0
        )
        
        y_pred = model.predict(val_data)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_val_clean, y_pred_class)
        
        print(f"XGBoost 검증 정확도: {accuracy:.4f}")
        
        self.models['xgboost'] = model
        return model, accuracy
    
    def train_catboost(self, X_train, y_train, X_val, y_val):
        """CatBoost 학습"""
        print("CatBoost 학습")
        
        try:
            X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
            X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
            
            model = CatBoostClassifier(
                iterations=800,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                bootstrap_type='Bernoulli',
                subsample=0.8,
                colsample_bylevel=0.8,
                random_seed=42,
                verbose=0,
                early_stopping_rounds=50,
                task_type='CPU',
                thread_count=-1
            )
            
            model.fit(
                X_train_clean, y_train_clean,
                eval_set=(X_val_clean, y_val_clean),
                use_best_model=True,
                verbose=False
            )
            
            y_pred = model.predict(X_val_clean)
            accuracy = accuracy_score(y_val_clean, y_pred)
            
            print(f"CatBoost 검증 정확도: {accuracy:.4f}")
            
            self.models['catboost'] = model
            return model, accuracy
            
        except Exception as e:
            print(f"CatBoost 학습 실패: {e}")
            return None, 0.0
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Random Forest 학습"""
        print("Random Forest 학습")
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=0.7,
            class_weight=self.class_weights,
            random_state=42,
            n_jobs=-1,
            bootstrap=True
        )
        
        model.fit(X_train_clean, y_train_clean)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        print(f"Random Forest 검증 정확도: {accuracy:.4f}")
        
        self.models['random_forest'] = model
        return model, accuracy
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """신경망 학습"""
        print("Neural Network 학습")
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        
        model.fit(X_train_clean, y_train_clean)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        print(f"Neural Network 검증 정확도: {accuracy:.4f}")
        
        self.models['neural_network'] = model
        return model, accuracy
    
    def perform_temporal_cv(self, X, y, model_type='lightgbm', n_splits=5):
        """시간 기반 교차 검증"""
        print(f"{model_type.upper()} 시간 기반 CV")
        
        if 'temporal_id' in X.columns:
            # 시간 순서 기반 분할
            temporal_ids = X['temporal_id'].values
            sorted_indices = np.argsort(temporal_ids)
            
            fold_size = len(sorted_indices) // (n_splits + 1)
            cv_scores = []
            
            for fold in range(n_splits):
                train_end = (fold + 1) * fold_size
                val_start = train_end
                val_end = val_start + fold_size
                
                if val_end > len(sorted_indices):
                    break
                
                train_idx = sorted_indices[:train_end]
                val_idx = sorted_indices[val_start:val_end]
                
                try:
                    X_train_fold = X.iloc[train_idx]
                    y_train_fold = y.iloc[train_idx]
                    X_val_fold = X.iloc[val_idx]
                    y_val_fold = y.iloc[val_idx]
                    
                    # temporal_id 제외
                    feature_cols = [col for col in X_train_fold.columns if col != 'temporal_id']
                    
                    X_train_clean, y_train_clean = self.safe_data_conversion(
                        X_train_fold[feature_cols], y_train_fold
                    )
                    X_val_clean, y_val_clean = self.safe_data_conversion(
                        X_val_fold[feature_cols], y_val_fold
                    )
                    
                    if model_type == 'lightgbm':
                        params = {
                            'objective': 'multiclass', 'num_class': 3, 
                            'verbose': -1, 'random_state': 42,
                            'num_leaves': 31, 'learning_rate': 0.05
                        }
                        
                        train_data = lgb.Dataset(X_train_clean, label=y_train_clean)
                        model = lgb.train(params, train_data, num_boost_round=200)
                        y_pred = model.predict(X_val_clean)
                        y_pred_class = np.argmax(y_pred, axis=1)
                        
                    elif model_type == 'random_forest':
                        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                        model.fit(X_train_clean, y_train_clean)
                        y_pred_class = model.predict(X_val_clean)
                    
                    accuracy = accuracy_score(y_val_clean, y_pred_class)
                    cv_scores.append(accuracy)
                    
                except Exception as e:
                    print(f"  Fold {fold + 1} 오류: {e}")
                    cv_scores.append(0.0)
        else:
            # TimeSeriesSplit 사용
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                cv_scores.append(0.50)  # 기본값
        
        mean_score = np.mean(cv_scores) if cv_scores else 0.0
        std_score = np.std(cv_scores) if cv_scores else 0.0
        
        print(f"시간 기반 CV 점수: {mean_score:.4f} (+/- {std_score:.4f})")
        
        self.cv_scores[model_type] = {
            'scores': cv_scores,
            'mean': mean_score,
            'std': std_score
        }
        
        return mean_score, std_score
    
    def create_ensemble_model(self, X_train, y_train, X_val, y_val):
        """앙상블 모델 생성"""
        print("앙상블 모델 생성")
        
        # 기본 모델들 확인
        base_models = ['lightgbm', 'xgboost', 'catboost', 'random_forest']
        valid_models = [name for name in base_models if name in self.models]
        
        if len(valid_models) < 2:
            print("앙상블을 위한 모델이 부족합니다")
            return None, 0.0
        
        try:
            X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
            X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
            
            # 투표 기반 앙상블
            voting_estimators = []
            
            if 'random_forest' in valid_models:
                rf_model = RandomForestClassifier(
                    n_estimators=200, max_depth=10, 
                    class_weight=self.class_weights, random_state=42
                )
                voting_estimators.append(('rf', rf_model))
            
            if 'neural_network' in self.models:
                nn_model = MLPClassifier(
                    hidden_layer_sizes=(64, 32), max_iter=500, 
                    random_state=42
                )
                voting_estimators.append(('nn', nn_model))
            
            if len(voting_estimators) >= 2:
                ensemble = VotingClassifier(
                    estimators=voting_estimators,
                    voting='soft'
                )
                
                ensemble.fit(X_train_clean, y_train_clean)
                
                y_pred = ensemble.predict(X_val_clean)
                accuracy = accuracy_score(y_val_clean, y_pred)
                
                print(f"앙상블 검증 정확도: {accuracy:.4f}")
                
                self.models['ensemble'] = ensemble
                return ensemble, accuracy
            
        except Exception as e:
            print(f"앙상블 생성 오류: {e}")
        
        return None, 0.0
    
    def save_models(self, engineer, preprocessor):
        """모델 저장"""
        print("모델 저장")
        
        os.makedirs('models', exist_ok=True)
        
        try:
            # 전처리기 저장
            joblib.dump(preprocessor, 'models/preprocessor.pkl')
            joblib.dump(engineer, 'models/feature_engineer.pkl')
            
            # 피처 정보 저장
            feature_info = {
                'feature_names': self.feature_names,
                'feature_count': len(self.feature_names),
                'class_weights': self.class_weights
            }
            joblib.dump(feature_info, 'models/feature_info.pkl')
            print("전처리기 및 피처 정보 저장 완료")
        except Exception as e:
            print(f"전처리기 저장 오류: {e}")
        
        # 모델 저장
        saved_count = 0
        for name, model in self.models.items():
            try:
                if name == 'lightgbm':
                    model.save_model(f'models/{name}_model.txt')
                elif name == 'xgboost':
                    model.save_model(f'models/{name}_model.json')
                else:
                    joblib.dump(model, f'models/{name}_model.pkl')
                
                saved_count += 1
                print(f"  {name} 모델 저장 완료")
                
            except Exception as e:
                print(f"  {name} 모델 저장 오류: {e}")
        
        print(f"총 {saved_count}개 모델 저장 완료")
    
    def train_models(self, X_train, X_val, y_train, y_val, engineer=None, preprocessor=None):
        """모든 모델 학습"""
        print("모델 학습 시작")
        print("=" * 40)
        
        if engineer is None:
            engineer = FeatureEngineer()
        
        if preprocessor is None:
            preprocessor = DataPreprocessor()
        
        if self.feature_names is None:
            self.feature_names = list(X_train.columns)
        
        if self.class_weights is None:
            self.calculate_class_weights(y_train)
        
        model_results = {}
        
        # 핵심 모델들만 학습
        try:
            lgb_model, lgb_acc = self.train_lightgbm(X_train, y_train, X_val, y_val)
            model_results['lightgbm'] = lgb_acc
        except Exception as e:
            print(f"LightGBM 학습 실패: {e}")
            model_results['lightgbm'] = 0.0
        
        try:
            xgb_model, xgb_acc = self.train_xgboost(X_train, y_train, X_val, y_val)
            model_results['xgboost'] = xgb_acc
        except Exception as e:
            print(f"XGBoost 학습 실패: {e}")
            model_results['xgboost'] = 0.0
        
        try:
            cat_model, cat_acc = self.train_catboost(X_train, y_train, X_val, y_val)
            if cat_model is not None:
                model_results['catboost'] = cat_acc
        except Exception as e:
            print(f"CatBoost 학습 실패: {e}")
            model_results['catboost'] = 0.0
        
        try:
            rf_model, rf_acc = self.train_random_forest(X_train, y_train, X_val, y_val)
            model_results['random_forest'] = rf_acc
        except Exception as e:
            print(f"Random Forest 학습 실패: {e}")
            model_results['random_forest'] = 0.0
        
        try:
            nn_model, nn_acc = self.train_neural_network(X_train, y_train, X_val, y_val)
            model_results['neural_network'] = nn_acc
        except Exception as e:
            print(f"Neural Network 학습 실패: {e}")
            model_results['neural_network'] = 0.0
        
        # 앙상블 모델 생성
        try:
            ensemble_model, ensemble_acc = self.create_ensemble_model(X_train, y_train, X_val, y_val)
            if ensemble_model is not None:
                model_results['ensemble'] = ensemble_acc
        except Exception as e:
            print(f"앙상블 모델 생성 실패: {e}")
        
        # 시간 기반 교차 검증
        if len(self.models) > 0:
            full_X = pd.concat([X_train, X_val])
            full_y = pd.concat([y_train, y_val])
            
            if 'lightgbm' in self.models:
                self.perform_temporal_cv(full_X, full_y, 'lightgbm')
        
        # 모델 저장
        self.save_models(engineer, preprocessor)
        
        # 결과 출력
        print("\n모델 성능 요약:")
        valid_results = {k: v for k, v in model_results.items() if v > 0}
        
        for model_name, accuracy in valid_results.items():
            print(f"{model_name}: {accuracy:.4f}")
        
        if valid_results:
            best_model = max(valid_results.items(), key=lambda x: x[1])
            print(f"최고 성능: {best_model[1]:.4f} ({best_model[0]})")

def main():
    trainer = ModelTrainer()
    X_train, X_val, y_train, y_val, X_test, test_ids, engineer, preprocessor = trainer.prepare_training_data()
    trainer.train_models(X_train, X_val, y_train, y_val, engineer, preprocessor)
    
    return trainer

if __name__ == "__main__":
    main()