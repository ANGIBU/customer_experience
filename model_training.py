# model_training.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, f1_score
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
        self.feature_names = None
        self.class_weights = None
        self.best_threshold = 0.5
        self.ensemble_weights = {}
        
    def calculate_class_weights(self, y_train):
        """클래스 가중치 계산"""
        y_array = np.array(y_train)
        class_counts = np.bincount(y_array)
        total_samples = len(y_array)
        
        # 균형 잡힌 가중치 계산
        weights = {}
        for i, count in enumerate(class_counts):
            if count > 0:
                weights[i] = total_samples / (len(class_counts) * count)
            else:
                weights[i] = 1.0
        
        # 원본 클래스 보정으로 복원
        weights[1] *= 1.32
        weights[2] *= 1.08
        
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
        """훈련 데이터 준비"""
        try:
            train_df = pd.read_csv('train.csv')
            test_df = pd.read_csv('test.csv')
            
            # 데이터 분석으로부터 temporal_threshold 가져오기
            try:
                from data_analysis import DataAnalyzer
                analyzer = DataAnalyzer()
                analysis_results = analyzer.run_analysis()
                temporal_threshold = analysis_results.get('temporal', {}).get('temporal_threshold')
                temporal_info = analysis_results.get('temporal')
            except Exception:
                temporal_threshold = None
                temporal_info = None
            
            engineer = FeatureEngineer()
            train_df, test_df = engineer.create_features(train_df, test_df, temporal_threshold)
            
            if train_df is None or test_df is None:
                raise ValueError("피처 생성 실패")
            
            preprocessor = DataPreprocessor()
            train_df, test_df = preprocessor.process_data(train_df, test_df, temporal_info)
            
            if train_df is None or test_df is None:
                raise ValueError("전처리 실패")
            
            X_train, X_val, y_train, y_val, X_test, test_ids = preprocessor.prepare_data_temporal_optimized(
                train_df, test_df, val_size=0.18, gap_size=0.005
            )
            
            if X_train is None or X_val is None:
                raise ValueError("데이터 분할 실패")
            
            self.feature_names = list(X_train.columns)
            self.calculate_class_weights(y_train)
            
            return X_train, X_val, y_train, y_val, X_test, test_ids, engineer, preprocessor
            
        except Exception as e:
            print(f"훈련 데이터 준비 오류: {e}")
            return None, None, None, None, None, None, None, None
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """LightGBM 모델 학습"""
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 45,
            'learning_rate': 0.028,
            'feature_fraction': 0.82,
            'bagging_fraction': 0.88,
            'bagging_freq': 5,
            'min_child_weight': 10,
            'min_split_gain': 0.12,
            'reg_alpha': 0.12,
            'reg_lambda': 0.12,
            'max_depth': 8,
            'verbose': -1,
            'random_state': 42,
            'force_col_wise': True
        }
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        # 클래스 가중치 적용
        sample_weight = np.ones(len(y_train_clean))
        for i, weight in self.class_weights.items():
            mask = y_train_clean == i
            sample_weight[mask] = weight
        
        train_data = lgb.Dataset(X_train_clean, label=y_train_clean, weight=sample_weight, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val_clean, label=y_val_clean, reference=train_data)
        
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_val_clean)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_val_clean, y_pred_class)
        
        self.models['lightgbm'] = model
        return model, accuracy
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """XGBoost 모델 학습"""
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'max_depth': 7,
            'learning_rate': 0.028,
            'subsample': 0.88,
            'colsample_bytree': 0.82,
            'reg_alpha': 0.12,
            'reg_lambda': 0.12,
            'min_child_weight': 10,
            'gamma': 0.12,
            'random_state': 42,
            'verbosity': 0,
            'tree_method': 'hist'
        }
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        # 클래스 가중치 적용
        sample_weight = np.ones(len(y_train_clean))
        for i, weight in self.class_weights.items():
            mask = y_train_clean == i
            sample_weight[mask] = weight
        
        train_data = xgb.DMatrix(X_train_clean, label=y_train_clean, weight=sample_weight, feature_names=self.feature_names)
        val_data = xgb.DMatrix(X_val_clean, label=y_val_clean, feature_names=self.feature_names)
        
        model = xgb.train(
            params,
            train_data,
            num_boost_round=2000,
            evals=[(val_data, 'eval')],
            early_stopping_rounds=120,
            verbose_eval=0
        )
        
        y_pred = model.predict(val_data)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_val_clean, y_pred_class)
        
        self.models['xgboost'] = model
        return model, accuracy
    
    def train_catboost(self, X_train, y_train, X_val, y_val):
        """CatBoost 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        # 클래스 가중치 적용
        sample_weight = np.ones(len(y_train_clean))
        for i, weight in self.class_weights.items():
            mask = y_train_clean == i
            sample_weight[mask] = weight
        
        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.028,
            depth=7,
            l2_leaf_reg=4,
            bootstrap_type='Bernoulli',
            subsample=0.82,
            colsample_bylevel=0.82,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=120,
            task_type='CPU',
            thread_count=-1
        )
        
        model.fit(
            X_train_clean, y_train_clean,
            sample_weight=sample_weight,
            eval_set=(X_val_clean, y_val_clean),
            use_best_model=True,
            verbose=False
        )
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['catboost'] = model
        return model, accuracy
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Random Forest 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = RandomForestClassifier(
            n_estimators=450,
            max_depth=13,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features=0.82,
            bootstrap=True,
            class_weight=self.class_weights,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_clean, y_train_clean)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['random_forest'] = model
        return model, accuracy
    
    def train_gradient_boosting(self, X_train, y_train, X_val, y_val):
        """Gradient Boosting 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = GradientBoostingClassifier(
            n_estimators=280,
            learning_rate=0.05,
            max_depth=7,
            min_samples_split=12,
            min_samples_leaf=6,
            subsample=0.88,
            random_state=42
        )
        
        # 클래스 가중치를 sample_weight으로 변환
        sample_weight = np.ones(len(y_train_clean))
        for i, weight in self.class_weights.items():
            mask = y_train_clean == i
            sample_weight[mask] = weight
        
        model.fit(X_train_clean, y_train_clean, sample_weight=sample_weight)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['gradient_boosting'] = model
        return model, accuracy
    
    def train_extra_trees(self, X_train, y_train, X_val, y_val):
        """Extra Trees 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = ExtraTreesClassifier(
            n_estimators=380,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=0.85,
            bootstrap=True,
            class_weight=self.class_weights,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_clean, y_train_clean)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['extra_trees'] = model
        return model, accuracy
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """신경망 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = MLPClassifier(
            hidden_layer_sizes=(180, 90, 45),
            activation='relu',
            solver='adam',
            alpha=0.0008,
            learning_rate='adaptive',
            learning_rate_init=0.0008,
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=40
        )
        
        model.fit(X_train_clean, y_train_clean)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['neural_network'] = model
        return model, accuracy
    
    def create_stacking_ensemble(self, X_train, y_train, X_val, y_val):
        """스태킹 앙상블"""
        base_models = ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'gradient_boosting']
        available_models = [name for name in base_models if name in self.models]
        
        if len(available_models) < 3:
            return None, 0.0
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        # 메타 피처 생성 (3-fold CV)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        meta_features_train = np.zeros((len(X_train_clean), len(available_models) * 3))
        meta_features_val = np.zeros((len(X_val_clean), len(available_models) * 3))
        
        for model_idx, model_name in enumerate(available_models):
            model = self.models[model_name]
            
            # 교차 검증으로 메타 피처 생성
            for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(skf.split(X_train_clean, y_train_clean)):
                X_fold_train = X_train_clean[fold_train_idx]
                y_fold_train = y_train_clean[fold_train_idx]
                X_fold_val = X_train_clean[fold_val_idx]
                
                # 임시 모델 학습
                if model_name == 'lightgbm':
                    temp_model = lgb.LGBMClassifier(random_state=42, verbose=-1, n_estimators=180)
                elif model_name == 'xgboost':
                    temp_model = xgb.XGBClassifier(random_state=42, verbosity=0, n_estimators=180)
                elif model_name == 'catboost':
                    temp_model = CatBoostClassifier(random_seed=42, verbose=0, iterations=180)
                elif model_name == 'random_forest':
                    temp_model = RandomForestClassifier(random_state=42, n_estimators=180)
                else:
                    temp_model = GradientBoostingClassifier(random_state=42, n_estimators=180)
                
                temp_model.fit(X_fold_train, y_fold_train)
                
                if hasattr(temp_model, 'predict_proba'):
                    fold_pred = temp_model.predict_proba(X_fold_val)
                    meta_features_train[fold_val_idx, model_idx*3:(model_idx+1)*3] = fold_pred
                
            # 검증 데이터에 대한 예측
            if model_name == 'lightgbm':
                val_pred = model.predict(X_val_clean)
                if val_pred.ndim == 2 and val_pred.shape[1] == 3:
                    meta_features_val[:, model_idx*3:(model_idx+1)*3] = val_pred
                    
            elif model_name == 'xgboost':
                val_data = xgb.DMatrix(X_val_clean, feature_names=self.feature_names)
                val_pred = model.predict(val_data)
                if val_pred.ndim == 2 and val_pred.shape[1] == 3:
                    meta_features_val[:, model_idx*3:(model_idx+1)*3] = val_pred
                    
            else:
                if hasattr(model, 'predict_proba'):
                    val_pred = model.predict_proba(X_val_clean)
                    if val_pred.shape[1] == 3:
                        meta_features_val[:, model_idx*3:(model_idx+1)*3] = val_pred
        
        # 메타 모델 학습
        meta_model = LogisticRegression(
            class_weight=self.class_weights,
            random_state=42,
            max_iter=2000,
            solver='liblinear'
        )
        
        meta_model.fit(meta_features_train, y_train_clean)
        
        y_pred = meta_model.predict(meta_features_val)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['stacking'] = {
            'base_models': available_models,
            'meta_model': meta_model,
            'base_model_objects': {name: self.models[name] for name in available_models}
        }
        
        return meta_model, accuracy
    
    def optimize_ensemble_weights(self, X_val, y_val):
        """앙상블 가중치 최적화"""
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        # 각 모델의 성능 평가
        model_scores = {}
        
        for name, model in self.models.items():
            if name == 'stacking':
                continue
                
            try:
                if name == 'lightgbm':
                    y_pred_proba = model.predict(X_val_clean)
                    if y_pred_proba.ndim == 2:
                        y_pred = np.argmax(y_pred_proba, axis=1)
                        
                elif name == 'xgboost':
                    val_data = xgb.DMatrix(X_val_clean, feature_names=self.feature_names)
                    y_pred_proba = model.predict(val_data)
                    if y_pred_proba.ndim == 2:
                        y_pred = np.argmax(y_pred_proba, axis=1)
                        
                else:
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_val_clean)
                        y_pred = np.argmax(y_pred_proba, axis=1)
                    else:
                        y_pred = model.predict(X_val_clean)
                
                accuracy = accuracy_score(y_val_clean, y_pred)
                f1 = f1_score(y_val_clean, y_pred, average='macro')
                
                # 성능 점수 (accuracy와 f1의 가중 평균)
                combined_score = 0.75 * accuracy + 0.25 * f1
                model_scores[name] = combined_score
                
            except Exception as e:
                model_scores[name] = 0.0
        
        # 원본 가중치 정규화
        total_score = sum(model_scores.values())
        if total_score > 0:
            for name in model_scores:
                self.ensemble_weights[name] = model_scores[name] / total_score
        else:
            # 균등 가중치
            num_models = len(model_scores)
            for name in model_scores:
                self.ensemble_weights[name] = 1.0 / num_models
        
        return self.ensemble_weights
    
    def save_models(self, engineer, preprocessor):
        """모델 저장"""
        os.makedirs('models', exist_ok=True)
        
        # 전처리기 저장
        joblib.dump(preprocessor, 'models/preprocessor.pkl')
        joblib.dump(engineer, 'models/feature_engineer.pkl')
        
        # 피처 정보 저장
        feature_info = {
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names),
            'class_weights': self.class_weights,
            'ensemble_weights': self.ensemble_weights
        }
        joblib.dump(feature_info, 'models/feature_info.pkl')
        
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
                
            except Exception as e:
                continue
        
        return saved_count
    
    def train_models(self, X_train, X_val, y_train, y_val, engineer=None, preprocessor=None):
        """모델 학습 파이프라인"""
        if engineer is None:
            engineer = FeatureEngineer()
        if preprocessor is None:
            preprocessor = DataPreprocessor()
        if self.feature_names is None:
            self.feature_names = list(X_train.columns)
        if self.class_weights is None:
            self.calculate_class_weights(y_train)
        
        model_results = {}
        
        # LightGBM
        try:
            lgb_model, lgb_acc = self.train_lightgbm(X_train, y_train, X_val, y_val)
            model_results['lightgbm'] = lgb_acc
        except Exception:
            model_results['lightgbm'] = 0.0
        
        # XGBoost
        try:
            xgb_model, xgb_acc = self.train_xgboost(X_train, y_train, X_val, y_val)
            model_results['xgboost'] = xgb_acc
        except Exception:
            model_results['xgboost'] = 0.0
        
        # CatBoost
        try:
            cat_model, cat_acc = self.train_catboost(X_train, y_train, X_val, y_val)
            model_results['catboost'] = cat_acc
        except Exception:
            model_results['catboost'] = 0.0
        
        # Random Forest
        try:
            rf_model, rf_acc = self.train_random_forest(X_train, y_train, X_val, y_val)
            model_results['random_forest'] = rf_acc
        except Exception:
            model_results['random_forest'] = 0.0
        
        # Gradient Boosting
        try:
            gb_model, gb_acc = self.train_gradient_boosting(X_train, y_train, X_val, y_val)
            model_results['gradient_boosting'] = gb_acc
        except Exception:
            model_results['gradient_boosting'] = 0.0
        
        # Extra Trees
        try:
            et_model, et_acc = self.train_extra_trees(X_train, y_train, X_val, y_val)
            model_results['extra_trees'] = et_acc
        except Exception:
            model_results['extra_trees'] = 0.0
        
        # Neural Network
        try:
            nn_model, nn_acc = self.train_neural_network(X_train, y_train, X_val, y_val)
            model_results['neural_network'] = nn_acc
        except Exception:
            model_results['neural_network'] = 0.0
        
        # 앙상블 가중치 최적화
        self.optimize_ensemble_weights(X_val, y_val)
        
        # 스태킹 앙상블
        try:
            stacking_model, stacking_acc = self.create_stacking_ensemble(X_train, y_train, X_val, y_val)
            if stacking_model is not None:
                model_results['stacking'] = stacking_acc
        except Exception:
            pass
        
        # 모델 저장
        self.save_models(engineer, preprocessor)
        
        # 결과 출력
        valid_results = {k: v for k, v in model_results.items() if v > 0}
        
        if valid_results:
            best_model = max(valid_results.items(), key=lambda x: x[1])

def main():
    try:
        trainer = ModelTrainer()
        result = trainer.prepare_training_data()
        
        if result[0] is not None:  # X_train이 None이 아니면
            X_train, X_val, y_train, y_val, X_test, test_ids, engineer, preprocessor = result
            trainer.train_models(X_train, X_val, y_train, y_val, engineer, preprocessor)
            return trainer
        else:
            print("훈련 데이터 준비 실패")
            return None
            
    except Exception as e:
        print(f"모델 훈련 오류: {e}")
        return None

if __name__ == "__main__":
    main()