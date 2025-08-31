# model_training.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import ADASYN
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
        
        weights[1] *= 1.4
        
        self.class_weights = weights
        return weights
    
    def safe_data_conversion(self, X, y=None):
        """데이터 변환"""
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
    
    def apply_resampling(self, X_train, y_train):
        """리샘플링 적용"""
        print("리샘플링")
        
        try:
            X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
            
            # ADASYN 적용
            adasyn = ADASYN(random_state=42, n_neighbors=5)
            X_resampled, y_resampled = adasyn.fit_resample(X_train_clean, y_train_clean)
            
            print(f"리샘플링: {len(X_train_clean)} → {len(X_resampled)}")
            
            # 클래스 분포 확인
            unique, counts = np.unique(y_resampled, return_counts=True)
            for cls, count in zip(unique, counts):
                print(f"클래스 {cls}: {count}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"리샘플링 오류: {e}")
            return self.safe_data_conversion(X_train, y_train)
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """LightGBM 학습"""
        print("LightGBM 학습")
        
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 50,
            'learning_rate': 0.03,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'min_child_weight': 8,
            'min_split_gain': 0.2,
            'reg_alpha': 0.2,
            'reg_lambda': 0.2,
            'max_depth': 8,
            'verbose': -1,
            'random_state': 42
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
            num_boost_round=1500,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_val_clean)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_val_clean, y_pred_class)
        
        print(f"LightGBM 정확도: {accuracy:.4f}")
        
        self.models['lightgbm'] = model
        return model, accuracy
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """XGBoost 학습"""
        print("XGBoost 학습")
        
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.2,
            'reg_lambda': 0.2,
            'min_child_weight': 8,
            'gamma': 0.2,
            'random_state': 42,
            'verbosity': 0
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
            num_boost_round=1500,
            evals=[(val_data, 'eval')],
            early_stopping_rounds=100,
            verbose_eval=0
        )
        
        y_pred = model.predict(val_data)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_val_clean, y_pred_class)
        
        print(f"XGBoost 정확도: {accuracy:.4f}")
        
        self.models['xgboost'] = model
        return model, accuracy
    
    def train_catboost(self, X_train, y_train, X_val, y_val):
        """CatBoost 학습"""
        print("CatBoost 학습")
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        # 클래스 가중치 적용
        sample_weight = np.ones(len(y_train_clean))
        for i, weight in self.class_weights.items():
            mask = y_train_clean == i
            sample_weight[mask] = weight
        
        model = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=5,
            bootstrap_type='Bernoulli',
            subsample=0.85,
            colsample_bylevel=0.85,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=100,
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
        
        print(f"CatBoost 정확도: {accuracy:.4f}")
        
        self.models['catboost'] = model
        return model, accuracy
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Random Forest 학습"""
        print("Random Forest 학습")
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=15,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features=0.75,
            class_weight=self.class_weights,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_clean, y_train_clean)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        print(f"Random Forest 정확도: {accuracy:.4f}")
        
        self.models['random_forest'] = model
        return model, accuracy
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """신경망 학습"""
        print("Neural Network 학습")
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30
        )
        
        model.fit(X_train_clean, y_train_clean)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        print(f"Neural Network 정확도: {accuracy:.4f}")
        
        self.models['neural_network'] = model
        return model, accuracy
    
    def create_stacking_ensemble(self, X_train, y_train, X_val, y_val):
        """스태킹 앙상블"""
        print("스태킹 앙상블")
        
        base_models = ['lightgbm', 'xgboost', 'catboost', 'random_forest']
        available_models = [name for name in base_models if name in self.models]
        
        if len(available_models) < 3:
            print("스태킹을 위한 모델 부족")
            return None, 0.0
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        # 메타 피처 생성
        meta_features_train = []
        meta_features_val = []
        
        for model_name in available_models:
            model = self.models[model_name]
            
            if model_name == 'lightgbm':
                train_pred = model.predict(X_train_clean)
                val_pred = model.predict(X_val_clean)
                
                if train_pred.ndim == 2:
                    meta_features_train.append(train_pred)
                    meta_features_val.append(val_pred)
                    
            elif model_name == 'xgboost':
                train_data = xgb.DMatrix(X_train_clean, feature_names=self.feature_names)
                val_data = xgb.DMatrix(X_val_clean, feature_names=self.feature_names)
                
                train_pred = model.predict(train_data)
                val_pred = model.predict(val_data)
                
                if train_pred.ndim == 2:
                    meta_features_train.append(train_pred)
                    meta_features_val.append(val_pred)
                    
            else:
                if hasattr(model, 'predict_proba'):
                    train_pred = model.predict_proba(X_train_clean)
                    val_pred = model.predict_proba(X_val_clean)
                    
                    if train_pred.shape[1] == 3:
                        meta_features_train.append(train_pred)
                        meta_features_val.append(val_pred)
        
        if len(meta_features_train) < 2:
            print("메타 피처 생성 실패")
            return None, 0.0
        
        # 메타 모델 학습
        meta_X_train = np.hstack(meta_features_train)
        meta_X_val = np.hstack(meta_features_val)
        
        meta_model = LogisticRegression(
            class_weight=self.class_weights,
            random_state=42,
            max_iter=1000
        )
        
        meta_model.fit(meta_X_train, y_train_clean)
        
        y_pred = meta_model.predict(meta_X_val)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        print(f"스태킹 정확도: {accuracy:.4f}")
        
        self.models['stacking'] = {
            'base_models': available_models,
            'meta_model': meta_model
        }
        
        return meta_model, accuracy
    
    def perform_temporal_cv(self, X, y, model_type='lightgbm', n_splits=5):
        """시간 기반 교차 검증"""
        print(f"{model_type.upper()} 시간 CV")
        
        if 'temporal_id' in X.columns:
            temporal_ids = X['temporal_id'].values
            sorted_indices = np.argsort(temporal_ids)
            
            fold_size = len(sorted_indices) // (n_splits + 1)
            cv_scores = []
            
            temporal_col_idx = list(X.columns).index('temporal_id')
            
            for fold in range(n_splits):
                train_end = (fold + 1) * fold_size
                val_start = train_end
                val_end = val_start + fold_size
                
                if val_end > len(sorted_indices):
                    break
                
                train_idx = sorted_indices[:train_end]
                val_idx = sorted_indices[val_start:val_end]
                
                X_train_fold = np.delete(X.iloc[train_idx].values, temporal_col_idx, axis=1)
                y_train_fold = y.iloc[train_idx].values
                X_val_fold = np.delete(X.iloc[val_idx].values, temporal_col_idx, axis=1)
                y_val_fold = y.iloc[val_idx].values
                
                X_train_clean, y_train_clean = self.safe_data_conversion(X_train_fold, y_train_fold)
                X_val_clean, y_val_clean = self.safe_data_conversion(X_val_fold, y_val_fold)
                
                if model_type == 'lightgbm':
                    params = {
                        'objective': 'multiclass', 'num_class': 3,
                        'verbose': -1, 'random_state': 42,
                        'num_leaves': 31, 'learning_rate': 0.05
                    }
                    
                    train_data = lgb.Dataset(X_train_clean, label=y_train_clean)
                    model = lgb.train(params, train_data, num_boost_round=300)
                    y_pred = model.predict(X_val_clean)
                    y_pred_class = np.argmax(y_pred, axis=1)
                    
                elif model_type == 'random_forest':
                    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                    model.fit(X_train_clean, y_train_clean)
                    y_pred_class = model.predict(X_val_clean)
                
                accuracy = accuracy_score(y_val_clean, y_pred_class)
                cv_scores.append(accuracy)
        else:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = []
            
            X_clean, y_clean = self.safe_data_conversion(X, y)
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
                model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                model.fit(X_clean[train_idx], y_clean[train_idx])
                y_pred = model.predict(X_clean[val_idx])
                accuracy = accuracy_score(y_clean[val_idx], y_pred)
                cv_scores.append(accuracy)
        
        mean_score = np.mean(cv_scores) if cv_scores else 0.0
        std_score = np.std(cv_scores) if cv_scores else 0.0
        
        print(f"시간 CV 점수: {mean_score:.4f} (+/- {std_score:.4f})")
        
        self.cv_scores[model_type] = {
            'scores': cv_scores,
            'mean': mean_score,
            'std': std_score
        }
        
        return mean_score, std_score
    
    def save_models(self, engineer, preprocessor):
        """모델 저장"""
        print("모델 저장")
        
        os.makedirs('models', exist_ok=True)
        
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
        
        # 모델 저장
        saved_count = 0
        for name, model in self.models.items():
            try:
                if name == 'lightgbm':
                    model.save_model(f'models/{name}_model.txt')
                elif name == 'xgboost':
                    model.save_model(f'models/{name}_model.json')
                elif name == 'stacking':
                    joblib.dump(model, f'models/{name}_model.pkl')
                else:
                    joblib.dump(model, f'models/{name}_model.pkl')
                
                saved_count += 1
                print(f"{name} 저장 완료")
                
            except Exception as e:
                print(f"{name} 저장 오류: {e}")
        
        print(f"총 {saved_count}개 모델 저장")
    
    def train_models(self, X_train, X_val, y_train, y_val, engineer=None, preprocessor=None):
        """모델 학습"""
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
        
        # 리샘플링 적용
        X_train_resampled, y_train_resampled = self.apply_resampling(X_train, y_train)
        
        # LightGBM
        try:
            lgb_model, lgb_acc = self.train_lightgbm(X_train_resampled, y_train_resampled, X_val, y_val)
            model_results['lightgbm'] = lgb_acc
        except Exception as e:
            print(f"LightGBM 실패: {e}")
            model_results['lightgbm'] = 0.0
        
        # XGBoost
        try:
            xgb_model, xgb_acc = self.train_xgboost(X_train_resampled, y_train_resampled, X_val, y_val)
            model_results['xgboost'] = xgb_acc
        except Exception as e:
            print(f"XGBoost 실패: {e}")
            model_results['xgboost'] = 0.0
        
        # CatBoost
        try:
            cat_model, cat_acc = self.train_catboost(X_train_resampled, y_train_resampled, X_val, y_val)
            model_results['catboost'] = cat_acc
        except Exception as e:
            print(f"CatBoost 실패: {e}")
            model_results['catboost'] = 0.0
        
        # Random Forest
        try:
            rf_model, rf_acc = self.train_random_forest(X_train_resampled, y_train_resampled, X_val, y_val)
            model_results['random_forest'] = rf_acc
        except Exception as e:
            print(f"Random Forest 실패: {e}")
            model_results['random_forest'] = 0.0
        
        # Neural Network
        try:
            nn_model, nn_acc = self.train_neural_network(X_train_resampled, y_train_resampled, X_val, y_val)
            model_results['neural_network'] = nn_acc
        except Exception as e:
            print(f"Neural Network 실패: {e}")
            model_results['neural_network'] = 0.0
        
        # 스태킹 앙상블
        try:
            stacking_model, stacking_acc = self.create_stacking_ensemble(X_train, y_train, X_val, y_val)
            if stacking_model is not None:
                model_results['stacking'] = stacking_acc
        except Exception as e:
            print(f"스태킹 실패: {e}")
        
        # 시간 기반 교차 검증
        if len(self.models) > 0:
            full_X = pd.concat([X_train, X_val])
            full_y = pd.concat([y_train, y_val])
            
            if 'lightgbm' in self.models:
                self.perform_temporal_cv(full_X, full_y, 'lightgbm')
        
        # 모델 저장
        self.save_models(engineer, preprocessor)
        
        # 결과 출력
        print("\n모델 성능:")
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