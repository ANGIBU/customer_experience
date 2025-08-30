# model_training.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from feature_engineering import FeatureEngineer
from preprocessing import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.cv_scores = {}
        self.feature_importance = {}
        self.best_params = {}
        self.meta_model = None
        
    def prepare_training_data(self):
        """학습 데이터 준비"""
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        engineer = FeatureEngineer()
        train_df, test_df = engineer.create_features(train_df, test_df)
        
        preprocessor = DataPreprocessor()
        train_df, test_df = preprocessor.process_data(train_df, test_df)
        
        X_train, X_val, y_train, y_val, X_test, test_ids = preprocessor.prepare_data(
            train_df, test_df
        )
        
        return X_train, X_val, y_train, y_val, X_test, test_ids, engineer, preprocessor
    
    def optimize_lightgbm_params(self, X_train, y_train):
        """LightGBM 파라미터 최적화"""
        def objective(params):
            lgb_params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': int(params['num_leaves']),
                'learning_rate': params['learning_rate'],
                'feature_fraction': params['feature_fraction'],
                'bagging_fraction': params['bagging_fraction'],
                'min_child_weight': params['min_child_weight'],
                'reg_alpha': params['reg_alpha'],
                'reg_lambda': params['reg_lambda'],
                'verbose': -1,
                'random_state': 42
            }
            
            # 시간 기반 교차 검증
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
                val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
                
                model = lgb.train(
                    lgb_params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=300,
                    callbacks=[lgb.early_stopping(30)]
                )
                
                y_pred = model.predict(X_fold_val)
                loss = log_loss(y_fold_val, y_pred)
                scores.append(loss)
            
            return {'loss': np.mean(scores), 'status': STATUS_OK}
        
        space = {
            'num_leaves': hp.choice('num_leaves', [20, 31, 50, 70]),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'feature_fraction': hp.uniform('feature_fraction', 0.7, 1.0),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.7, 1.0),
            'min_child_weight': hp.uniform('min_child_weight', 1, 15),
            'reg_alpha': hp.uniform('reg_alpha', 0, 0.5),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1.0)
        }
        
        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)
        
        self.best_params['lightgbm'] = best
        return best
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """LightGBM 모델 학습"""
        print("LightGBM 학습")
        
        # 파라미터 최적화
        best_params = self.optimize_lightgbm_params(X_train, y_train)
        
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': int(best_params.get('num_leaves', 31)),
            'learning_rate': best_params.get('learning_rate', 0.05),
            'feature_fraction': best_params.get('feature_fraction', 0.9),
            'bagging_fraction': best_params.get('bagging_fraction', 0.8),
            'min_child_weight': best_params.get('min_child_weight', 5),
            'reg_alpha': best_params.get('reg_alpha', 0.1),
            'reg_lambda': best_params.get('reg_lambda', 0.1),
            'verbose': -1,
            'random_state': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50)]
        )
        
        y_pred = model.predict(X_val)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_val, y_pred_class)
        
        print(f"LightGBM 검증 정확도: {accuracy:.4f}")
        
        self.models['lightgbm'] = model
        return model, accuracy
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """XGBoost 모델 학습"""
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
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbosity': 0
        }
        
        train_data = xgb.DMatrix(X_train, label=y_train)
        val_data = xgb.DMatrix(X_val, label=y_val)
        
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
        accuracy = accuracy_score(y_val, y_pred_class)
        
        print(f"XGBoost 검증 정확도: {accuracy:.4f}")
        
        self.models['xgboost'] = model
        return model, accuracy
    
    def train_catboost(self, X_train, y_train, X_val, y_val):
        """CatBoost 모델 학습"""
        print("CatBoost 학습")
        
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            bootstrap_type='Bernoulli',
            subsample=0.8,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=50
        )
        
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"CatBoost 검증 정확도: {accuracy:.4f}")
        
        self.models['catboost'] = model
        return model, accuracy
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Random Forest 모델 학습"""
        print("Random Forest 학습")
        
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Random Forest 검증 정확도: {accuracy:.4f}")
        
        self.models['random_forest'] = model
        return model, accuracy
    
    def train_extra_trees(self, X_train, y_train, X_val, y_val):
        """Extra Trees 모델 학습"""
        print("Extra Trees 학습")
        
        model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Extra Trees 검증 정확도: {accuracy:.4f}")
        
        self.models['extra_trees'] = model
        return model, accuracy
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """신경망 모델 학습"""
        print("Neural Network 학습")
        
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Neural Network 검증 정확도: {accuracy:.4f}")
        
        self.models['neural_network'] = model
        return model, accuracy
    
    def train_svm(self, X_train, y_train, X_val, y_val):
        """SVM 모델 학습"""
        print("SVM 학습")
        
        model = SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"SVM 검증 정확도: {accuracy:.4f}")
        
        self.models['svm'] = model
        return model, accuracy
    
    def perform_time_series_cv(self, X, y, model_type='lightgbm', n_splits=5):
        """시간 기반 교차 검증"""
        print(f"{model_type.upper()} 시간 기반 CV")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            if model_type == 'lightgbm':
                lgb_params = {
                    'objective': 'multiclass',
                    'num_class': 3,
                    'metric': 'multi_logloss',
                    'verbose': -1,
                    'random_state': 42,
                    'num_leaves': 31,
                    'learning_rate': 0.05
                }
                
                train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
                model = lgb.train(lgb_params, train_data, num_boost_round=200)
                y_pred = model.predict(X_val_fold)
                y_pred_class = np.argmax(y_pred, axis=1)
                
            elif model_type == 'catboost':
                model = CatBoostClassifier(
                    iterations=200,
                    learning_rate=0.05,
                    depth=6,
                    random_seed=42,
                    verbose=0
                )
                model.fit(X_train_fold, y_train_fold)
                y_pred_class = model.predict(X_val_fold)
            
            accuracy = accuracy_score(y_val_fold, y_pred_class)
            cv_scores.append(accuracy)
            
            print(f"  Fold {fold + 1}: {accuracy:.4f}")
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        print(f"평균 CV 점수: {mean_score:.4f} (+/- {std_score:.4f})")
        
        self.cv_scores[model_type] = {
            'scores': cv_scores,
            'mean': mean_score,
            'std': std_score
        }
        
        return mean_score, std_score
    
    def create_stacking_model(self, X_train, y_train, X_val, y_val):
        """스태킹 모델 생성"""
        print("스태킹 모델 생성")
        
        # 기본 모델들의 예측값 수집
        base_models = ['lightgbm', 'xgboost', 'catboost', 'random_forest']
        base_predictions = np.zeros((X_val.shape[0], len(base_models) * 3))
        
        col_idx = 0
        for model_name in base_models:
            if model_name in self.models:
                model = self.models[model_name]
                
                if model_name == 'lightgbm':
                    pred_proba = model.predict(X_val)
                elif model_name == 'xgboost':
                    pred_proba = model.predict(xgb.DMatrix(X_val))
                else:
                    pred_proba = model.predict_proba(X_val)
                
                base_predictions[:, col_idx:col_idx+3] = pred_proba
                col_idx += 3
        
        # 메타 모델 학습
        self.meta_model = RidgeClassifier(alpha=1.0, random_state=42)
        
        # 교차 검증으로 메타 피처 생성
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        meta_features = np.zeros((X_train.shape[0], len(base_models) * 3))
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            
            fold_predictions = np.zeros((len(val_idx), len(base_models) * 3))
            col_idx = 0
            
            # 간단한 모델로 빠른 학습
            for model_name in base_models:
                if model_name == 'lightgbm':
                    lgb_params = {'objective': 'multiclass', 'num_class': 3, 'verbose': -1, 'random_state': 42}
                    train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
                    fold_model = lgb.train(lgb_params, train_data, num_boost_round=100)
                    pred_proba = fold_model.predict(X_fold_val)
                    
                elif model_name == 'catboost':
                    fold_model = CatBoostClassifier(iterations=100, random_seed=42, verbose=0)
                    fold_model.fit(X_fold_train, y_fold_train)
                    pred_proba = fold_model.predict_proba(X_fold_val)
                    
                elif model_name == 'random_forest':
                    fold_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    fold_model.fit(X_fold_train, y_fold_train)
                    pred_proba = fold_model.predict_proba(X_fold_val)
                    
                else:
                    continue
                
                fold_predictions[:, col_idx:col_idx+3] = pred_proba
                col_idx += 3
            
            meta_features[val_idx] = fold_predictions
        
        # 메타 모델 학습
        self.meta_model.fit(meta_features, y_train)
        
        # 스태킹 예측
        stacking_pred = self.meta_model.predict(base_predictions)
        stacking_accuracy = accuracy_score(y_val, stacking_pred)
        
        print(f"스태킹 검증 정확도: {stacking_accuracy:.4f}")
        
        self.models['stacking'] = self.meta_model
        return self.meta_model, stacking_accuracy
    
    def create_voting_ensemble(self, X_train, y_train, X_val, y_val):
        """보팅 앙상블 생성"""
        print("보팅 앙상블 생성")
        
        # 확률 기반 소프트 보팅을 위한 모델들
        voting_models = []
        
        if 'random_forest' in self.models:
            voting_models.append(('rf', self.models['random_forest']))
        if 'extra_trees' in self.models:
            voting_models.append(('et', self.models['extra_trees']))
        if 'neural_network' in self.models:
            voting_models.append(('nn', self.models['neural_network']))
        if 'svm' in self.models:
            voting_models.append(('svm', self.models['svm']))
        
        if len(voting_models) >= 2:
            voting_classifier = VotingClassifier(
                estimators=voting_models,
                voting='soft'
            )
            
            voting_classifier.fit(X_train, y_train)
            
            y_pred = voting_classifier.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            print(f"보팅 앙상블 검증 정확도: {accuracy:.4f}")
            
            self.models['voting'] = voting_classifier
            return voting_classifier, accuracy
        
        return None, 0.0
    
    def save_all_models(self, engineer, preprocessor):
        """모든 모델 저장"""
        print("모델 저장")
        
        # 전처리기와 피처 엔지니어 저장
        joblib.dump(preprocessor, 'models/preprocessor.pkl')
        joblib.dump(engineer, 'models/feature_engineer.pkl')
        
        # 개별 모델 저장
        for name, model in self.models.items():
            if name == 'lightgbm':
                model.save_model(f'models/{name}_model.txt')
            elif name == 'xgboost':
                model.save_model(f'models/{name}_model.json')
            else:
                joblib.dump(model, f'models/{name}_model.pkl')
            
            print(f"  {name} 모델 저장 완료")
    
    def train_models(self, X_train, X_val, y_train, y_val):
        """모든 모델 학습"""
        print("모델 학습 시작")
        print("=" * 40)
        
        # 개별 모델 학습
        lgb_model, lgb_acc = self.train_lightgbm(X_train, y_train, X_val, y_val)
        xgb_model, xgb_acc = self.train_xgboost(X_train, y_train, X_val, y_val)
        cat_model, cat_acc = self.train_catboost(X_train, y_train, X_val, y_val)
        rf_model, rf_acc = self.train_random_forest(X_train, y_train, X_val, y_val)
        et_model, et_acc = self.train_extra_trees(X_train, y_train, X_val, y_val)
        nn_model, nn_acc = self.train_neural_network(X_train, y_train, X_val, y_val)
        svm_model, svm_acc = self.train_svm(X_train, y_train, X_val, y_val)
        
        # 앙상블 모델 생성
        stacking_model, stacking_acc = self.create_stacking_model(X_train, y_train, X_val, y_val)
        voting_model, voting_acc = self.create_voting_ensemble(X_train, y_train, X_val, y_val)
        
        # 교차 검증
        full_X = pd.concat([X_train, X_val])
        full_y = pd.concat([y_train, y_val])
        
        self.perform_time_series_cv(full_X, full_y, 'lightgbm')
        self.perform_time_series_cv(full_X, full_y, 'catboost')
        
        # 모델 저장
        engineer = FeatureEngineer()
        preprocessor = DataPreprocessor()
        self.save_all_models(engineer, preprocessor)
        
        print("\n모델 성능 요약:")
        print(f"LightGBM: {lgb_acc:.4f}")
        print(f"XGBoost: {xgb_acc:.4f}")
        print(f"CatBoost: {cat_acc:.4f}")
        print(f"Random Forest: {rf_acc:.4f}")
        print(f"Extra Trees: {et_acc:.4f}")
        print(f"Neural Network: {nn_acc:.4f}")
        print(f"SVM: {svm_acc:.4f}")
        if stacking_acc > 0:
            print(f"Stacking: {stacking_acc:.4f}")
        if voting_acc > 0:
            print(f"Voting: {voting_acc:.4f}")

def main():
    trainer = ModelTrainer()
    X_train, X_val, y_train, y_val, X_test, test_ids, engineer, preprocessor = trainer.prepare_training_data()
    trainer.train_models(X_train, X_val, y_train, y_val)
    
    return trainer

if __name__ == "__main__":
    main()