# model_training.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
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
    """모델 학습 클래스"""
    
    def __init__(self):
        self.models = {}
        self.calibrated_models = {}
        self.cv_scores = {}
        self.feature_importance = {}
        self.best_params = {}
        
    def prepare_data(self):
        """데이터 준비"""
        print("데이터 전처리 및 피처 엔지니어링")
        
        # 데이터 로드
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        # 피처 엔지니어링
        feature_engineer = FeatureEngineer()
        train_df, test_df = feature_engineer.process_all_features(train_df, test_df)
        
        # 전처리
        preprocessor = DataPreprocessor()
        train_df, test_df = preprocessor.process_complete_pipeline(train_df, test_df)
        
        # 모델링용 데이터 준비
        X_train, X_val, y_train, y_val, X_test, test_ids = preprocessor.prepare_model_data(
            train_df, test_df
        )
        
        return X_train, X_val, y_train, y_val, X_test, test_ids, feature_engineer, preprocessor
    
    def optimize_lightgbm(self, X_train, y_train, X_val, y_val):
        """LightGBM 하이퍼파라미터 최적화"""
        print("=== LightGBM 하이퍼파라미터 최적화 ===")
        
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
                'bagging_freq': int(params['bagging_freq']),
                'min_child_weight': params['min_child_weight'],
                'reg_alpha': params['reg_alpha'],
                'reg_lambda': params['reg_lambda'],
                'verbose': -1,
                'random_state': 42
            }
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=500,
                callbacks=[lgb.early_stopping(50)]
            )
            
            y_pred = model.predict(X_val)
            loss = log_loss(y_val, y_pred)
            
            return {'loss': loss, 'status': STATUS_OK}
        
        space = {
            'num_leaves': hp.choice('num_leaves', [31, 50, 70, 100]),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'feature_fraction': hp.uniform('feature_fraction', 0.6, 1.0),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 1.0),
            'bagging_freq': hp.choice('bagging_freq', [1, 3, 5]),
            'min_child_weight': hp.uniform('min_child_weight', 1, 20),
            'reg_alpha': hp.uniform('reg_alpha', 0, 1),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1)
        }
        
        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=30, trials=trials)
        
        self.best_params['lightgbm'] = best
        return best
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val, params=None):
        """LightGBM 모델 학습"""
        print("=== LightGBM 모델 학습 ===")
        
        if params is None:
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(0)
            ]
        )
        
        y_pred = model.predict(X_val)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_val, y_pred_class)
        
        print(f"LightGBM 검증 정확도: {accuracy:.4f}")
        
        self.models['lightgbm'] = model
        self.feature_importance['lightgbm'] = model.feature_importance()
        
        return model, accuracy
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """XGBoost 모델 학습"""
        print("=== XGBoost 모델 학습 ===")
        
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
            early_stopping_rounds=100,
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
        print("=== CatBoost 모델 학습 ===")
        
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            border_count=128,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=100
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
        self.feature_importance['catboost'] = model.feature_importances_
        
        return model, accuracy
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Random Forest 모델 학습"""
        print("=== Random Forest 모델 학습 ===")
        
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Random Forest 검증 정확도: {accuracy:.4f}")
        
        self.models['random_forest'] = model
        self.feature_importance['random_forest'] = model.feature_importances_
        
        return model, accuracy
    
    def train_extra_trees(self, X_train, y_train, X_val, y_val):
        """Extra Trees 모델 학습"""
        print("=== Extra Trees 모델 학습 ===")
        
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Extra Trees 검증 정확도: {accuracy:.4f}")
        
        self.models['extra_trees'] = model
        self.feature_importance['extra_trees'] = model.feature_importances_
        
        return model, accuracy
    
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """Logistic Regression 모델 학습"""
        print("=== Logistic Regression 모델 학습 ===")
        
        model = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Logistic Regression 검증 정확도: {accuracy:.4f}")
        
        self.models['logistic'] = model
        
        return model, accuracy
    
    def calibrate_models(self, X_train, y_train):
        """모델 확률 보정"""
        print("=== 모델 확률 보정 ===")
        
        for name, model in self.models.items():
            if name not in ['lightgbm', 'xgboost']:  # 이미 확률을 반환하는 모델 제외
                print(f"{name} 보정 중...")
                calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
                calibrated.fit(X_train, y_train)
                self.calibrated_models[name] = calibrated
    
    def cross_validation(self, X, y, model_name='lightgbm', cv_folds=5):
        """교차 검증"""
        print(f"=== {model_name.upper()} 교차 검증 (K={cv_folds}) ===")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            if model_name == 'lightgbm':
                lgb_params = {
                    'objective': 'multiclass',
                    'num_class': 3,
                    'metric': 'multi_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': 42
                }
                
                train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
                val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
                
                model = lgb.train(
                    lgb_params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=1000,
                    callbacks=[lgb.early_stopping(100)]
                )
                
                y_pred = model.predict(X_val_fold)
                y_pred_class = np.argmax(y_pred, axis=1)
                
            elif model_name == 'catboost':
                model = CatBoostClassifier(
                    iterations=1000,
                    learning_rate=0.05,
                    depth=6,
                    random_seed=42,
                    verbose=0,
                    early_stopping_rounds=100
                )
                
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=(X_val_fold, y_val_fold),
                    use_best_model=True
                )
                
                y_pred_class = model.predict(X_val_fold)
            
            fold_accuracy = accuracy_score(y_val_fold, y_pred_class)
            cv_scores.append(fold_accuracy)
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        print(f"{model_name.upper()} CV 평균 정확도: {mean_score:.4f} (+/- {std_score:.4f})")
        
        self.cv_scores[model_name] = {
            'scores': cv_scores,
            'mean': mean_score,
            'std': std_score
        }
        
        return mean_score, std_score
    
    def create_stacking_ensemble(self, X_train, y_train, X_val, y_val):
        """스태킹 앙상블 생성"""
        print("=== 스태킹 앙상블 생성 ===")
        
        # 1단계: 베이스 모델 예측 수집
        base_predictions = np.zeros((X_val.shape[0], len(self.models) * 3))  # 3클래스
        
        col_idx = 0
        for name, model in self.models.items():
            if name == 'lightgbm':
                pred_proba = model.predict(X_val)
            elif name == 'xgboost':
                pred_proba = model.predict(xgb.DMatrix(X_val))
            else:
                pred_proba = model.predict_proba(X_val)
            
            base_predictions[:, col_idx:col_idx+3] = pred_proba
            col_idx += 3
        
        # 2단계: 메타 모델 학습
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
        
        # 훈련 데이터에 대한 베이스 예측 (교차 검증 방식)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_meta_features = np.zeros((X_train.shape[0], len(self.models) * 3))
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            
            fold_predictions = np.zeros((len(val_idx), len(self.models) * 3))
            col_idx = 0
            
            for name, _ in self.models.items():
                # 각 폴드에서 모델 재학습
                if name == 'lightgbm':
                    fold_model = lgb.train(
                        {'objective': 'multiclass', 'num_class': 3, 'verbose': -1, 'random_state': 42},
                        lgb.Dataset(X_fold_train, label=y_fold_train),
                        num_boost_round=100
                    )
                    pred_proba = fold_model.predict(X_fold_val)
                elif name == 'catboost':
                    fold_model = CatBoostClassifier(iterations=100, random_seed=42, verbose=0)
                    fold_model.fit(X_fold_train, y_fold_train)
                    pred_proba = fold_model.predict_proba(X_fold_val)
                else:
                    fold_model = self.models[name]
                    fold_model.fit(X_fold_train, y_fold_train)
                    pred_proba = fold_model.predict_proba(X_fold_val)
                
                fold_predictions[:, col_idx:col_idx+3] = pred_proba
                col_idx += 3
            
            train_meta_features[val_idx] = fold_predictions
        
        # 메타 모델 학습
        meta_model.fit(train_meta_features, y_train)
        
        # 검증 데이터에 대한 최종 예측
        stacking_pred_proba = meta_model.predict_proba(base_predictions)
        stacking_pred = np.argmax(stacking_pred_proba, axis=1)
        stacking_accuracy = accuracy_score(y_val, stacking_pred)
        
        print(f"스태킹 앙상블 검증 정확도: {stacking_accuracy:.4f}")
        
        self.models['stacking'] = meta_model
        return meta_model, stacking_accuracy, base_predictions
    
    def save_models(self, feature_engineer, preprocessor):
        """모델 저장"""
        print("=== 모델 저장 ===")
        
        # 폴더 생성
        pkl_dir = 'models/pkl'
        json_dir = 'models/json'
        os.makedirs(pkl_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)
        
        # 전처리기와 피처 엔지니어 저장
        joblib.dump(preprocessor, os.path.join(pkl_dir, 'preprocessor.pkl'))
        joblib.dump(feature_engineer, os.path.join(pkl_dir, 'feature_engineer.pkl'))
        print("전처리기 및 피처 엔지니어 저장 완료")
        
        # 개별 모델 저장
        for name, model in self.models.items():
            if name == 'lightgbm':
                model_path = os.path.join(json_dir, f'{name}_model.txt')
                model.save_model(model_path)
            elif name == 'xgboost':
                model_path = os.path.join(json_dir, f'{name}_model.json')
                model.save_model(model_path)
            else:
                model_path = os.path.join(pkl_dir, f'{name}_model.pkl')
                joblib.dump(model, model_path)
            
            print(f"{name} 모델 저장 완료: {model_path}")
    
    def train_all_models(self):
        """모든 모델 학습"""
        print("모델 학습 시작")
        print("="*40)
        
        # 데이터 준비
        X_train, X_val, y_train, y_val, X_test, test_ids, feature_engineer, preprocessor = self.prepare_data()
        
        # 개별 모델 학습
        lgb_model, lgb_acc = self.train_lightgbm(X_train, y_train, X_val, y_val)
        xgb_model, xgb_acc = self.train_xgboost(X_train, y_train, X_val, y_val)
        cat_model, cat_acc = self.train_catboost(X_train, y_train, X_val, y_val)
        rf_model, rf_acc = self.train_random_forest(X_train, y_train, X_val, y_val)
        et_model, et_acc = self.train_extra_trees(X_train, y_train, X_val, y_val)
        lr_model, lr_acc = self.train_logistic_regression(X_train, y_train, X_val, y_val)
        
        # 모델 보정
        self.calibrate_models(X_train, y_train)
        
        # 교차 검증
        full_X = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        full_y = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        
        self.cross_validation(full_X, full_y, 'lightgbm')
        self.cross_validation(full_X, full_y, 'catboost')
        
        # 스태킹 앙상블
        stacking_model, stacking_acc, base_preds = self.create_stacking_ensemble(X_train, y_train, X_val, y_val)
        
        # 모델 저장
        self.save_models(feature_engineer, preprocessor)
        
        # 결과 요약
        print("\n=== 모델 성능 요약 ===")
        print(f"LightGBM: {lgb_acc:.4f}")
        print(f"XGBoost: {xgb_acc:.4f}")
        print(f"CatBoost: {cat_acc:.4f}")
        print(f"Random Forest: {rf_acc:.4f}")
        print(f"Extra Trees: {et_acc:.4f}")
        print(f"Logistic Regression: {lr_acc:.4f}")
        print(f"Stacking Ensemble: {stacking_acc:.4f}")
        
        return X_test, test_ids

def main():
    """메인 실행 함수"""
    trainer = ModelTrainer()
    X_test, test_ids = trainer.train_all_models()
    
    print("\n모델 학습 완료!")
    return trainer, X_test, test_ids

if __name__ == "__main__":
    main()