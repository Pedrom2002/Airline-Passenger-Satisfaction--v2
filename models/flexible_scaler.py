# models/flexible_scaler.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import json

class FlexibleScaler(BaseEstimator, TransformerMixin):
    """
    Scaler flexível que pode lidar com features opcionais
    """
    
    def __init__(self, required_features=None):
        self.required_features = required_features or []
        self.scaler = StandardScaler()
        self.feature_defaults = {}
        self.feature_indices = {}
        self.n_features_expected = 0
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """
        Fit the scaler and calculate default values for each feature
        """
        try:
            if isinstance(X, pd.DataFrame):
                X_array = X.values
                if not self.required_features:
                    self.required_features = list(X.columns)
            else:
                X_array = np.array(X)
            
            # Criar mapeamento de índices
            self.feature_indices = {feature: i for i, feature in enumerate(self.required_features)}
            
            # Calcular valores padrão (usar valores simples para evitar problemas de serialização)
            self.feature_defaults = {}
            for i, feature in enumerate(self.required_features):
                if i < X_array.shape[1]:
                    # Converter para float nativo do Python (não numpy)
                    self.feature_defaults[feature] = float(np.median(X_array[:, i]))
                else:
                    self.feature_defaults[feature] = 0.0
            
            # Fit do scaler padrão
            self.scaler.fit(X_array)
            self.n_features_expected = len(self.required_features)
            self.is_fitted = True
            
            return self
            
        except Exception as e:
            print(f"Erro no fit: {e}")
            raise
    
    def transform(self, X):
        """
        Transform data, handling missing features
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
            
        try:
            if isinstance(X, pd.DataFrame):
                X_array = X.values
                input_features = list(X.columns)
            else:
                X_array = np.array(X)
                input_features = [f"feature_{i}" for i in range(X_array.shape[1])]
            
            # Se já temos o número correto de features, usar diretamente
            if X_array.shape[1] == self.n_features_expected:
                return self.scaler.transform(X_array)
            
            # Caso contrário, construir array completo
            n_samples = X_array.shape[0]
            X_complete = np.zeros((n_samples, self.n_features_expected))
            
            # Preencher features disponíveis
            for i, feature in enumerate(self.required_features):
                if feature in input_features:
                    input_idx = input_features.index(feature)
                    if input_idx < X_array.shape[1]:
                        X_complete[:, i] = X_array[:, input_idx]
                    else:
                        X_complete[:, i] = self.feature_defaults[feature]
                else:
                    # Feature não disponível, usar valor padrão
                    X_complete[:, i] = self.feature_defaults[feature]
            
            return self.scaler.transform(X_complete)
            
        except Exception as e:
            print(f"Erro no transform: {e}")
            raise
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """
        Compatibilidade com sklearn
        """
        return np.array(self.required_features, dtype=object)
    
    def __getstate__(self):
        """
        Preparar o objeto para serialização
        """
        state = self.__dict__.copy()
        # Converter numpy types para Python nativos
        if 'feature_defaults' in state:
            state['feature_defaults'] = {k: float(v) for k, v in state['feature_defaults'].items()}
        return state
    
    def __setstate__(self, state):
        """
        Restaurar o objeto após deserialização
        """
        self.__dict__.update(state)
    
    def save(self, filepath):
        """
        Salvar o scaler usando joblib (mais robusto)
        """
        try:
            # Salvar metadados separadamente
            metadata = {
                'required_features': self.required_features,
                'feature_defaults': self.feature_defaults,
                'feature_indices': self.feature_indices,
                'n_features_expected': self.n_features_expected,
                'is_fitted': self.is_fitted
            }
            
            # Salvar o StandardScaler separadamente
            joblib.dump({
                'scaler': self.scaler,
                'metadata': metadata
            }, filepath)
            
        except Exception as e:
            print(f"Erro ao salvar: {e}")
            raise
    
    @classmethod
    def load(cls, filepath):
        """
        Carregar o scaler
        """
        try:
            data = joblib.load(filepath)
            
            # Criar nova instância
            instance = cls()
            
            # Restaurar o scaler
            instance.scaler = data['scaler']
            
            # Restaurar metadados
            metadata = data['metadata']
            instance.required_features = metadata['required_features']
            instance.feature_defaults = metadata['feature_defaults']
            instance.feature_indices = metadata['feature_indices']
            instance.n_features_expected = metadata['n_features_expected']
            instance.is_fitted = metadata['is_fitted']
            
            return instance
            
        except Exception as e:
            print(f"Erro ao carregar: {e}")
            raise