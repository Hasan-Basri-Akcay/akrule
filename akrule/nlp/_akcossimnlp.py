import pandas as pd
import numpy as np
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from kneed import KneeLocator


class AKCosSimNLP():
    """
    """
    
    def __init__(self, data: pd.DataFrame, n_components: int=None, spacy_path: str= "en_core_web_sm", round: int=6):
        self.data = data.copy()
        self.nlp = spacy.load(spacy_path)
        self.n_components = n_components
        self.round = round
        self.vectorizer = None
        self.model_nmf = None
        self.norm_matrix = None
    
    def _pca_find_best_component_num(self, data):
        model_pca = PCA(random_state=0).fit(data)
        error_list = np.cumsum(model_pca.explained_variance_ratio_)
        error_list = error_list * -1
        error_list = error_list - error_list.min()
        
        kn = KneeLocator(np.arange(len(error_list)), error_list, curve='convex', direction='decreasing')
        return int(kn.knee)
    
    def _get_lemma(self, text):
        # Create doc object
        doc = self.nlp(text)
        # Generate list of lemmas
        lemma = [token.lemma_ for token in doc if (not token.is_stop)&(not token.is_punct)&
                 (not token.is_space)&(not token.is_digit)&(token.is_alpha)]
        return ' '.join(lemma).lower()
    
    def _transform(self, X: pd.DataFrame):        
        lemma_list = []
        for row in X:    
            lemma = self._get_lemma(row)
            lemma_list.append(lemma)
        X["LEMMA"] = lemma_list
        return X["LEMMA"]
    
    def fit(self, X: pd.DataFrame):
        self.fit_transform(X)
        
    def fit_transform(self, X: pd.DataFrame):        
        # Preprocesses
        X = self._transform(X.copy())
        
        # Vectorization
        self.vectorizer = TfidfVectorizer(encoding='utf-8', ngram_range=(1, 1))
        tfidf_matrix = self.vectorizer.fit_transform(X)
        
        # Dimension Reduction
        if self.n_components is None: self.n_components = self._pca_find_best_component_num(tfidf_matrix.toarray())
        self.model_nmf = NMF(n_components=self.n_components, max_iter=1000, random_state=0)
        nmf_matrix = self.model_nmf.fit_transform(tfidf_matrix)
        
        # Normalization 
        self.norm_matrix = normalize(nmf_matrix, norm="l2") 
        return pd.Series(X, name="CLEAN_DATA")
    
    def predict(self, text: str):
        X = pd.Series([text], name="text")  
        
        # Preprocesses
        X = self._transform(X.copy())
        
        # Vectorization
        tfidf_matrix = self.vectorizer.transform(X)
        
        # Dimension Reduction
        nmf_matrix = self.model_nmf.transform(tfidf_matrix)
        
        # Normalization
        norm_matrix = normalize(nmf_matrix, norm="l2") 
        
        # Cosine Similarity
        similarity_text = self.norm_matrix.dot(norm_matrix.reshape(-1))
        
        self.data["COS_SIMILARITY"] = similarity_text  
        if self.round > 0: self.data["COS_SIMILARITY"] = self.data["COS_SIMILARITY"].round(self.round)      
        return self.data.sort_values("COS_SIMILARITY", ascending=False)