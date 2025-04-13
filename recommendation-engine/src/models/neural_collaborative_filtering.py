"""
Module untuk implementasi Neural Collaborative Filtering (NCF)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import sys
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable
import json
import pickle
from datetime import datetime
import traceback

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Setup logging
from central_logging import get_logger
logger = get_logger(__name__)

from config.config import (
    PROCESSED_DATA_PATH,
    MODELS_PATH,
    NCF_EMBEDDING_SIZE,
    NCF_LAYERS,
    NCF_LEARNING_RATE,
    NCF_BATCH_SIZE,
    NCF_NUM_EPOCHS,
    NCF_VALIDATION_RATIO,
    NCF_NEGATIVE_RATIO
)

class NCF(nn.Module):
    """
    Neural Collaborative Filtering Model
    
    Implementasi dari paper "Neural Collaborative Filtering" (He et al., 2017)
    dengan modifikasi untuk konteks proyek Web3.
    """
    
    def __init__(self, num_users: int, num_items: int, embedding_size: int = NCF_EMBEDDING_SIZE, layers: Optional[List[int]] = None):
        """
        Inisialisasi model NCF
        
        Args:
            num_users: Jumlah pengguna
            num_items: Jumlah proyek Web3
            embedding_size: Dimensi embedding
            layers: List ukuran layer MLP
        """
        super(NCF, self).__init__()

        # Default layers if None
        if layers is None:
            layers = NCF_LAYERS
        
        # User dan Item embeddings untuk GMF (General Matrix Factorization)
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_size)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_size)
        
        # User dan Item embeddings untuk MLP
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_size)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_size)
        
        # Bangun MLP layers
        self.mlp_layers = nn.ModuleList()
        input_size = 2 * embedding_size
        
        for i, layer_size in enumerate(layers):
            self.mlp_layers.append(nn.Linear(input_size, layer_size))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.BatchNorm1d(layer_size))
            self.mlp_layers.append(nn.Dropout(p=0.1))
            input_size = layer_size
        
        # Output layer
        self.output_layer = nn.Linear(layers[-1] + embedding_size, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Inisialisasi weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """
        Inisialisasi weights dengan distribusi normal
        """
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            user_indices: Indeks user
            item_indices: Indeks item
            
        Returns:
            torch.Tensor: Prediksi rating/preferensi
        """
        # GMF path
        user_embedding_gmf = self.user_embedding_gmf(user_indices)
        item_embedding_gmf = self.item_embedding_gmf(item_indices)
        gmf_vector = user_embedding_gmf * item_embedding_gmf
        
        # MLP path
        user_embedding_mlp = self.user_embedding_mlp(user_indices)
        item_embedding_mlp = self.item_embedding_mlp(item_indices)
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        
        # MLP forward
        for i, layer in enumerate(self.mlp_layers):
            mlp_vector = layer(mlp_vector)
        
        # Concatenate GMF dan MLP outputs
        concat_vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
        
        # Output layer
        prediction = self.output_layer(concat_vector)
        rating = self.sigmoid(prediction)
        
        return rating.view(-1)
    
    def predict(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        """
        Prediksi rating untuk satu user dan beberapa item
        
        Args:
            user_id: User ID
            item_ids: List item IDs
            
        Returns:
            np.ndarray: Prediksi rating
        """
        self.eval()
        with torch.no_grad():
            user_indices = torch.tensor([user_id] * len(item_ids), device=next(self.parameters()).device)
            item_indices = torch.tensor(item_ids, device=next(self.parameters()).device)
            ratings = self.forward(user_indices, item_indices)
            return ratings.cpu().numpy()

class NCFRecommender:
    """
    Neural Collaborative Filtering Recommender untuk proyek Web3
    """
    
    def __init__(self, 
                 user_item_df: pd.DataFrame, 
                 projects_df: pd.DataFrame, 
                 embedding_size: int = NCF_EMBEDDING_SIZE, 
                 layers: Optional[List[int]] = None, 
                 learning_rate: float = NCF_LEARNING_RATE, 
                 batch_size: int = NCF_BATCH_SIZE, 
                 num_epochs: int = NCF_NUM_EPOCHS, 
                 gpu: bool = False):
        """
        Inisialisasi NCF Recommender
        
        Args:
            user_item_df: User-item interaction matrix
            projects_df: DataFrame proyek Web3
            embedding_size: Ukuran embedding
            layers: Ukuran layer MLP
            learning_rate: Learning rate
            batch_size: Batch size
            num_epochs: Jumlah epoch
            gpu: Gunakan GPU jika tersedia
        """
        self.user_item_df = user_item_df
        self.projects_df = projects_df
        self.embedding_size = embedding_size
        # Default layers if None
        if layers is None:
            self.layers = NCF_LAYERS
        else:
            self.layers = layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Set device
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Mapping ID ke indeks
        self.user_ids = user_item_df.index.tolist()
        self.item_ids = user_item_df.columns.tolist()
        
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.item_id_map = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)
        
        # Inisialisasi model
        self.model = NCF(
            self.num_users, 
            self.num_items, 
            embedding_size=self.embedding_size,
            layers=self.layers
        )
        self.model = self.model.to(self.device)
        
        # Optimizer dengan weight_decay
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5  # Weight decay untuk regularisasi
        )
        self.criterion = nn.BCELoss()
        
        # Metrik untuk tracking
        self.train_losses = []
        self.val_losses = []
    
    def _preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Konversi data ke format training untuk NCF
        
        Returns:
            tuple: (user_indices, item_indices, ratings)
        """
        try:
            # Create list of (user, item, rating) tuples dari user-item matrix
            interactions = []
            
            for user_id in self.user_ids:
                user_idx = self.user_id_map[user_id]
                
                for item_id in self.item_ids:
                    rating = self.user_item_df.loc[user_id, item_id]
                    
                    if rating > 0:  # Hanya ambil interaksi positif
                        item_idx = self.item_id_map[item_id]
                        interactions.append((user_idx, item_idx, rating / 5.0))  # Normalize to 0-1
            
            # Convert to arrays
            if not interactions:
                logger.warning("No positive interactions found")
                return np.array([]), np.array([]), np.array([])
                
            interactions = np.array(interactions)
            user_indices = interactions[:, 0].astype(np.int64)
            item_indices = interactions[:, 1].astype(np.int64)
            user_ratings = interactions[:, 2].astype(np.float32)
            
            return user_indices, item_indices, user_ratings
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            logger.debug(traceback.format_exc())
            return np.array([]), np.array([]), np.array([])
    
    def _generate_negative_samples(self, 
                                  user_indices: np.ndarray, 
                                  item_indices: np.ndarray, 
                                  user_ratings: np.ndarray, 
                                  negative_ratio: int = NCF_NEGATIVE_RATIO) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate negative samples untuk training
        
        Args:
            user_indices: User indices
            item_indices: Item indices
            user_ratings: User ratings untuk item positif
            negative_ratio: Rasio negative:positive samples
            
        Returns:
            tuple: (all_user_indices, all_item_indices, all_ratings)
        """
        try:
            # Menggunakan numpy random generator terbaru
            rng = np.random.default_rng(42)

            # Buat set interaksi positif untuk lookup cepat
            positive_interactions = set()
            for u, i in zip(user_indices, item_indices):
                positive_interactions.add((u, i))
            
            # Generate negative samples
            negative_samples = []
            
            for u in range(self.num_users):
                positive_items = [i for i in range(self.num_items) if (u, i) in positive_interactions]
                if not positive_items:
                    continue
                    
                # Pilih num_positive * negative_ratio item yang tidak diinteraksikan
                num_positive = len(positive_items)
                num_negative = min(num_positive * negative_ratio, self.num_items - num_positive)
                
                # Pilih secara acak dari item yang tidak diinteraksikan
                all_items = set(range(self.num_items))
                negative_items = list(all_items - set(positive_items))
                
                if len(negative_items) > num_negative:
                    negative_items = rng.choice(negative_items, num_negative, replace=False)
                
                for i in negative_items:
                    negative_samples.append((u, i, 0.0))
            
            # Gabungkan positive dan negative samples
            negative_samples = np.array(negative_samples)

            # Combine dengan samples positif
            all_user_indices = np.concatenate([user_indices, negative_samples[:, 0].astype(np.int64)])
            all_item_indices = np.concatenate([item_indices, negative_samples[:, 1].astype(np.int64)])
            all_ratings = np.concatenate([user_ratings, negative_samples[:, 2].astype(np.float32)])
            
            # Shuffle
            indices = np.arange(len(all_ratings))
            rng.shuffle(indices)
            
            all_user_indices = all_user_indices[indices]
            all_item_indices = all_item_indices[indices]
            all_ratings = all_ratings[indices]
            
            return all_user_indices, all_item_indices, all_ratings
            
        except Exception as e:
            logger.error(f"Error generating negative samples: {e}")
            logger.debug(traceback.format_exc())
            return user_indices, item_indices, user_ratings
    
    def train(self, val_ratio: float = NCF_VALIDATION_RATIO) -> Dict[str, List[float]]:
        """
        Melatih model NCF
        
        Args:
            val_ratio: Rasio data validasi
            
        Returns:
            dict: Metrik training
        """
        try:
            logger.info("Preprocessing data for NCF training")
            
            # Preprocess data
            user_indices, item_indices, user_ratings = self._preprocess_data()
            
            if len(user_indices) == 0:
                logger.error("No valid interactions available for training")
                return {"train_loss": [], "val_loss": []}
            
            # Generate negative samples
            user_indices, item_indices, user_ratings = self._generate_negative_samples(
                user_indices, item_indices, user_ratings
            )
            
            # Train-validation split
            x_train, x_val, y_train, y_val = train_test_split(
                np.column_stack((user_indices, item_indices)),
                user_ratings,
                test_size=val_ratio,
                random_state=42
            )
            
            train_user_indices, train_item_indices = x_train[:, 0], x_train[:, 1]
            val_user_indices, val_item_indices = x_val[:, 0], x_val[:, 1]
            
            # Convert to torch tensors
            train_user_indices = torch.LongTensor(train_user_indices).to(self.device)
            train_item_indices = torch.LongTensor(train_item_indices).to(self.device)
            train_ratings = torch.FloatTensor(y_train).to(self.device)
            
            val_user_indices = torch.LongTensor(val_user_indices).to(self.device)
            val_item_indices = torch.LongTensor(val_item_indices).to(self.device)
            val_ratings = torch.FloatTensor(y_val).to(self.device)
            
            # Buat dataset dan dataloader
            train_dataset = torch.utils.data.TensorDataset(
                train_user_indices, train_item_indices, train_ratings
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                num_workers=0
            )
            
            # Reset loss tracking
            self.train_losses = []
            self.val_losses = []
            
            # Training loop
            logger.info(f"Starting training for {self.num_epochs} epochs")
            
            # Use early stopping
            best_val_loss = float('inf')
            patience = 3
            patience_counter = 0
            
            for epoch in range(self.num_epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                
                for batch_idx, (u, i, r) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    predictions = self.model(u, i)
                    loss = self.criterion(predictions, r)
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                self.train_losses.append(train_loss)
                
                # Validasi
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(val_user_indices, val_item_indices)
                    val_loss = self.criterion(val_predictions, val_ratings).item()
                    self.val_losses.append(val_loss)
                
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
            
            logger.info("Training completed")
            
            return {
                "train_loss": self.train_losses,
                "val_loss": self.val_losses
            }
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            logger.debug(traceback.format_exc())
            return {
                "train_loss": self.train_losses,
                "val_loss": self.val_losses,
                "error": str(e)
            }
    
    def recommend_projects(self, user_id: str, n: int = 10, exclude_interacted: bool = True) -> List[Dict[str, Any]]:
        """
        Memberikan rekomendasi proyek Web3 untuk pengguna
        
        Args:
            user_id: User ID
            n: Jumlah rekomendasi
            exclude_interacted: Exclude proyek yang sudah diinteraksikan
            
        Returns:
            list: List rekomendasi proyek dengan detail
        """
        try:
            if user_id not in self.user_id_map:
                logger.warning(f"User {user_id} not found in training data")
                return []
            
            # Dapatkan indeks user
            user_idx = self.user_id_map[user_id]
            
            # Dapatkan item yang sudah diinteraksikan
            interacted_items = set()
            if exclude_interacted:
                user_row = self.user_item_df.loc[user_id]
                for item_id, rating in zip(user_row.index, user_row.values):
                    if rating > 0:
                        if item_id in self.item_id_map:
                            interacted_items.add(self.item_id_map[item_id])
            
            # Dapatkan kandidat item (semua item kecuali yang sudah diinteraksikan)
            candidate_items = [i for i in range(self.num_items) if i not in interacted_items]
            
            if not candidate_items:
                logger.warning(f"No candidate items available for user {user_id}")
                return []
            
            # Konversi item indices ke original IDs
            candidate_item_ids = [self.item_ids[i] for i in candidate_items]
            
            # Prediksi rating
            predicted_ratings = self.model.predict(user_idx, candidate_items)
            
            # Urutkan item berdasarkan rating
            item_rating_pairs = list(zip(candidate_item_ids, predicted_ratings))
            item_rating_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Ambil top-n items
            top_items = item_rating_pairs[:n]
            
            # Gabungkan dengan data proyek
            recommendations = []
            
            for item_id, rating in top_items:
                project_data = self.projects_df[self.projects_df['id'] == item_id]
                
                if not project_data.empty:
                    project_info = project_data.iloc[0].to_dict()
                    
                    # Convert numpy values to Python native types for JSON serialization
                    for key, value in project_info.items():
                        if isinstance(value, (np.int64, np.int32)):
                            project_info[key] = int(value)
                        elif isinstance(value, (np.float64, np.float32)):
                            project_info[key] = float(value)
                    
                    # Add recommendation score
                    project_info['recommendation_score'] = float(rating)
                    
                    recommendations.append(project_info)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    def evaluate(self, test_user_item_df: pd.DataFrame, metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluasi model pada test data
        
        Args:
            test_user_item_df: Test user-item matrix
            metrics: List metrik evaluasi
            
        Returns:
            dict: Hasil evaluasi
        """
        if metrics is None:
            metrics = ['precision', 'recall', 'ndcg']
        
        try:
            self.model.eval()
            results = {}
            
            # Evaluasi untuk setiap user di test set
            for user_id in test_user_item_df.index:
                # Skip jika user tidak ada di training data
                if user_id not in self.user_id_map:
                    continue
                
                # Get ground truth (proyek dengan rating > 0)
                ground_truth = test_user_item_df.loc[user_id]
                relevant_items = ground_truth[ground_truth > 0].index.tolist()
                
                if not relevant_items:
                    continue
                
                # Generate recommendations
                recommendations = self.recommend_projects(user_id, n=10)
                recommended_items = [rec['id'] for rec in recommendations]
                
                # Calculate metrics
                if 'precision' in metrics:
                    precision = self._calculate_precision(relevant_items, recommended_items)
                    results.setdefault('precision', []).append(precision)
                
                if 'recall' in metrics:
                    recall = self._calculate_recall(relevant_items, recommended_items)
                    results.setdefault('recall', []).append(recall)
                
                if 'ndcg' in metrics:
                    ndcg = self._calculate_ndcg(relevant_items, recommended_items)
                    results.setdefault('ndcg', []).append(ndcg)
            
            # Calculate average metrics
            for metric in results:
                results[metric] = float(np.mean(results[metric]))
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            logger.debug(traceback.format_exc())
            return {metric: 0.0 for metric in metrics}
    
    def _calculate_precision(self, relevant_items: List[str], recommended_items: List[str]) -> float:
        """Calculate precision@K"""
        if not recommended_items:
            return 0.0
        
        hits = len(set(relevant_items) & set(recommended_items))
        return hits / len(recommended_items)
    
    def _calculate_recall(self, relevant_items: List[str], recommended_items: List[str]) -> float:
        """Calculate recall@K"""
        if not relevant_items:
            return 0.0
        
        hits = len(set(relevant_items) & set(recommended_items))
        return hits / len(relevant_items)
    
    def _calculate_ndcg(self, relevant_items: List[str], recommended_items: List[str]) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG)"""
        if not relevant_items or not recommended_items:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended_items):
            if item in relevant_items:
                # Formula: rel_i / log_2(i+2)
                dcg += 1.0 / np.log2(i + 2)
        
        # Calculate IDCG (Ideal DCG)
        idcg = 0.0
        for i in range(min(len(relevant_items), len(recommended_items))):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
            
        return dcg / idcg
    
    def save_model(self, filepath: str) -> bool:
        """
        Save model to file
        
        Args:
            filepath: Path untuk menyimpan model
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'user_id_map': self.user_id_map,
                'item_id_map': self.item_id_map,
                'embedding_size': self.embedding_size,
                'layers': self.layers,
                'config': {
                    'learning_rate': self.learning_rate,
                    'batch_size': self.batch_size,
                    'num_epochs': self.num_epochs
                },
                'metrics': {
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                },
                'saved_at': datetime.now().isoformat()
            }
            
            torch.save(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load model from file
        
        Args:
            filepath: Path ke file model
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Model file not found: {filepath}")
                return False
                
            logger.info(f"Loading model from {filepath}")
            
            # Load model data
            model_data = torch.load(filepath, map_location=self.device)
            
            # Extract mapping of IDs
            self.user_id_map = model_data['user_id_map']
            self.item_id_map = model_data['item_id_map']
            self.embedding_size = model_data['embedding_size']
            self.layers = model_data['layers']
            
            # Update counts
            self.num_users = len(self.user_id_map)
            self.num_items = len(self.item_id_map)
            
            # Extract additional configurations if available
            if 'config' in model_data:
                config = model_data['config']
                self.learning_rate = config.get('learning_rate', self.learning_rate)
                self.batch_size = config.get('batch_size', self.batch_size)
                self.num_epochs = config.get('num_epochs', self.num_epochs)
            
            # Extract metrics if available
            if 'metrics' in model_data:
                metrics = model_data['metrics']
                self.train_losses = metrics.get('train_losses', [])
                self.val_losses = metrics.get('val_losses', [])
            
            # Recreate model
            self.model = NCF(
                self.num_users, 
                self.num_items, 
                embedding_size=self.embedding_size,
                layers=self.layers
            )
            
            # Load state dict
            self.model.load_state_dict(model_data['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Reinitialize optimizer
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=1e-5
            )
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def batch_recommendations(self, user_ids: List[str], n: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate recommendations for multiple users in batch
        
        Args:
            user_ids: List of user IDs
            n: Number of recommendations per user
            
        Returns:
            dict: Mapping of user ID to list of recommendations
        """
        try:
            results = {}
            
            # Filter valid users
            valid_users = [uid for uid in user_ids if uid in self.user_id_map]
            
            if not valid_users:
                logger.warning("No valid users found in the provided list")
                return {}
                
            logger.info(f"Generating recommendations for {len(valid_users)} users")
            
            # Process each user
            for user_id in valid_users:
                recommendations = self.recommend_projects(user_id, n=n)
                results[user_id] = recommendations
                
            return results
            
        except Exception as e:
            logger.error(f"Error generating batch recommendations: {e}")
            logger.debug(traceback.format_exc())
            return {}

if __name__ == "__main__":
    try:
        # Load data
        from src.models.matrix_builder import MatrixBuilder
        from src.processors.data_processor import DataProcessor
        
        logger.info("Loading data for NCF example")
        
        # Load data
        processor = DataProcessor()
        projects_df, interactions_df, _ = processor.load_latest_processed_data()
        
        if projects_df is None or interactions_df is None:
            logger.error("Required data not available")
            sys.exit(1)
            
        logger.info(f"Loaded {len(projects_df)} projects and {len(interactions_df)} interactions")
        
        # Build matrix
        matrix_builder = MatrixBuilder()
        user_item_df, _, _ = matrix_builder.build_user_item_matrix(interactions_df)
        
        if user_item_df is None:
            logger.error("Failed to build user-item matrix")
            sys.exit(1)
            
        logger.info(f"Built user-item matrix with shape {user_item_df.shape}")
        
        # Inisialisasi dan latih NCF
        logger.info("Initializing NCF model")
        ncf = NCFRecommender(user_item_df, projects_df)
        
        # Train model (with fewer epochs for demonstration)
        logger.info("Training NCF model with 5 epochs")
        metrics = ncf.train(val_ratio=0.1)
        
        # Check if we have validation loss
        if metrics["val_loss"]:
            final_val_loss = metrics["val_loss"][-1]
            logger.info(f"Final validation loss: {final_val_loss:.4f}")
        
        # Generate recommendations
        if len(user_item_df) > 0:
            user_id = user_item_df.index[0]
            logger.info(f"Generating recommendations for sample user {user_id}")
            recommendations = ncf.recommend_projects(user_id, n=5)
            
            print(f"\nTop 5 recommendations for {user_id}:")
            for i, rec in enumerate(recommendations, 1):
                name = rec.get('name', rec.get('id', 'Unknown'))
                score = rec.get('recommendation_score', 0)
                print(f"{i}. {name} - Score: {score:.4f}")
                
            # Save model
            model_dir = os.path.join(MODELS_PATH, "ncf")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"ncf_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
            
            if ncf.save_model(model_path):
                print(f"\nModel saved to {model_path}")
                
                # Test loading
                test_ncf = NCFRecommender(user_item_df, projects_df)
                if test_ncf.load_model(model_path):
                    print("Model loading test successful")
                else:
                    print("Model loading test failed")
        else:
            print("No users found in data")
            
    except Exception as e:
        logger.error(f"Error in NCF example: {e}")
        logger.debug(traceback.format_exc())
        print(f"Error: {e}")