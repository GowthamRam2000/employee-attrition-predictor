import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import os

# Try to import SMOTE, but make it optional
try:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: SMOTE not available. Will use class weights instead.")


class AttritionPredictor:
    def __init__(self, input_dim=None):
        self.model = None
        self.input_dim = input_dim
        self.history = None

    def build_model(self, input_dim):
        """Build a deep neural network for attrition prediction"""
        self.input_dim = input_dim

        model = models.Sequential([
            # Input layer
            layers.Input(shape=(input_dim,)),

            # First hidden layer with dropout and batch normalization
            layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Second hidden layer
            layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Third hidden layer
            layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            # Fourth hidden layer
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),

            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        self.model = model
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, use_smote=True):
        """Train the model with optional SMOTE for handling imbalanced data"""

        if self.model is None:
            self.build_model(X_train.shape[1])

        # Apply SMOTE if requested and available
        if use_smote and SMOTE_AVAILABLE:
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        elif use_smote and not SMOTE_AVAILABLE:
            print("SMOTE requested but not available. Using class weights instead.")
            X_train_balanced, y_train_balanced = X_train, y_train
            # Calculate class weights for imbalanced data
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, class_weights))
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
            class_weight_dict = None

        # If no validation set provided, create one
        if X_val is None or y_val is None:
            X_train_balanced, X_val, y_train_balanced, y_val = train_test_split(
                X_train_balanced, y_train_balanced, test_size=0.2, random_state=42, stratify=y_train_balanced
            )

        # Define callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )

        # Train the model
        if use_smote and not SMOTE_AVAILABLE:
            # Use class weights if SMOTE not available
            self.history = self.model.fit(
                X_train_balanced, y_train_balanced,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                class_weight=class_weight_dict,
                verbose=0
            )
        else:
            self.history = self.model.fit(
                X_train_balanced, y_train_balanced,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )

        return self.history

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        predictions = self.model.predict(X, verbose=0)
        return predictions

    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.predict(X)

    def predict_classes(self, X, threshold=0.5):
        """Get binary predictions"""
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int).flatten()

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        # Get predictions
        y_pred_proba = self.predict_proba(X_test)
        y_pred = self.predict_classes(X_test)

        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        # Get classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Get ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        return {
            'accuracy': accuracy,
            'auc': auc_score,
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_curve': (fpr, tpr, thresholds)
        }

    def get_feature_importance(self, feature_names, X_sample, num_samples=100):
        """Get feature importance using permutation importance"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Use a sample for faster computation
        if len(X_sample) > num_samples:
            indices = np.random.choice(len(X_sample), num_samples, replace=False)
            X_sample = X_sample[indices]

        base_predictions = self.predict_proba(X_sample)
        importances = []

        for i in range(X_sample.shape[1]):
            X_permuted = X_sample.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_predictions = self.predict_proba(X_permuted)
            importance = np.mean(np.abs(base_predictions - permuted_predictions))
            importances.append(importance)

        # Normalize importances
        importances = np.array(importances)
        if importances.sum() > 0:
            importances = importances / importances.sum()

        return dict(zip(feature_names, importances))

    def save_model(self, path='models/'):
        """Save the trained model"""
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, 'attrition_model.h5'))

        # Save model metadata
        metadata = {
            'input_dim': self.input_dim
        }
        joblib.dump(metadata, os.path.join(path, 'model_metadata.pkl'))

    def load_model(self, path='models/'):
        """Load a saved model"""
        self.model = keras.models.load_model(os.path.join(path, 'attrition_model.h5'))
        metadata = joblib.load(os.path.join(path, 'model_metadata.pkl'))
        self.input_dim = metadata['input_dim']