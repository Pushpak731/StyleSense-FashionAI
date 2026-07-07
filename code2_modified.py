# Recommended: Python 3.9–3.11
# pip install mediapipe opencv-python numpy scikit-learn pandas xgboost matplotlib seaborn

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Optional, Dict
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.impute import SimpleImputer
import math
import warnings
import sys
import json
import os

warnings.filterwarnings('ignore')


class PersonalAnalyzer:
    """Analyzes face skin tone, body shape, hair, and eye color from images"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,  # Needed for Iris landmarks
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def correct_white_balance(self, image: np.ndarray) -> np.ndarray:
        """Adjusts the image to correct for yellow/warm indoor lighting."""
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result

    def get_dominant_color(self, pixels: np.ndarray, k: int = 3) -> np.ndarray:
        if len(pixels) == 0:
            return np.array([0, 0, 0], dtype=int)

        clt = KMeans(n_clusters=k, n_init=10, random_state=42)
        clt.fit(pixels)
        labels = clt.labels_
        counts = np.bincount(labels)
        dominant_idx = np.argmax(counts)
        return clt.cluster_centers_[dominant_idx].astype(int)

    @staticmethod
    def rgb_to_hex(color: np.ndarray) -> str:
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

    # --- CATEGORIZATION HELPERS ---
    @staticmethod
    def get_skin_tone_category(rgb: np.ndarray) -> str:
        if rgb is None: return 'unknown'
        r, g, b = rgb
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        if luminance > 230: return 'very fair'
        elif luminance > 200: return 'fair'
        elif luminance > 170: return 'light'
        elif luminance > 140: return 'medium'
        elif luminance > 100: return 'tan'
        elif luminance > 60: return 'deep'
        else: return 'dark'

    @staticmethod
    def get_undertone(rgb: np.ndarray) -> str:
        if rgb is None: return 'unknown'
        r, g, b = rgb
        if r > g + 20 and r > b + 20: return 'warm'
        elif b > r + 20 and b > g + 20: return 'cool'
        else: return 'neutral'

    @staticmethod
    def get_eye_color_category(rgb: np.ndarray) -> str:
        """Simple heuristic to categorize eye color from RGB."""
        if rgb is None: return 'unknown'
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        
        # Convert to HSV for better color logic
        hsv_pixel = np.uint8([[[b, g, r]]]) # OpenCV uses BGR
        hsv = cv2.cvtColor(hsv_pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv[0], hsv[1], hsv[2] # H:0-179, S:0-255, V:0-255

        if v < 40: return 'dark brown' # Very dark
        if s < 30: return 'gray'       # Low saturation
        
        # Hue based (OpenCV Hue is 0-179)
        # Blue: ~100-130, Green: ~40-80, Brown/Hazel: < 20 or > 160
        if 100 <= h <= 140: return 'blue'
        if 35 <= h <= 90: return 'green'
        if v > 100 and s > 100 and (h < 25 or h > 155): return 'hazel'
        
        return 'brown' # Default

    @staticmethod
    def get_hair_color_category(rgb: np.ndarray) -> str:
        """Simple heuristic to categorize hair color from RGB."""
        if rgb is None: return 'unknown'
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        
        hsv_pixel = np.uint8([[[b, g, r]]]) 
        hsv = cv2.cvtColor(hsv_pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv[0], hsv[1], hsv[2]

        if v < 50: return 'black'
        if v > 180 and s < 50: return 'white/gray'
        if v > 160: return 'blonde'
        
        # Check for redness (low Hue)
        if (h < 15 or h > 165) and s > 60: return 'red'
        
        return 'brown'

    # --- ANALYSIS METHODS ---
    def analyze_skin_tone(self, image: np.ndarray) -> Tuple[Optional[str], Optional[np.ndarray]]:
        image = self.correct_white_balance(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None, None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape

        face_oval_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
        ]
        
        face_points = []
        for idx in face_oval_indices:
            if idx >= len(landmarks): continue
            lm = landmarks[idx]
            face_points.append([int(lm.x * w), int(lm.y * h)])

        if len(face_points) < 10: return None, None

        face_points = np.array(face_points, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [face_points], 255)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)

        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        # Using broad skin range
        skin_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 15, 30], np.uint8), np.array([25, 170, 255], np.uint8)),
            cv2.inRange(hsv, np.array([0, 10, 60], np.uint8), np.array([20, 150, 255], np.uint8))
        )
        final_mask = cv2.bitwise_and(mask, skin_mask)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

        skin_pixels = image_rgb[final_mask > 0]
        if len(skin_pixels) < 50: skin_pixels = image_rgb[mask > 0]
        if len(skin_pixels) < 50: return None, None

        dominant = self.get_dominant_color(skin_pixels, k=3)
        return self.rgb_to_hex(dominant), dominant

    def analyze_hair_and_eyes(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extracts dominant hair and eye colors."""
        image = self.correct_white_balance(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None, None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape

        # Eye landmarks
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        
        left_eye_pixels, right_eye_pixels = [], []
        for idx in left_eye_indices:
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            if 0 <= x < w and 0 <= y < h: left_eye_pixels.append(image_rgb[y, x])
        for idx in right_eye_indices:
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            if 0 <= x < w and 0 <= y < h: right_eye_pixels.append(image_rgb[y, x])
        
        eye_pixels = np.array(left_eye_pixels + right_eye_pixels)
        eye_color = self.get_dominant_color(eye_pixels, k=2) if len(eye_pixels) > 0 else None

        # Hair landmarks (forehead region)
        forehead_indices = [10, 67, 109, 103, 54, 21, 162, 127, 234, 93, 132, 58]
        forehead_pixels = []
        for idx in forehead_indices:
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h - 30)
            if 0 <= x < w and 0 <= y < h: forehead_pixels.append(image_rgb[y, x])
        
        hair_color = self.get_dominant_color(np.array(forehead_pixels), k=2) if len(forehead_pixels) > 0 else None

        return hair_color, eye_color

    def analyze_body_shape(self, image: np.ndarray) -> Tuple[Optional[str], Dict]:
        """Analyzes body shape and proportions."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return None, {}

        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape

        def get_point(idx): 
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

        def dist(p1, p2): 
            return np.linalg.norm(p1 - p2)

        try:
            left_shoulder = get_point(11)
            right_shoulder = get_point(12)
            left_hip = get_point(23)
            right_hip = get_point(24)
            left_knee = get_point(25)
            right_knee = get_point(26)

            shoulder_width = dist(left_shoulder, right_shoulder)
            hip_width = dist(left_hip, right_hip)
            torso_length = dist((left_shoulder + right_shoulder) / 2, (left_hip + right_hip) / 2)
            leg_length = dist((left_hip + right_hip) / 2, (left_knee + right_knee) / 2)

            total_height = torso_length + leg_length
            torso_length_norm = torso_length / total_height if total_height > 0 else 0.5
            shoulder_hip_ratio = shoulder_width / hip_width if hip_width > 0 else 1.0

            # Body shape classification
            if shoulder_hip_ratio > 1.05:
                shape = "inverted triangle"
            elif shoulder_hip_ratio < 0.95:
                shape = "pear"
            elif abs(shoulder_width - hip_width) < 0.05 * shoulder_width:
                shape = "rectangle"
            else:
                shape = "hourglass"

            metrics = {
                'shoulder_width': shoulder_width,
                'hip_width': hip_width,
                'torso_length': torso_length,
                'leg_length': leg_length,
                'torso_length_norm': torso_length_norm,
                'shoulder_hip_ratio': shoulder_hip_ratio
            }

            return shape, metrics

        except Exception as e:
            return None, {}

    def visualize_results(self, face_img, body_img, hex_color, rgb_color, shape_name):
        """Visualizes analysis results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Face with skin tone
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(face_rgb)
        axes[0].set_title(f"Face Analysis\nSkin Tone: {hex_color}", fontsize=10)
        axes[0].axis('off')

        # Body shape
        body_rgb = cv2.cvtColor(body_img, cv2.COLOR_BGR2RGB)
        axes[1].imshow(body_rgb)
        axes[1].set_title(f"Body Shape: {shape_name}", fontsize=10)
        axes[1].axis('off')

        # Color swatch
        color_patch = np.ones((100, 100, 3), dtype=np.uint8)
        color_patch[:, :] = rgb_color
        axes[2].imshow(color_patch)
        axes[2].set_title(f"Dominant Skin Color\n{hex_color}", fontsize=10)
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig('analysis_results.png', dpi=150, bbox_inches='tight')
        print("✅ Visualization saved as 'analysis_results.png'")
        plt.close()


class OutfitRecommender:
    """ML-based outfit recommender system"""
    
    def __init__(self, dataset_path='_recommendations_with_hex.csv'):
        self.dataset_path = dataset_path
        self.model_pipeline = None
        self.label_encoders = {}
        self.feature_columns = []
        self.target_columns = [
            'Recommended Clothing Colors', 'Avoid Clothing Colors', 
            'Recommended Fitting Style', 'Recommended Materials',
            'Recommended Patterns', 'Recommended Jewelry Metal',
            'Recommended Shoes', 'Recommended Clothing Color Wheel Region',
            'Avoid Clothing Color Wheel Region', 'Fabric Nature',
            "Don't Exaggerate", 'Do Exaggerate'
        ]
    
    def load_and_preprocess_data(self):
        print("\n" + "="*60)
        print(" LOADING DATASET")
        print("="*60)
        df = pd.read_csv(self.dataset_path)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def prepare_features_and_target(self, df):
        print("\n" + "="*60)
        print(" PREPARING FEATURES AND TARGETS")
        print("="*60)
        X = df.drop(columns=self.target_columns, errors='ignore')
        Y = df[self.target_columns]
        print(f"Features: {X.shape[1]} columns | Targets: {Y.shape[1]} columns")
        return X, Y
    
    def split_data(self, X, Y):
        X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
        X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.176, random_state=42)
        print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
        return X_train, X_val, X_test, Y_train, Y_val, Y_test
    
    def remove_outliers(self, X_train, Y_train, threshold=3):
        numeric_cols = X_train.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0: return X_train, Y_train
        z_scores = np.abs((X_train[numeric_cols] - X_train[numeric_cols].mean()) / X_train[numeric_cols].std())
        outlier_mask = (z_scores < threshold).all(axis=1)
        X_train = X_train[outlier_mask]
        Y_train = Y_train.loc[X_train.index]
        print(f"Removed {(~outlier_mask).sum()} outliers | Remaining: {X_train.shape[0]}")
        return X_train, Y_train
    
    def build_pipeline(self, X_train):
        print("\n" + "="*60)
        print(" BUILDING MODEL PIPELINE")
        print("="*60)
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
        
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])
        classifier = MultiOutputClassifier(XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, random_state=42, eval_metric='mlogloss'
        ))
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('variance', VarianceThreshold()),
            ('classifier', classifier)
        ])
        return pipeline
    
    def encode_targets(self, Y_train, Y_val, Y_test):
        Y_train_enc, Y_val_enc, Y_test_enc = pd.DataFrame(index=Y_train.index), pd.DataFrame(index=Y_val.index), pd.DataFrame(index=Y_test.index)
        for col in self.target_columns:
            le = LabelEncoder()
            Y_train_enc[col] = le.fit_transform(Y_train[col].astype(str))
            try:
                Y_val_enc[col] = le.transform(Y_val[col].astype(str))
                Y_test_enc[col] = le.transform(Y_test[col].astype(str))
            except ValueError:
                print(f"Warning: New labels in Val/Test for {col}. Refitting combined.")
                le.fit(pd.concat([Y_train[col], Y_val[col], Y_test[col]]).astype(str))
                Y_train_enc[col] = le.transform(Y_train[col].astype(str))
                Y_val_enc[col] = le.transform(Y_val[col].astype(str))
                Y_test_enc[col] = le.transform(Y_test[col].astype(str))
            self.label_encoders[col] = le
        return Y_train_enc, Y_val_enc, Y_test_enc

    def train_model(self, X_train, Y_train_enc, X_val, Y_val_enc):
        print("\n" + "="*60)
        print(" TRAINING MODEL")
        print("="*60)
        self.model_pipeline.fit(X_train, Y_train_enc)
        val_pred = self.model_pipeline.predict(X_val)
        avg_acc = np.mean([accuracy_score(Y_val_enc.iloc[:, i], val_pred[:, i]) for i in range(Y_val_enc.shape[1])])
        print(f"Average Validation Accuracy: {avg_acc:.4f}")
        print("Training complete! ✅")
    
    def evaluate_model(self, X_test, Y_test_enc):
        print("\n" + "="*60)
        print(" MODEL EVALUATION")
        print("="*60)
        y_pred = self.model_pipeline.predict(X_test)
        f1s = []
        for i, col in enumerate(self.target_columns):
            f1 = f1_score(Y_test_enc.iloc[:, i], y_pred[:, i], average='weighted', zero_division=0)
            f1s.append(f1)
        print(f"Average F1-score : {np.mean(f1s):.4f}")
    
    def train_complete_pipeline(self):
        df = self.load_and_preprocess_data()
        X, Y = self.prepare_features_and_target(df)
        cat_cols = X.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            dummy_X = X.iloc[[0]].copy()
            dummy_X[cat_cols] = 'unknown'
            X = pd.concat([X, dummy_X], ignore_index=True)
            Y = pd.concat([Y, Y.iloc[[0]]], ignore_index=True)

        X_train, X_val, X_test, Y_train, Y_val, Y_test = self.split_data(X, Y)
        X_train, Y_train = self.remove_outliers(X_train, Y_train)
        Y_train_enc, Y_val_enc, Y_test_enc = self.encode_targets(Y_train, Y_val, Y_test)
        
        self.feature_columns = X_train.columns.tolist()
        self.model_pipeline = self.build_pipeline(X_train)
        self.train_model(X_train, Y_train_enc, X_val, Y_val_enc)
        self.evaluate_model(X_test, Y_test_enc)
    
    def recommend_outfit(self, user_features: dict):
        print("\n" + "="*60)
        print(" OUTFIT RECOMMENDATION")
        print("="*60)
        if self.model_pipeline is None: raise ValueError("Model not trained.")
        
        user_df = pd.DataFrame([user_features])
        for col in self.feature_columns:
            if col not in user_df.columns: user_df[col] = np.nan
        user_df = user_df[self.feature_columns]
        
        prediction = self.model_pipeline.predict(user_df)[0]
        recommendations = {}
        for i, col in enumerate(self.target_columns):
            rec = self.label_encoders[col].inverse_transform([prediction[i]])[0]
            recommendations[col] = rec
        
        print("\nUser Profile:")
        for k, v in user_features.items(): print(f"  {k}: {v}")
        print("\nRecommendations:")
        for col, rec in recommendations.items(): print(f"  {col}: {rec}")
        return recommendations


# ADDED: Function to analyze images and return results as dictionary
def analyze_user_images(face_image_path: str, body_image_path: str) -> dict:
    """
    Analyzes face and body images and returns a dictionary with all results.
    This function is called by the chatbot UI.
    """
    analyzer = PersonalAnalyzer()
    
    face_img = cv2.imread(face_image_path)
    body_img = cv2.imread(body_image_path)
    
    if face_img is None or body_img is None:
        return None
    
    # Analyze Skin Tone
    hex_color, rgb_color = analyzer.analyze_skin_tone(face_img)
    if hex_color is None:
        return None

    # Analyze Body Shape
    shape_name, metrics = analyzer.analyze_body_shape(body_img)
    if shape_name is None:
        return None

    # Analyze Hair and Eye Color
    hair_rgb, eye_rgb = analyzer.analyze_hair_and_eyes(face_img)
    
    # Convert results to categories
    skin_tone_cat = PersonalAnalyzer.get_skin_tone_category(rgb_color)
    undertone = PersonalAnalyzer.get_undertone(rgb_color)
    eye_cat = PersonalAnalyzer.get_eye_color_category(eye_rgb)
    hair_cat = PersonalAnalyzer.get_hair_color_category(hair_rgb)
    
    # Visualize results
    analyzer.visualize_results(face_img, body_img, hex_color, rgb_color, shape_name)
    
    # Train recommender
    recommender = OutfitRecommender(dataset_path=r"_recommendations_with_hex.csv")
    try:
        recommender.train_complete_pipeline()
    except Exception as e:
        print(f"Training error: {e}")
        return None
    
    # Get recommendations
    user_profile = {
        'HEX code': hex_color,
        'Skin Tone': skin_tone_cat,
        'Under Tone': undertone,
        'Torso length': metrics.get('torso_length_norm', 0.5),
        'Body Proportion': shape_name.lower(),
        'Hair Color': hair_cat,  
        'Eye Color': eye_cat,  
    }
    
    recommendations = recommender.recommend_outfit(user_profile)
    
    # Return complete analysis result
    return {
        'user_profile': user_profile,
        'recommendations': recommendations,
        'hex_color': hex_color,
        'skin_tone': skin_tone_cat,
        'undertone': undertone,
        'hair_color': hair_cat,
        'eye_color': eye_cat,
        'body_shape': shape_name
    }


def main():
    print("\n" + "█"*60)
    print("█" + " "*18 + "PERSONAL STYLE ANALYZER" + " "*19 + "█")
    print("█"*60 + "\n")
    
    print("STEP 1: Analyzing your images...")
    print("-" * 60)
    
    analyzer = PersonalAnalyzer()
    
    face_image_path = r"skin.jpg"
    body_image_path = r"body-1.jpg"
    
    face_img = cv2.imread(face_image_path)
    body_img = cv2.imread(body_image_path)
    
    if face_img is None:
        print(f"❌ ERROR: Could not read '{face_image_path}'.")
        sys.exit(1)
    if body_img is None:
        print(f"❌ ERROR: Could not read '{body_image_path}'.")
        sys.exit(1)
    
    # 1. Analyze Skin Tone
    hex_color, rgb_color = analyzer.analyze_skin_tone(face_img)
    if hex_color is None:
        print("\n❌ CRITICAL FAILURE: Could not detect skin tone/face.")
        sys.exit(1)

    # 2. Analyze Body Shape
    shape_name, metrics = analyzer.analyze_body_shape(body_img)
    if shape_name is None:
        print("\n❌ CRITICAL FAILURE: Could not detect body shape.")
        sys.exit(1)

    # 3. Analyze Hair and Eye Color (NEW)
    hair_rgb, eye_rgb = analyzer.analyze_hair_and_eyes(face_img)
    
    # Convert results to categories
    skin_tone_cat = PersonalAnalyzer.get_skin_tone_category(rgb_color)
    undertone = PersonalAnalyzer.get_undertone(rgb_color)
    eye_cat = PersonalAnalyzer.get_eye_color_category(eye_rgb)
    hair_cat = PersonalAnalyzer.get_hair_color_category(hair_rgb)
    
    print("\n" + "="*60)
    print(" ANALYSIS RESULTS")
    print("="*60)
    print(f"Skin Tone: {hex_color} ({skin_tone_cat}) | Undertone: {undertone}")
    print(f"Hair Color: {hair_cat}")
    print(f"Eye Color : {eye_cat}")
    print(f"Body Shape: {shape_name}")
    
    analyzer.visualize_results(face_img, body_img, hex_color, rgb_color, shape_name)
    
    print("\n\nSTEP 2: Training outfit recommendation model...")
    print("-" * 60)
    
    recommender = OutfitRecommender(dataset_path=r"_recommendations_with_hex.csv")
    try:
        recommender.train_complete_pipeline()
    except Exception as e:
        print(f"\n❌ TRAINING ERROR: {e}")
        sys.exit(1)
    
    print("\n\nSTEP 3: Generating outfit recommendation...")
    print("-" * 60)
    
    user_profile = {
        'HEX code': hex_color,
        'Skin Tone': skin_tone_cat,
        'Under Tone': undertone,
        'Torso length': metrics.get('torso_length_norm', 0.5),
        'Body Proportion': shape_name.lower(),
        'Hair Color': hair_cat,  
        'Eye Color': eye_cat,  
    }
    
    recommendations = recommender.recommend_outfit(user_profile)
    
    print("\n" + "█"*60)
    print("█" + " "*19 + "ANALYSIS COMPLETE!" + " "*20 + "█")
    print("█"*60 + "\n")


if __name__ == "__main__":
    main()
