"""
Detection metrics for evaluating temporal object detection.

Includes classification metrics (precision, recall, F1) and
localization metrics (keypoint distance error, detection rate).
"""

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Dict, Tuple


class DetectionMetrics:
    """
    Compute detection metrics for evaluation.

    Args:
        distance_threshold: Maximum distance for a detection to be considered correct (pixels)
    """

    def __init__(self, distance_threshold: float = 15.0):
        self.distance_threshold = distance_threshold

    def compute_metrics(
        self,
        predictions: List[Dict],
        targets: List[Dict],
    ) -> Dict[str, float]:
        """
        Compute all metrics from batch predictions and targets.

        Args:
            predictions: List of prediction dicts (from model.decode_predictions)
                Each dict contains:
                - 'class_probs': (num_classes,) probabilities
                - 'detections': List of {'class': int, 'center': (x, y), 'score': float}
            targets: List of target dicts
                Each dict contains:
                - 'class_labels': (num_classes,) binary labels
                - 'keypoints': List of (x, y) coordinates
                - 'labels': List of class IDs

        Returns:
            Dictionary of metrics:
            - cls_precision, cls_recall, cls_f1: Classification metrics
            - det_precision, det_recall, det_f1: Detection metrics (with localization)
            - keypoint_error: Mean distance error for correct detections (pixels)
            - detection_rate: Fraction of ground truth objects detected
        """
        # Collect batch data
        all_class_probs = []
        all_class_labels = []
        all_predicted_keypoints = []
        all_target_keypoints = []
        all_predicted_classes = []
        all_target_classes = []

        for pred, tgt in zip(predictions, targets):
            # Classification
            all_class_probs.append(pred['class_probs'].cpu().numpy())
            all_class_labels.append(tgt['class_labels'].cpu().numpy())

            # Detections
            for det in pred['detections']:
                all_predicted_keypoints.append(det['center'])
                all_predicted_classes.append(det['class'])

            for kp, cls in zip(tgt['keypoints'], tgt['labels']):
                all_target_keypoints.append(kp)
                all_target_classes.append(cls)

        # Classification metrics
        if all_class_probs:
            cls_metrics = self._classification_metrics(
                np.array(all_class_probs),
                np.array(all_class_labels)
            )
        else:
            cls_metrics = {'cls_precision': 0.0, 'cls_recall': 0.0, 'cls_f1': 0.0}

        # Detection metrics (with localization)
        if all_predicted_keypoints and all_target_keypoints:
            det_metrics = self._detection_metrics(
                all_predicted_keypoints,
                all_predicted_classes,
                all_target_keypoints,
                all_target_classes,
            )
        else:
            det_metrics = {
                'det_precision': 0.0,
                'det_recall': 0.0,
                'det_f1': 0.0,
                'keypoint_error': 0.0,
                'detection_rate': 0.0,
            }

        return {**cls_metrics, **det_metrics}

    def _classification_metrics(
        self,
        class_probs: np.ndarray,
        class_labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute classification metrics (multi-label).

        Args:
            class_probs: (N, num_classes) probabilities
            class_labels: (N, num_classes) binary labels

        Returns:
            Dict with precision, recall, F1
        """
        # Threshold at 0.5
        preds = (class_probs > 0.5).astype(int)

        # Compute per-class metrics with macro averaging
        precision = precision_score(class_labels, preds, average='macro', zero_division=0)
        recall = recall_score(class_labels, preds, average='macro', zero_division=0)
        f1 = f1_score(class_labels, preds, average='macro', zero_division=0)

        return {
            'cls_precision': float(precision),
            'cls_recall': float(recall),
            'cls_f1': float(f1),
        }

    def _detection_metrics(
        self,
        predicted_keypoints: List[Tuple[float, float]],
        predicted_classes: List[int],
        target_keypoints: List[Tuple[float, float]],
        target_classes: List[int],
    ) -> Dict[str, float]:
        """
        Compute detection metrics (precision, recall, F1, keypoint error).

        A detection is correct if:
        1. Class matches
        2. Distance to nearest ground truth < threshold

        Args:
            predicted_keypoints: List of (x, y) predicted centers
            predicted_classes: List of predicted class IDs
            target_keypoints: List of (x, y) ground truth centers
            target_classes: List of ground truth class IDs

        Returns:
            Dict with detection metrics
        """
        if not predicted_keypoints or not target_keypoints:
            return {
                'det_precision': 0.0,
                'det_recall': 0.0,
                'det_f1': 0.0,
                'keypoint_error': 0.0,
                'detection_rate': 0.0,
            }

        # Match predictions to targets
        matched_targets = set()
        correct_detections = []
        keypoint_errors = []

        for pred_kp, pred_cls in zip(predicted_keypoints, predicted_classes):
            # Find nearest target of same class
            best_match_idx = None
            best_match_dist = float('inf')

            for i, (tgt_kp, tgt_cls) in enumerate(zip(target_keypoints, target_classes)):
                if tgt_cls != pred_cls or i in matched_targets:
                    continue

                dist = self._euclidean_distance(pred_kp, tgt_kp)
                if dist < best_match_dist:
                    best_match_dist = dist
                    best_match_idx = i

            # Check if match is within threshold
            if best_match_idx is not None and best_match_dist < self.distance_threshold:
                correct_detections.append(True)
                matched_targets.add(best_match_idx)
                keypoint_errors.append(best_match_dist)
            else:
                correct_detections.append(False)

        # Compute metrics
        tp = sum(correct_detections)
        fp = len(correct_detections) - tp
        fn = len(target_keypoints) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        detection_rate = len(matched_targets) / len(target_keypoints) if target_keypoints else 0.0
        avg_error = np.mean(keypoint_errors) if keypoint_errors else 0.0

        return {
            'det_precision': float(precision),
            'det_recall': float(recall),
            'det_f1': float(f1),
            'keypoint_error': float(avg_error),
            'detection_rate': float(detection_rate),
        }

    @staticmethod
    def _euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Compute Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def test_detection_metrics():
    """Test the DetectionMetrics."""
    print("Testing DetectionMetrics...")

    metrics = DetectionMetrics(distance_threshold=15.0)

    # Create dummy predictions
    predictions = [
        {
            'class_probs': torch.tensor([0.8, 0.3]),  # mover: yes, dipole: no
            'detections': [
                {'class': 0, 'center': (100, 100), 'score': 0.8},
            ]
        },
        {
            'class_probs': torch.tensor([0.2, 0.9]),  # mover: no, dipole: yes
            'detections': [
                {'class': 1, 'center': (200, 200), 'score': 0.9},
            ]
        },
    ]

    # Create dummy targets
    targets = [
        {
            'class_labels': torch.tensor([1.0, 0.0]),  # has mover
            'keypoints': [(105, 102)],  # Within threshold of prediction
            'labels': [0],
        },
        {
            'class_labels': torch.tensor([0.0, 1.0]),  # has dipole
            'keypoints': [(198, 201)],  # Within threshold
            'labels': [1],
        },
    ]

    # Compute metrics
    result = metrics.compute_metrics(predictions, targets)

    print("\nMetrics:")
    for key, value in result.items():
        print(f"  {key}: {value:.4f}")

    # Test expected values
    assert result['cls_f1'] > 0.5, "Classification F1 should be decent"
    assert result['det_f1'] > 0.5, "Detection F1 should be decent"
    assert result['keypoint_error'] < 10, "Keypoint error should be small"
    assert result['detection_rate'] == 1.0, "Should detect all objects"

    print("\nTest passed!")


if __name__ == "__main__":
    test_detection_metrics()
