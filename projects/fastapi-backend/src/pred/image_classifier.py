from tf_keras.src.utils.image_dataset import load_image

from src.pred.models.tf_pred import *
from typing import Any

def tf_run_classifier(image: str) -> Any:
    img = load_image(image)
    if img is None:
        return None
    pred_results = tf_predict(img)
    return pred_results
