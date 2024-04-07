import logging

import numpy as np
from PIL import Image
import numpy as np


logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

def preprocess_imagenet(image, channels=3, height=640, width=640):
    try:
        # Tu código existente aquí.
        resized_image = image.resize((width, height), Image.LANCZOS)
        img_data = np.asarray(resized_image).astype(np.float32)

        if len(img_data.shape) == 2:
            img_data = np.stack([img_data] * 3)
            logger.debug("Received grayscale image. Reshaped to {:}".format(img_data.shape))
        else:
            img_data = img_data.transpose([2, 0, 1])

        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])

        if img_data.shape[0] != channels:
            raise AssertionError("Número de canales no coincide con el esperado.")

        for i in range(img_data.shape[0]):
            img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]

        return img_data
    except AssertionError as e:
        # Manejar la excepción de aserción aquí.
        logger.error(f"Error procesando la imagen: {e}")
        return None