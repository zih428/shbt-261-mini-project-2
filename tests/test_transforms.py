import numpy as np
from PIL import Image

from vocseg.constants import IGNORE_INDEX
from vocseg.data.transforms import ResizeAndPad


def test_resize_and_pad_preserves_target_shape_and_ignore_fill():
    image = Image.fromarray(np.zeros((10, 20, 3), dtype=np.uint8))
    mask = Image.fromarray(np.zeros((10, 20), dtype=np.uint8))
    transform = ResizeAndPad((32, 32))
    transformed_image, transformed_mask, meta = transform(image, mask)
    assert transformed_image.size == (32, 32)
    assert transformed_mask.size == (32, 32)
    mask_array = np.array(transformed_mask)
    assert IGNORE_INDEX in mask_array
    assert meta["padding"][0] >= 0
