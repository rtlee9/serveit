"""Test utility methods."""
import unittest

from serveit.utils import get_bytes_to_image_callback


class CallbackTest(unittest.TestCase):
    """Test utility callbacks."""

    def _test_get_bytes_to_image_callback(self, image_dims):
        """Convert image bytes to an image for ImageNet."""
        with open('tests/SuccessKid.jpg', 'rb') as f:
            image_bytes = f.read()
        bytes_to_image_callback = get_bytes_to_image_callback(image_dims=image_dims)
        image = bytes_to_image_callback(image_bytes)
        self.assertEqual((1, *image_dims, 3), image.shape)

    def test_get_bytes_to_image_callback_224_224(self):
        """Convert image bytes to 224x224 image for ImageNet."""
        self._test_get_bytes_to_image_callback((224, 224))

    def test_get_bytes_to_image_callback_128_128(self):
        """Convert image bytes to 128x128 image for ImageNet."""
        self._test_get_bytes_to_image_callback((128, 128))

if __name__ == '__main__':
    unittest.main()
