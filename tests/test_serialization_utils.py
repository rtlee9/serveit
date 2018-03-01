"""Test utility methods."""
import unittest
import numpy as np

from serveit.utils import is_serializable, make_serializable


class SeralizationTest(unittest.TestCase):
    """Test serialization."""

    def setUp(self):
        """Unittest setup."""
        self.serializable_data = [
            [1, 2, 3],
            ['a', 'b', 'c'],
            {'a': 1, 'b': 'c'},
            {'a': [1, 2, 3], 'b': 'c'},
        ]

        class DummyClass(object):
            def __repr__(self):
                return 'XpBheIxCcm'

        self.dummy_class = DummyClass()
        self.unserializable_data = [
            np.array([1, 2, 3]),
            np.array(['a', 'b', 'c']),
            self.dummy_class,
        ]

    def test_is_serializable(self):
        """Test is_serializable against example data."""
        for data in self.serializable_data:
            self.assertTrue(is_serializable(data))
        for data in self.unserializable_data:
            self.assertFalse(is_serializable(data))

    def test_make_serializable_data_serializable(self):
        """make_serializable should return the same object if serializable."""
        for data in self.serializable_data:
            self.assertEqual(make_serializable(data), data)

    def test_make_serializable_numpy_data(self):
        """make_serializable should cast numpy array to list."""
        self.assertEqual(make_serializable(np.array([1, 2, 3])), [1, 2, 3])
        self.assertEqual(make_serializable(np.array(['a', 'b', 'c'])), ['a', 'b', 'c'])

    def test_make_serializable_object_data(self):
        """make_serializable should return an objects __repr__ if no `tolist` method."""
        self.assertEqual(make_serializable(self.dummy_class), 'XpBheIxCcm')


if __name__ == '__main__':
    unittest.main()
