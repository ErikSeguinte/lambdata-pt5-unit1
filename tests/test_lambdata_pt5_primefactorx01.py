import unittest
import pandas as pd
from lambdata_pt5_primefactorx01.lambdata_pt5_primefactorx01 import Helper

from lambdata_pt5_primefactorx01 import __version__


def test_version():
    assert __version__ == "0.1.0"


class TestLambdata(unittest.TestCase):
    def test_state_names(self):
        states = pd.Series(["CA", "TX", "ND"])
        state_names = Helper.convert_states(states)

        correct = ["California", "Texas", "North Dakota"]

        for test, correct in zip(state_names, correct):
            self.assertEqual(test, correct)


if __name__ == "__main__":
    unittest.main()
