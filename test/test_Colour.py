import unittest

from src.Colour import Colour


class TestColour(unittest.TestCase):
    """Both call forms must work: agents were written for engine versions
    before and after commit 487517f (25 Nov 2025)."""

    def test_members_unchanged(self):
        self.assertEqual(list(Colour), [Colour.RED, Colour.BLUE])

    def test_get_char_class_form(self):
        self.assertEqual(Colour.get_char(Colour.RED), "R")
        self.assertEqual(Colour.get_char(Colour.BLUE), "B")
        self.assertEqual(Colour.get_char(None), "0")

    def test_get_char_instance_form(self):
        self.assertEqual(Colour.RED.get_char(), "R")
        self.assertEqual(Colour.BLUE.get_char(), "B")

    def test_to_protocol_char(self):
        self.assertEqual(Colour.RED.to_protocol_char(), "R")
        self.assertEqual(Colour.to_protocol_char(Colour.BLUE), "B")
        self.assertEqual(Colour.to_protocol_char(None), "0")

    def test_display_char_is_coloured(self):
        self.assertIn("R", Colour.display_char(Colour.RED))
        self.assertTrue(Colour.display_char(Colour.RED).startswith("\033["))

    def test_opposite_both_forms(self):
        self.assertEqual(Colour.opposite(Colour.RED), Colour.BLUE)
        self.assertEqual(Colour.BLUE.opposite(), Colour.RED)
        self.assertIsNone(Colour.opposite(None))

    def test_from_char(self):
        self.assertEqual(Colour.from_char("R"), Colour.RED)
        self.assertEqual(Colour.from_char("b"), Colour.BLUE)
        self.assertIsNone(Colour.from_char("0"))


if __name__ == "__main__":
    unittest.main()
