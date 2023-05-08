"""
Cleaners are transformations that run over the input text at both training and eval time.
Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).

Code from here:
https://github.com/keithito/tacotron/blob/master/text/cleaners.py
https://github.com/keithito/tacotron/blob/master/text/numbers.py
"""

from __future__ import annotations

import re

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
# WARNING: Every change here means an incompatible change,
# so better leave it always as it is!
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misses"),
        ("ms", "miss"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    """
    :param str text:
    :rtype: str
    """
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    """
    :param str text:
    :rtype: str
    """
    return text.lower()


def lowercase_keep_special(text):
    """
    :param str text:
    :rtype: str
    """
    # Anything which is not [..] or <..>.
    return re.sub("(\\s|^)(?!(\\[\\S*])|(<\\S*>))\\S+(?=\\s|$)", lambda m: m.group(0).lower(), text)


def collapse_whitespace(text):
    """
    :param str text:
    :rtype: str
    """
    text = re.sub(_whitespace_re, " ", text)
    text = text.strip()
    return text


def convert_to_ascii(text):
    """
    :param str text:
    :rtype: str
    """
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from unidecode import unidecode

    return unidecode(text)


def basic_cleaners(text):
    """
    Basic pipeline that lowercases and collapses whitespace without transliteration.

    :param str text:
    :rtype: str
    """
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """
    Pipeline for non-English text that transliterates to ASCII.

    :param str text:
    :rtype: str
    """
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    """
    Pipeline for English text, including number and abbreviation expansion.
    :param str text:
    :rtype: str
    """
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = normalize_numbers(text, with_spacing=True)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners_keep_special(text):
    """
    Pipeline for English text, including number and abbreviation expansion.
    :param str text:
    :rtype: str
    """
    text = convert_to_ascii(text)
    text = lowercase_keep_special(text)
    text = normalize_numbers(text, with_spacing=True)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def get_remove_chars(chars):
    """
    :param str|list[str] chars:
    :rtype: (str)->str
    """

    def remove_chars(text):
        """
        :param str text:
        :rtype: str
        """
        for c in chars:
            text = text.replace(c, " ")
        text = collapse_whitespace(text)
        return text

    return remove_chars


def get_replace(old, new):
    """
    :param str old:
    :param str new:
    :rtype: (str)->str
    """

    def replace(text):
        """
        :param str text:
        :rtype: str
        """
        text = text.replace(old, new)
        return text

    return replace


_inflect = None


def _get_inflect():
    global _inflect
    if _inflect:
        return _inflect
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    import inflect

    _inflect = inflect.engine()
    return _inflect


_comma_number_re = re.compile(r"([0-9][0-9,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9.,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")


def _remove_commas(m):
    """
    :param typing.Match m:
    :rtype: str
    """
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    """
    :param typing.Match m:
    :rtype: str
    """
    return m.group(1).replace(".", " point ")


def _expand_dollars(m):
    """
    :param typing.Match m:
    :rtype: str
    """
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_ordinal(m):
    """
    :param typing.Match m:
    :rtype: str
    """
    return _get_inflect().number_to_words(m.group(0))


def _expand_number(m):
    """
    :param typing.Match m:
    :rtype: str
    """
    num_s = m.group(0)
    num_s = num_s.strip()
    if "." in num_s:
        return _get_inflect().number_to_words(num_s, andword="")
    num = int(num_s)
    if num_s.startswith("0") or num in {747}:
        digits = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
        }
        return " ".join([digits.get(c, c) for c in num_s])
    if 1000 < num < 3000:
        if num == 2000:
            return "two thousand"
        elif 2000 < num < 2010:
            return "two thousand " + _get_inflect().number_to_words(num % 100)
        elif num % 100 == 0:
            return _get_inflect().number_to_words(num // 100) + " hundred"
        else:
            return _get_inflect().number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    else:
        return _get_inflect().number_to_words(num, andword="")


def _expand_number_with_spacing(m):
    """
    :param typing.Match m:
    :rtype: str
    """
    return " %s " % _expand_number(m)


def normalize_numbers(text, with_spacing=False):
    """
    :param str text:
    :param bool with_spacing:
    :rtype: str
    """
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number_with_spacing if with_spacing else _expand_number, text)
    return text
