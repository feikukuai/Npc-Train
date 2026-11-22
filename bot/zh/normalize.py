"""Chinese input handling: normalisation and language detection."""
import re
import unicodedata

HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
CJK_OR_OTHER = re.compile(
    r"[\u3000-\u303f\uff00-\uffef\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\U0001F000-\U0001FAFF\u2600-\u27bf]")


def handles(text):
    """True when a turn should go to the Chinese engine."""
    if not isinstance(text, str) or not text.strip():
        return False
    if HAN.search(text):
        return True
    # Chinese punctuation, kana, hangul, Cyrillic or emoji all signal a non-English turn.
    return bool(CJK_OR_OTHER.search(text))


def normalize(text):
    """NFKC folds full-width digits and punctuation; whitespace is collapsed."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact(text):
    return re.sub(r"\s+", "", normalize(text))


def is_question(text):
    return bool(re.search(r"[?？]|什么|怎么|怎样|为什么|为啥|哪|谁|多少|多久|多大|多远|"
                          r"几点|几号|几天|几次|吗|呢|如何|能不能|可不可以|是否|"
                          r"讲讲|说说|介绍一下|解释一下|写一首|写首|作诗", text))
