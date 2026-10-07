"""Multilingual answers: detection, protected facts in translation, and the translate–answer–translate flow."""
import pytest

from app.i18n.languages import answer_language, detect_language
from app.i18n.translate import TranslationError, _join, _split, check_translation
from app.rag import service


@pytest.mark.parametrize("text, code", [
    ("ബിസിഎ കോഴ്സിന്റെ ഫീസ് എത്രയാണ്?", "ml"),
    ("कॉलेज के प्रिंसिपल कौन हैं?", "hi"),
    ("હોસ્ટેલ સુવિધા છે?", "gu"),
    ("தேர்வு எப்போது?", "ta"),
    ("పరీక్షలు ఎప్పుడు?", "te"),
    ("ಪರೀಕ್ಷೆ ಯಾವಾಗ?", "kn"),
    ("ফি কত?", "bn"),
    ("ਫੀਸ ਕਿੰਨੀ ਹੈ?", "pa"),
    ("ଫି କେତେ?", "or"),
    ("فیس کتنی ہے؟", "ur"),
    ("What is the BCA fee?", "en"),
    ("fees kitni hai BCA ki", "en"),  # romanised Hindi: the answering model reads it as is
    ("BCA ફી કેટલી છે?", "gu"),  # mixed with an English course code
    ("2026", "en"),
])
def test_language_detected_from_script(text, code):
    assert detect_language(text) == code


def test_devanagari_is_marathi_when_marathi_was_chosen():
    assert detect_language("परीक्षा कधी आहे?", preferred="mr") == "mr"
    assert detect_language("परीक्षा कधी आहे?") == "hi"


def test_answer_language_follows_what_was_typed_then_the_selector():
    assert answer_language("en", "ഫീസ് എത്ര?") == "ml"  # typed Malayalam, selector says English
    assert answer_language("ml", "What is the fee?") == "ml"  # quick chips are English: selector wins
    assert answer_language("auto", "What is the fee?") == "en"
    assert answer_language("xx", "What is the fee?") == "en"


def test_translation_must_keep_numbers_emails_and_links():
    src = "The BCA fee is Rs. 62,500. Pay by 25 June 2026. Email admissions@sxca.edu.in or see https://sxca.edu.in/fees/."
    good = "ബിസിഎ ഫീസ് 62,500 രൂപയാണ്. 2026 ജൂൺ 25-നകം അടയ്ക്കുക. admissions@sxca.edu.in https://sxca.edu.in/fees/"
    assert check_translation(src, good) == []
    native_digits = good.replace("62,500", "६२,५००")  # Indian-script digits count as the same number
    assert check_translation(src, native_digits) == []
    assert "62500" in check_translation(src, good.replace("62,500", "26,500"))
    assert "admissions@sxca.edu.in" in check_translation(src, good.replace("admissions@sxca.edu.in", "admission@sxca.edu.in"))


def test_numbers_cut_short_by_the_model_are_put_back():
    from app.i18n.translate import repair_numbers

    src = "The last date to pay the semester fees is 25 June 2026."
    assert repair_numbers(src, "202 ജൂൺ 25 വരെ ഫീസ് അടയ്ക്കണം.") == "2026 ജൂൺ 25 വരെ ഫീസ് അടയ്ക്കണം."
    # A wrong number that isn't a cut-off version stays wrong (and check_translation rejects it)
    assert "2019" in repair_numbers(src, "2019 ജൂൺ 25")


async def test_translator_chain_falls_back_and_checks(monkeypatch):
    from app.i18n import translate as tr

    monkeypatch.setattr(tr, "providers", lambda: ["llm", "indictrans2"])
    tr._cache.clear()
    src = "The BCA fee is Rs. 62,500."

    async def run(p, text, s, t):
        # The LLM "translates" by echoing English; the backup gets it right
        return text if p == "llm" else "ബിസിഎ ഫീസ് 62,500 രൂപ."

    monkeypatch.setattr(tr, "_run", run)
    assert await tr.translate(src, "en", "ml") == "ബിസിഎ ഫീസ് 62,500 രൂപ."

    async def changes_number(p, text, s, t):
        return "ബിസിഎ ഫീസ് 26,500 രൂപ."

    tr._cache.clear()
    monkeypatch.setattr(tr, "_run", changes_number)
    with pytest.raises(TranslationError):
        await tr.translate(src, "en", "ml")

    async def answers_instead(p, text, s, t):
        return "The fee for BCA is Rs. 62,500 per year, payable in two instalments. " * 5

    tr._cache.clear()
    monkeypatch.setattr(tr, "_run", answers_instead)
    with pytest.raises(TranslationError):
        await tr.translate("ബിസിഎ ഫീസ് എത്ര?", "ml", "en")


def test_college_and_bot_names_are_locked_during_translation():
    from app.i18n.translate import lock_names, unlock_names

    text, found = lock_names("Hello! I'm Xavier's Assistant at St. Xavier's College, Ahmedabad.")
    assert "Xavier" not in text and text.count("[[") == 2
    back = unlock_names(text.replace("Hello! I'm", "नमस्ते! मैं"), found)
    assert back == "नमस्ते! मैं Xavier's Assistant at St. Xavier's College, Ahmedabad."
    with pytest.raises(TranslationError):
        unlock_names("नमस्ते! [[1]]", found)  # a name was dropped


def test_welcome_is_hand_written_for_every_language():
    from app.i18n.languages import LANGUAGES
    from app.i18n.messages import WELCOME

    assert set(WELCOME) == set(LANGUAGES) - {"en"}
    for code, text in WELCOME.items():
        assert "St. Xavier's College, Ahmedabad" in text and "Xavier's Assistant" in text
        assert detect_language(text, preferred=code) == code


def test_markdown_lines_and_bullets_survive_translation():
    text = "Fees:\n- **BCA**: Rs. 62,500. Paid yearly.\n\n1. Apply online"
    lines = _split(text)
    assert lines[1][0] == "- " and lines[1][1] == ["BCA: Rs. 62,500.", "Paid yearly."]
    upper = [(p, [s.upper() for s in sents]) for p, sents in lines]
    assert _join(upper) == "FEES:\n- BCA: RS. 62,500. PAID YEARLY.\n\n1. APPLY ONLINE"
    # Abbreviations and initials don't end a sentence
    assert _split("Contact Dr. Pravida Raja A. C. at St. Xavier's. Fee is Rs. 500.")[0][1] == [
        "Contact Dr. Pravida Raja A. C. at St. Xavier's.", "Fee is Rs. 500."]


# ---------------------------------------------------------------- full flow (fake translator and pipeline)
def _fake_pipeline(seen):
    async def fake(question, history=None):
        seen["question"], seen["history"] = question, history
        yield service.ChatEvent(type="sources", sources=[{"title": "Fees", "url": "https://sxca.edu.in/fees/"}])
        yield service.ChatEvent(type="token", text="The BCA fee is ")
        yield service.ChatEvent(type="token", text="Rs. 62,500.")
        yield service.ChatEvent(type="done", answered=True)
    return fake


async def _collect(gen):
    return [ev async for ev in gen]


async def test_malayalam_question_is_answered_in_malayalam(monkeypatch):
    seen, calls = {}, []

    async def fake_translate(text, src, tgt):
        calls.append((src, tgt))
        return "What is the BCA fee?" if tgt == "en" else "ബിസിഎ ഫീസ് 62,500 രൂപയാണ്."

    monkeypatch.setattr(service, "_answer_english", _fake_pipeline(seen))
    monkeypatch.setattr(service, "translate", fake_translate)
    events = await _collect(service.answer_stream("ബിസിഎ ഫീസ് എത്ര?", [], "en"))
    types = [e.type for e in events]
    assert events[0].type == "language" and events[0].text == "ml"
    assert seen["question"] == "What is the BCA fee?"  # answered and fact-checked in English
    assert ("ml", "en") in calls and ("en", "ml") in calls
    tokens = [e.text for e in events if e.type == "token"]
    assert tokens == ["ബിസിഎ ഫീസ് 62,500 രൂപയാണ്."]  # one translated answer, no English tokens leak
    assert "sources" in types and types[-1] == "done" and events[-1].answered


async def test_failed_translation_falls_back_to_english(monkeypatch):
    async def broken(text, src, tgt):
        if tgt == "en":
            return "What is the BCA fee?"
        raise TranslationError("changed a number")

    monkeypatch.setattr(service, "_answer_english", _fake_pipeline({}))
    monkeypatch.setattr(service, "translate", broken)
    events = await _collect(service.answer_stream("ബിസിഎ ഫീസ് എത്ര?", [], "ml"))
    text = "".join(e.text for e in events if e.type == "token")
    assert text.startswith("The BCA fee is Rs. 62,500.") and "shown in English" in text


async def test_english_question_is_streamed_unchanged(monkeypatch):
    async def never(*a):
        raise AssertionError("no translation for English")

    monkeypatch.setattr(service, "_answer_english", _fake_pipeline({}))
    monkeypatch.setattr(service, "translate", never)
    events = await _collect(service.answer_stream("What is the BCA fee?", [], "auto"))
    assert [e.text for e in events if e.type == "token"] == ["The BCA fee is ", "Rs. 62,500."]


async def test_previous_malayalam_turn_is_translated_for_follow_ups(monkeypatch):
    from app.llm.base import ChatMessage

    seen = {}

    async def fake_translate(text, src, tgt):
        return {"ബിസിഎ ഫീസ് എത്ര?": "What is the BCA fee?", "ഹോസ്റ്റൽ?": "And the hostel?"}.get(text, "ഉത്തരം 62,500.")

    monkeypatch.setattr(service, "_answer_english", _fake_pipeline(seen))
    monkeypatch.setattr(service, "translate", fake_translate)
    history = [ChatMessage("user", "ബിസിഎ ഫീസ് എത്ര?"), ChatMessage("assistant", "ബിസിഎ ഫീസ് 62,500 രൂപയാണ്.")]
    await _collect(service.answer_stream("ഹോസ്റ്റൽ?", history, "ml"))
    assert seen["question"] == "And the hostel?"
    assert [m.content for m in seen["history"]] == ["What is the BCA fee?"]


async def test_who_are_you_in_any_language_is_answered_at_once(monkeypatch):
    from app.i18n.messages import WELCOME
    from app.rag import service

    def boom(*_a, **_k):
        raise AssertionError("small talk must not translate or search")

    monkeypatch.setattr(service, "translate", boom)
    monkeypatch.setattr(service, "build_graph", boom)
    events = [e async for e in service.answer_stream("നിങ്ങൾ ആരാണ്?", [], language="auto")]
    assert [e.type for e in events] == ["language", "token", "done"]
    assert events[1].text == WELCOME["ml"]


async def test_fixed_replies_are_hand_written_not_translated(monkeypatch):
    from app.i18n.messages import NO_INFO
    from app.rag import service
    from app.rag.prompts import no_info_message

    async def english(question, history=None):
        yield service.ChatEvent(type="token", text=no_info_message())
        yield service.ChatEvent(type="done", answered=False)

    def boom(*_a, **_k):
        raise AssertionError("fixed replies must not be machine-translated (it garbled the contacts)")

    monkeypatch.setattr(service, "_answer_english", english)
    monkeypatch.setattr(service, "translate", boom)
    monkeypatch.setattr(service, "detect_language", lambda text, preferred=None: "en")
    events = [e async for e in service.answer_stream("bus timings?", [], language="ml")]
    text = "".join(e.text for e in events if e.type == "token")
    assert text.startswith(NO_INFO["ml"].split("{")[0]) and "info@sxca.edu.in" in text and "079-29708056/7" in text
