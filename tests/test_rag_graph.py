import pytest

from app.crawler.extract_html import extract_html
from app.llm.base import ChatMessage, LLMProvider, LLMResult
from app.rag import graph as graph_mod
from app.rag.bm25 import tokenize
from app.rag.factcheck import check_answer, is_no_info_answer
from app.rag.query import contextualize, expand_abbreviations
from app.rag.retriever import Retrieval, keyword_coverage
from app.rag.vectorstore import Hit


# ---------------------------------------------------------------- query handling
def test_follow_up_questions_get_context():
    history = [ChatMessage("user", "What is the eligibility for B.Sc Physics?"), ChatMessage("assistant", "…")]
    assert contextualize("and the fees?", history).startswith("What is the eligibility for B.Sc Physics?")
    assert contextualize("Who is the principal of the college?", history) == "Who is the principal of the college?"
    assert contextualize("anything", []) == "anything"


def test_abbreviations_expanded():
    q = expand_abbreviations("BCA fee")
    assert "Bachelor of Computer Applications" in q and "fee structure" in q
    assert "Bachelor of Commerce" in expand_abbreviations("B.Com eligibility")


def test_bm25_tokenizer_normalises_course_codes_and_amounts():
    assert "bcom" in tokenize("B.Com. (Hons)")
    assert "bsc" in tokenize("B.Sc Physics")
    assert "20000" in tokenize("Rs. 20,000 per year")
    assert "the" not in tokenize("the fee")


def test_keyword_coverage():
    assert keyword_coverage("BCA fee", "Fee structure for BCA 2026-27") == 1.0
    assert keyword_coverage("BCA fee", "Hostel rules") == 0.0


# ---------------------------------------------------------------- fact check
CONTEXT = "B.Com fee: Rs. 20,000 per year. Contact admissions@sxca.edu.in or 079-29708056. https://sxca.edu.in/fees/"


def test_fact_check_accepts_supported_facts():
    ans = "The B.Com fee is ₹20,000 per year. Email admissions@sxca.edu.in, call 079 2970 8056 (https://sxca.edu.in/fees/)."
    assert check_answer(ans, CONTEXT).ok


@pytest.mark.parametrize("bad", [
    "The B.Com fee is ₹25,000 per year.",
    "Email fees@sxca.edu.in for details.",
    "Call 079-26301111.",
    "See https://sxca.edu.in/other-page/",
])
def test_fact_check_rejects_invented_facts(bad):
    assert not check_answer(bad, CONTEXT).ok


def test_no_info_detection():
    assert is_no_info_answer("I don't have that information. Please contact the college office.")
    assert is_no_info_answer("The provided context does not contain information on the application process.")
    assert not is_no_info_answer("The fee is Rs. 20,000.")


# ---------------------------------------------------------------- extraction
def test_faq_accordion_questions_become_headings():
    html = ('<html><body><main><div class="elementor-accordion">'
            '<div class="elementor-tab-title"><a class="elementor-accordion-title">Is there a hostel?</a></div>'
            '<div class="elementor-tab-content">Yes, for women.</div>'
            '<div class="elementor-tab-title"><a class="elementor-accordion-title">What documents are needed?</a></div>'
            '<div class="elementor-tab-content">Marksheet and ID.</div></div></main></body></html>')
    text = extract_html(html, "https://sxca.edu.in/admissions/faqs/").text
    assert "#### Is there a hostel?" in text and "#### What documents are needed?" in text


# ---------------------------------------------------------------- graph routing
class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, answer: str):
        super().__init__(model="fake", temperature=0, max_output_tokens=100)
        self.answer = answer
        self.calls = 0

    async def generate(self, system, messages):
        self.calls += 1
        return LLMResult(text="rewritten query")

    async def stream(self, system, messages):
        self.calls += 1
        for word in self.answer.split(" "):
            yield word + " "


def _hit(text: str, grade: float) -> Hit:
    return Hit(chunk_id="1:0", text=text, score=grade, metadata={"url": "https://sxca.edu.in/fees/", "title": "Fees"},
               signals={"grade": grade})


async def _run(monkeypatch, retrieval: Retrieval, answer: str):
    llm = FakeLLM(answer)
    monkeypatch.setattr(graph_mod, "get_llm", lambda: llm)
    monkeypatch.setattr(graph_mod, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(graph_mod, "hybrid_search", lambda *a, **k: retrieval)
    events, final = [], {}
    async for mode, chunk in graph_mod.build_graph().astream({"question": "BCA fee?", "history": []},
                                                              stream_mode=["custom", "values"]):
        if mode == "custom":
            events.append(chunk)
        else:
            final = chunk
    return events, final, llm


async def test_graph_answers_from_relevant_context(monkeypatch):
    r = Retrieval(hits=[_hit(CONTEXT, 0.7)], best_score=0.7, relevant=True)
    events, final, _ = await _run(monkeypatch, r, "The B.Com fee is Rs. 20,000 per year.")
    assert final["answered"] is True
    assert [e["type"] for e in events][0] == "sources"
    assert not any(e["type"] == "replace" for e in events)


async def test_graph_refuses_without_llm_when_nothing_relevant(monkeypatch):
    r = Retrieval(hits=[], best_score=0.2, relevant=False)
    events, final, llm = await _run(monkeypatch, r, "should not be used")
    assert final["answered"] is False
    assert llm.calls == 1  # only the query rewrite, never an answer generation
    assert "don't have that information" in events[-1]["text"]


async def test_graph_replaces_hallucinated_fee(monkeypatch):
    r = Retrieval(hits=[_hit(CONTEXT, 0.7)], best_score=0.7, relevant=True)
    events, final, _ = await _run(monkeypatch, r, "The B.Com fee is Rs. 45,000 per year.")
    assert final["answered"] is False
    assert events[-1]["type"] == "replace"
    assert "fact check failed" in final["reason"]


async def test_wrong_figure_gets_one_corrected_retry(monkeypatch):
    r = Retrieval(hits=[_hit(CONTEXT, 0.7)], best_score=0.7, relevant=True)
    answers = iter(["The B.Com fee is Rs. 45,000 per year.", "The B.Com fee is Rs. 20,000 per year."])

    class Retrying(FakeLLM):
        async def stream(self, system, messages):
            self.calls += 1
            if self.calls == 2:  # the retry is told which figure was wrong
                assert "45,000" in messages[-1].content
            for word in next(answers).split(" "):
                yield word + " "

    llm = Retrying("")
    monkeypatch.setattr(graph_mod, "get_llm", lambda: llm)
    monkeypatch.setattr(graph_mod, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(graph_mod, "hybrid_search", lambda *a, **k: r)
    final = await graph_mod.build_graph().ainvoke({"question": "B.Com fee?", "history": []})
    assert final["answered"] is True and final["answer"] == "The B.Com fee is Rs. 20,000 per year."


async def test_only_the_wrong_line_is_dropped_when_the_retry_is_wrong_too(monkeypatch):
    r = Retrieval(hits=[_hit(CONTEXT, 0.7)], best_score=0.7, relevant=True)
    events, final, llm = await _run(
        monkeypatch, r, "The B.Com fee is Rs. 20,000 per year.\n- Library fee: Rs. 1,500\n- Email: admissions@sxca.edu.in")
    assert llm.calls == 2  # answer + one retry
    assert final["answered"] is True
    assert "1,500" not in final["answer"] and "20,000" in final["answer"] and "admissions@" in final["answer"]
    assert events[-1] == {"type": "replace", "text": final["answer"]}


def test_fee_period_must_match_the_source():
    sem = "UNDERGRADUATE (UG): S.No; Programme; 2026-27 (Sem-1)\nUNDERGRADUATE (UG): 1; B.S.(BCA); 31,250"
    assert check_answer("The BCA fee for 2026-27 is ₹31,250 for Semester 1.", sem).ok
    fc = check_answer("The BCA fee is ₹31,250 per year.", sem)
    assert not fc.ok and "per year" in fc.unsupported
    assert not check_answer("BCA costs ₹31,250 annually.", sem).ok
    assert check_answer("The B.Com fee is Rs. 20,000 per year.", CONTEXT).ok  # the source says per year
    assert check_answer("Admissions open every year in May.", "Admissions open every year in May.").ok


@pytest.mark.parametrize("answer, wrong, kept", [
    # a wrong period is removed, the checked amount stays
    ("The total approved fee for B.S. (BCA) is ₹31,250 per year.\n* Library: ₹1,000",
     ["per year"], "The total approved fee for B.S. (BCA) is ₹31,250.\n* Library: ₹1,000"),
    # a wrong figure in running text drops only its sentence, never splitting at "B.S." or "M.Sc."
    ("The M.Sc. AI fee is ₹50,000 for Sem-1. The library fee is ₹1,500.", ["1,500"],
     "The M.Sc. AI fee is ₹50,000 for Sem-1."),
    # a wrong bullet is dropped
    ("BCA fee: ₹31,250.\n* Misc & Library: ₹1,500\n* Society: ₹1,600", ["1,500"], "BCA fee: ₹31,250.\n* Society: ₹1,600"),
    # nothing real left
    ("The BCA fee is ₹99,999.", ["99,999"], ""),
])
def test_drop_unsupported_lines(answer, wrong, kept):
    from app.rag.factcheck import drop_unsupported_lines

    assert drop_unsupported_lines(answer, wrong) == kept


def test_fact_check_understands_short_academic_years():
    ctx = "Scholarships for the academic year 2024-25 are listed below."
    assert check_answer("Scholarships for 2024-2025 are available.", ctx).ok
    assert not check_answer("Scholarships for 2019-2020 are available.", ctx).ok


CALENDAR = ("Academic Calendar 2026-27. Last date for Subject selection & fees payment for Sem III, V & VII (UG): "
            "25 June 2026. After the deadline of 19/06/2025, a late fee of Rs.250/- will be charged.")


@pytest.mark.parametrize("good", [
    "The last date to pay the fees is 25 June 2026.",
    "Pay by June 25th, 2026.",
    "The deadline is 25/06/2026.",
    "The old deadline was 19 June 2025.",
    "Fees are due on the 25th of June.",
])
def test_fact_check_accepts_dates_from_the_sources(good):
    assert check_answer(good, CALENDAR).ok


@pytest.mark.parametrize("bad", [
    # The real mix-up: late-fee days invented next to a correct deadline.
    "The last date is 25 June 2026; late fees apply from 26–29 June 2026.",
    "The last date is 26 June 2026.",
    "The last date is 25 June 2025.",
    "Pay between June 20 and 25, 2026.",
    "The deadline is 30/06/2026.",
])
def test_fact_check_rejects_invented_or_mixed_dates(bad):
    assert not check_answer(bad, CALENDAR).ok


def test_fact_check_accepts_source_dates_shown_to_the_model():
    from app.rag.prompts import format_context

    hit = Hit(chunk_id="1:0", text="Examination Office: coe@sxca.edu.in", score=0.7,
              metadata={"url": "https://sxca.edu.in/contact-us/", "title": "Contact us", "date": "2026-03-25"})
    ans = "Email coe@sxca.edu.in (contact page updated 2026-03-25)."
    assert check_answer(ans, format_context([hit])).ok
    assert check_answer("The contact page was updated on 25 March 2026.", format_context([hit])).ok
