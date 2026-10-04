"""Which documents are kept out of search: junk and older editions (app.crawler.quality)."""
import pytest

from app.crawler.quality import edition_year, junk_reason, series_key, superseded_ids

U = "https://sxca.edu.in/wp-content/uploads"


@pytest.mark.parametrize("text", [
    "Annexure\nList of Books in Library\n" + "AccessionNo: 1; Title: X; Author: Y\n" * 60_000,
    "Curriculum Feedback: Students (2024-2025)\nStrongly Agree / Agree / Neutral\n" * 3,
    "Student's Feedback Form\nhttps://docs.google.com/forms/d/abc/edit#responses\nRohan Shah",
    "Student's Feedback on BCA Curriculum 2024-25\nAverage\nGood\nExcellent",
    "21ST BOARD OF STUDIES\nCHEMISTRY DEPARTMENT\nFEEDBACK FROM STUDENTS\n(40 RESPONSES)",
    "INTERNSHIP REPORT\nName: Jayana Rahevar\nRoll No: 23-PPS-033\nCourse Code: PPS-4801",
    "CERTIFICATE\nThis is to certify that Kashish Gupta, student of Master's in Clinical Psychology, has completed",
    "TO WHOMSOEVER IT MAY CONCERN\nThis letter is to confirm that Mr. Sourav Charan has been working with us as an Intern",
    "EFFICACIOUS OF ACTING\nA thesis submitted for the Non-Funded Undergraduate Research\nSubmitted by:\nTvesha Raval\n23-PSH-067",
    "Understanding Parasocial Relationships\nPradhi Mehta (23-PSH-022)\nTYBA (SF Psychology)\nSubmitted to\nProf. Stephen",
    "Botany Research Project:2024-2025\nSEEDS IN THE DIGITAL ENVIRONMENT\nBY\nMISS. GINI VYAS\n(23-BO-0618)",
], ids=lambda t: t[:30])
def test_junk_is_recognised(text):
    assert junk_reason(text)


@pytest.mark.parametrize("text", [
    # A syllabus that describes a project report is not a student's report
    "M.Sc. Artificial Intelligence Syllabus\nSemester IV\nCORE Paper: Internship. A final project report must be\n"
    "submitted, followed by a viva, guided by two supervisors: one from the industry.",
    # A college letter addressed to whomever it may concern, about students but not an internship
    "To whomsoever it may concern\nIn the academic year 2020-21 the College was functional online and students "
    "were mentored through various methodologies.",
    "The fee structure approved by the Governing Body for the Self-Financed Programmes 2026-27.",
    "St. Xavier's College Handbook 2026-27. Students must maintain 75% attendance.",
], ids=lambda t: t[:30])
def test_official_documents_are_kept(text):
    assert junk_reason(text) is None


def test_series_and_edition_years_from_file_names():
    assert series_key(f"{U}/2025/06/SXC-Handbook-2025-26.pdf") == series_key(f"{U}/2026/06/SXCA-Handbook-2026-27.pdf")
    assert series_key(f"{U}/2024/01/Prospectus-23-24.pdf") == "prospectus"
    assert series_key(f"{U}/2024/11/Annual-Report-2023-24.pdf") is None  # history, not a yearly replacement
    assert edition_year(f"{U}/2024/01/Prospectus-23-24.pdf", None) == 2023
    assert edition_year(f"{U}/2024/08/Prospectus-2024-25.pdf", "2019-20") == 2024  # the file name wins
    assert edition_year(f"{U}/2022/02/Academic-Calendar-2020.pdf", None) == 2020


def test_only_the_newest_edition_is_searched():
    docs = [
        (1, f"{U}/2025/06/SXC-Handbook-2025-26.pdf", None),
        (2, f"{U}/2026/06/SXCA-Handbook-2026-27.pdf", "2026-27"),
        (3, f"{U}/2021/07/Policy-Scholarships-and-Freeships-21-22-1.pdf", None),
        (4, f"{U}/2026/03/Policy-Scholarship-and-Freeships-25-26.pdf", None),
        (5, f"{U}/2021/07/Policy-Differently-abled-students-21-22-1.pdf", None),
        (6, f"{U}/2026/03/Policy-Differerently-abled-students-25-26.pdf", None),  # typo on the website
        (7, f"{U}/2024/11/Annual-Report-2023-24.pdf", "2023-24"),
        (8, f"{U}/2026/04/Annual-Report-of-SXCA-2025-26.pdf", "2025-26"),
        (9, "https://admissions.sxca.edu.in/SXCA/downloads/Fees Structure 2026-2027 Self Financed - UG&PG.pdf", None),
        (10, "https://admissions.sxca.edu.in/SXCA/downloads/Fees Structure 2026-2027 Grant-In-Aid - UG&PG.pdf", None),
    ]
    assert set(superseded_ids(docs)) == {1, 3, 5}


def test_clean_stage_sets_documents_aside_and_brings_them_back(monkeypatch):
    from sqlalchemy import select

    from app.crawler import pipeline
    from app.db.models import Source
    from app.db.session import session_scope

    monkeypatch.setattr(pipeline.indexer, "remove_source", lambda src: None)
    base = "https://quality-test.sxca.edu.in/wp-content/uploads"
    docs = {
        f"{base}/2026/07/JAYANA-RAHEVAR.pdf": "INTERNSHIP REPORT\nName: Jayana Rahevar\nRoll No: 23-PPS-033",
        f"{base}/2025/06/SXC-Handbook-2025-26.pdf": "Handbook 2025-26. Library open 9 to 5.",
        f"{base}/2026/06/SXCA-Handbook-2026-27.pdf": "Handbook 2026-27. Library open 8 to 6.",
    }
    with session_scope() as db:
        for url, text in docs.items():
            db.add(Source(url=url, source_type="website", content_type="pdf", raw_text=text, text=text,
                          index_status="indexed", chunk_count=3))

    def statuses():
        with session_scope() as db:
            return {s.url.rsplit("/", 1)[-1]: (s.index_status, s.excluded_reason)
                    for s in db.scalars(select(Source).where(Source.url.startswith(base)))}

    pipeline.Crawler()._set_aside_documents()
    st = statuses()
    assert st["JAYANA-RAHEVAR.pdf"][0] == "excluded" and "student" in st["JAYANA-RAHEVAR.pdf"][1]
    assert st["SXC-Handbook-2025-26.pdf"][0] == "superseded"
    assert st["SXCA-Handbook-2026-27.pdf"] == ("indexed", None)

    with session_scope() as db:  # the 2026-27 handbook disappears from the website
        db.delete(db.scalar(select(Source).where(Source.url == f"{base}/2026/06/SXCA-Handbook-2026-27.pdf")))
    pipeline.Crawler()._set_aside_documents()
    assert statuses()["SXC-Handbook-2025-26.pdf"] == ("pending", None)  # searchable again after re-indexing
