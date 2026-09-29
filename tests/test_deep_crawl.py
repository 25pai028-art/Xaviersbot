from app.crawler.boilerplate import common_information, find_boilerplate, strip_boilerplate
from app.crawler.extract_html import extract_html
from app.crawler.pipeline import choose_document_title
from app.crawler.urls import classify

BASE = "https://sxca.edu.in/page/"


def page(body: str) -> str:
    return f"<html><head><title>T</title></head><body><main>{body}</main></body></html>"


def test_faculty_profiles_are_crawled():
    assert classify("https://sxca.edu.in/author/rashmi-yadav/") == "html"


def test_hidden_contact_attributes_become_text():
    html = page('<h1>Dr. R Yadav</h1><p><button class="profile_email" data-emailid="r.yadav@sxca.edu.in">View Email</button></p>'
                '<p><a href="mailto:office@sxca.edu.in">Write to us</a> <a href="tel:+917926308055">Call</a></p>')
    text = extract_html(html, BASE).text
    assert "Email: r.yadav@sxca.edu.in" in text
    assert "Email: office@sxca.edu.in" in text
    assert "Phone: +917926308055" in text
    assert "View Email" not in text


def test_adjacent_links_do_not_glue_together():
    html = page('<p><a href="/a/">Dr. Bijal Shah</a><a href="/b/">Dr. Mallika Sanyal</a></p>')
    assert "Dr. Bijal Shah Dr. Mallika Sanyal" in extract_html(html, BASE).text


def test_accordion_panels_kept_but_hidden_text_and_mobile_copies_dropped():
    html = page('<div class="elementor-accordion"><div class="elementor-tab-content" hidden>Hostel fee answer</div></div>'
                '<p hidden>secret instruction</p>'
                '<div class="elementor-hidden-desktop">Mobile duplicate</div>')
    text = extract_html(html, BASE).text
    assert "Hostel fee answer" in text
    assert "secret instruction" not in text
    assert "Mobile duplicate" not in text


def test_links_discovered_from_iframes_viewers_data_attrs_onclick_and_text():
    html = page(
        '<iframe src="https://sxca.edu.in/viewer.html?file=https%3A%2F%2Fsxca.edu.in%2Fwp-content%2Fuploads%2F2026%2F01%2Fprospectus.pdf"></iframe>'
        '<div data-href="/admissions/"></div>'
        '<span onclick="window.open(\'/downloads/forms/\')">Forms</span>'
        '<p>Details at https://sxca.edu.in/exam-notices/ today.</p>')
    links = {u for u, _ in extract_html(html, BASE).links}
    assert "https://sxca.edu.in/wp-content/uploads/2026/01/prospectus.pdf" in links
    assert "https://sxca.edu.in/admissions/" in links
    assert "https://sxca.edu.in/downloads/forms/" in links
    assert "https://sxca.edu.in/exam-notices/" in links


def test_breadcrumb_section_from_url_and_faculty():
    assert extract_html(page("<p>x</p>"), "https://sxca.edu.in/admissions/ug-admissions/ug-science/").section \
        == "Admissions › Ug Admissions"
    assert extract_html(page("<p>x</p>"), "https://sxca.edu.in/author/rashmi-yadav/").section == "Faculty profile"


def test_site_wide_boilerplate_removed_and_kept_once():
    pages = [f"## Page {i}\nUnique fact {i}\nSubscribe to Newsletter\nCall 079-29708056" for i in range(30)]
    bp = find_boilerplate(pages)
    assert "subscribe to newsletter" in bp and "call 079-29708056" in bp
    assert not any(f"unique fact {i}" in bp for i in range(30))
    cleaned = strip_boilerplate(pages[0], bp)
    assert "Unique fact 0" in cleaned and "Subscribe" not in cleaned
    common = common_information(pages, bp)
    assert common.count("Call 079-29708056") == 1


def test_document_title_prefers_link_text_over_file_name():
    url = "https://sxca.edu.in/wp-content/uploads/2026/06/Notice-23.pdf"
    assert choose_document_title(url, link_text="Fee structure B.Com 2026-27", api_title="Notice-23",
                                 doc_title="") == "Fee structure B.Com 2026-27"
    assert choose_document_title(url, link_text="Download", api_title="Notice 23", doc_title="Microsoft Word - Exam Timetable") \
        == "Exam Timetable"
    assert choose_document_title(url, link_text=None, api_title=None, doc_title=None) == "Notice 23"


def test_staff_cards_keep_role_with_the_right_person():
    html = page('<h2>Faculty Members</h2>'
                '<div class="card"><div class="content"><h2 class="drName">Dr. Pravida Raja A.C.</h2>'
                '<h4 class="drPosition">Head, Assistant Professor</h4></div></div>'
                '<div class="card"><div class="content"><h2 class="drName">Nisarg Vyas</h2>'
                '<h4 class="drPosition">Assistant Professor</h4></div></div>')
    text = extract_html(html, BASE).text
    assert "Dr. Pravida Raja A.C. — Head, Assistant Professor" in text
    assert "Nisarg Vyas — Assistant Professor" in text


def test_headings_without_content_are_not_dropped_by_chunker():
    from app.rag.chunker import chunk_text

    text = "## Faculty Members\n\n### Dr. A\n\n### Dr. B\n\n## About\n\nThe department was founded in 2001."
    joined = "\n".join(c.text for c in chunk_text(text, title="Dept", counter=lambda s: len(s.split())))
    assert "Dr. A" in joined and "Dr. B" in joined and "founded in 2001" in joined


def test_footer_leadership_and_contact_are_captured_without_menus():
    html = ('<html><body><main><p>Page body</p></main><footer class="footer-section">'
            '<ul class="f-link"><li><a href="/academics/departments/">Departments</a></li></ul>'
            '<ul class="f-link contact"><li><span class="name">Fr. Dr. David K. Roy SJ</span>'
            '<span class="designation">Director</span></li>'
            '<li><span class="name">Dr. Sebastian. V.A.</span><span class="designation">Principal</span></li></ul>'
            '<p>Navrangpura, Ahmedabad-380009.</p><a href="mailto:info@sxca.edu.in">info@sxca.edu.in</a>'
            '<p>© Copyright 2021-2025 by St Xavier\'s College. Website Design Company: Atlas SoftWeb</p>'
            '</footer></body></html>')
    page = extract_html(html, BASE)
    assert "Fr. Dr. David K. Roy SJ — Director" in page.footer
    assert "Dr. Sebastian. V.A. — Principal" in page.footer
    assert "info@sxca.edu.in" in page.footer and "380009" in page.footer
    assert "Departments" not in page.footer and "Copyright" not in page.footer
    assert "Director" not in page.text  # the footer is indexed once site-wide, not on every page
