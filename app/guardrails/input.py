"""Input guardrails, applied before any retrieval or LLM call.

- Abusive language: profanity/insults are removed and the factual question (if any)
  is answered neutrally; abuse alone gets a calm, de-escalating reply.
- Policy violations (cheating, hacking, forgery, marks tampering): refused with the
  legitimate administrative channel. The LLM is never asked, so it cannot comply.
- Out of scope (sports, weather, recipes, coding help, general knowledge): polite
  "I only help with St. Xavier's College questions" fallback.

Rule-based on purpose: instant on a CPU, predictable, and easy for the college to extend.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------- abusive language
# English + common Hindi/Gujarati abuse (Roman script and native scripts). Matched as whole words.
_PROFANITY = [
    r"f+u+c+k+\w*", r"f\W?ck\w*", r"f[*#@$]+(c?k)?", r"sh[*#@$]+t", r"b[*#@$]+ch", r"fk+", r"wtf", r"stfu", r"sh[i1]t+\w*", r"bullshit", r"b[i1]tch\w*", r"bastard\w*",
    r"a+ss+hole\w*", r"dick(head)?s?", r"crap", r"damn", r"piss\w*", r"moron\w*", r"idiot\w*", r"stupid", r"dumb\w*",
    r"useless", r"retard\w*", r"loser\w*", r"suck\w*", r"shut up", r"go to hell",
    r"bc", r"mc", r"bkl", r"bsdk", r"chutiy\w*", r"chutia\w*", r"bhenchod\w*", r"behenchod\w*", r"madarchod\w*",
    r"gandu\w*", r"gaand\w*", r"gaandu\w*", r"harami\w*", r"kamina\w*", r"kutt[ae]\w*", r"saala\w*", r"sala", r"lod[ue]\w*",
    r"lavd[ae]\w*", r"lund", r"randi\w*", r"ullu", r"bewakoof\w*", r"pagal", r"nalayak\w*",
    r"चूतिया", r"चुतिया", r"मादरचोद", r"भेनचोद", r"बहनचोद", r"गांडू", r"हरामी", r"कमीना", r"कुत्ते", r"साला",
    r"बेवकूफ", r"પાગલ", r"હરામી", r"ગાંડો", r"ચૂતિયા", r"મૂર્ખ",
]
PROFANITY = re.compile(r"(?<![\w*#@$])(" + "|".join(_PROFANITY) + r")(?![\w*#@$])", re.IGNORECASE | re.UNICODE)
# Filler that remains after removing abuse ("you ___ bot", "hey ___!!").
_FILLER = re.compile(r"\b(you|u|ur|your|are|r|is|this|bot|chatbot|stupid|hey|oi|man|bro|yaar|re|ka|ki|ke|na|so|such|a|an)\b",
                     re.IGNORECASE)

# ---------------------------------------------------------------- policy violations
_MISCONDUCT = [
    # exam paper / answer leaks and cheating
    r"\b(leak(ed)?|get|send|share|buy|sell|give)\b.{0,40}\b(question\s*paper|exam\s*paper|answer\s*key|paper)s?\b.{0,30}\b(before|in advance|early|leak(ed)?|tomorrow)",
    r"\b(question\s*paper|exam\s*paper|answer\s*key)s?\b.{0,30}\b(leak(ed)?|in advance|before the exam)",
    r"\b(cheat|cheating)\b.{0,30}\b(exam|test|paper|quiz|viva|assignment)",
    r"\bcopy(ing)?\b\s+(in|during|from someone in)\s+(the\s+|my\s+)?(exam|test|quiz|viva)",
    r"\b(exam|test)\b.{0,30}\b(cheat|cheating|copy without|chits?|hidden phone|smart ?watch)",
    r"\bchits?\b.{0,20}\b(exam|hide)",
    # hacking / unauthorised access
    r"\b(hack|hacking|crack|cracking|bypass|brute[\s-]?force|phish|ddos|sql\s*injection|exploit|keylogger)\w*\b.{0,40}"
    r"\b(portal|erp|lms|moodle|website|site|server|account|password|login|wi-?fi|system|database|admin|results?|marks?)",
    r"\b(someone else'?s?|another student'?s?|teacher'?s?|professor'?s?|admin)\b.{0,20}\b(password|login|account|credentials)",
    r"\b(steal|stealing|sniff|intercept)\b.{0,20}\b(password|credentials|login details)",
    # tampering / forgery / proxy
    r"\b(change|edit|alter|increase|modify|tamper|manipulate|fix)\w*\b.{0,25}\b(my\s+)?(marks|grades?|attendance|results?|cgpa|sgpa)\b"
    r".{0,30}\b(without|secretly|in the system|in the portal|in erp|database|hack)",
    r"\b(tamper|manipulat)\w*\b.{0,25}\b(marks|grades?|attendance|results?)",
    r"\b(fake|forge|forged|forging|bogus|make a false)\b.{0,25}\b(certificate|marksheet|mark\s*sheet|bonafide|signature|"
    r"medical|id\s*card|document|attendance|receipt)",
    r"\bproxy\b.{0,20}\b(attendance|exam|class|lecture)|\b(mark|give)\b.{0,15}\battendance\b.{0,20}\b(for my friend|for someone|absent)",
    # plagiarism evasion
    r"\b(bypass|beat|avoid|fool|trick)\b.{0,25}\b(plagiarism|turnitin|ai detector|urkund|ouriginal)",
]
MISCONDUCT = re.compile("|".join(f"(?:{p})" for p in _MISCONDUCT), re.IGNORECASE | re.DOTALL)

# ---------------------------------------------------------------- out of scope
_OFF_TOPIC = [
    r"\b(ipl|cricket|football|fifa|world cup|match score|live score)\b", r"\bweather\b", r"\b(movie|film|web ?series|netflix|song|lyrics)s?\b",
    r"\b(recipe|cook|cooking)\b", r"\b(joke|poem|shayari|story)\b", r"\b(horoscope|astrology|zodiac)\b",
    r"\b(bitcoin|crypto|stock|share market|sensex|nifty)\b", r"\b(election|prime minister|chief minister|political party)\b",
    r"\bcapital of\b", r"\b(write|generate|debug|fix)\b.{0,20}\b(code|program|script|function|sql query)\b",
    r"\b(solve|integrate|differentiate|calculate)\b.{0,30}\b(equation|integral|derivative|problem|sum)\b",
    r"\btranslate\b", r"\b(chatgpt|openai)\b",
    r"\b(girlfriend|boyfriend|date me|love me|marry)\b",
]
OFF_TOPIC = re.compile("|".join(_OFF_TOPIC), re.IGNORECASE)

# Words that make a question about the college (used when retrieval finds nothing).
COLLEGE_TOPIC = re.compile(
    r"\b(college|xavier'?s?|sxca|campus|admissions?|admit|apply|application|eligib\w*|fees?|tuition|scholarship|freeship|"
    r"course|courses|programme|program|degree|department|dept|faculty|professor|prof|teacher|lecturer|sir|madam|ma'?am|hod|"
    r"head|principal|staff|exam\w*|results?|marks?|grade|syllabus|timetable|time table|calendar|semester|sem|attendance|"
    r"hostel|library|canteen|cafeteria|lab|laboratory|wi-?fi|placement|career|internship|certificate|bonafide|migration|"
    r"transcript|id card|notice|holiday|club|fest|event|nss|ncc|sports|alumni|office|contact|email|phone|timings?|hours|"
    r"ug|pg|bcom|b\.com|bca|bba|bsc|b\.sc|ba|b\.a|ma|m\.a|msc|m\.sc|mcom|m\.com|phd|research|nep|naac|iqac|nirf|"
    r"class|lecture|student|parking|uniform|dress code|ragging|counsell?ing|medical|clinic|refund|payment|"
    r"withdraw\w*|elective|dean|convocation|graduation|prospectus|brochure|form)\b",
    re.IGNORECASE,
)
_NON_LATIN = re.compile(r"[^\x00-\x7F]")

# ---------------------------------------------------------------- small talk
# Whole-message conversational turns: answered instantly, never searched.
_END = r"[\s!.?,:)🙂😊👍🙏]*$"
SMALL_TALK: list[tuple[str, re.Pattern]] = [
    ("greeting", re.compile(r"^(hi+|hello+|hey+|hii+|helo|hlo|good (morning|afternoon|evening|day)|namaste|namaskar|"
                            r"kem cho|hola|greetings|yo|नमस्ते|નમસ્તે)( there| bot| xavier'?s assistant)?" + _END, re.I)),
    ("thanks", re.compile(r"^(thanks?( a lot| so much| you( so much| very much)?)?|thank u|thx|ty|tysm|dhanyavaa?d|"
                          r"shukriya|aabhar|abhar|धन्यवाद|ધન્યવાદ|thanks? (for|that) .{0,30})" + _END, re.I)),
    ("bye", re.compile(r"^(bye+|good ?bye|see (you|ya)( later)?|tata|take care|good night|that'?s all|nothing else|"
                       r"no thanks?|no,? that'?s all)" + _END, re.I)),
    ("how_are_you", re.compile(r"^(how are (you|u)|how r u|how'?s it going|kaise ho|kem cho majama|what'?s up|sup)( today)?"
                               + _END, re.I)),
    ("identity", re.compile(r"^(who are (you|u)|what are you|what can you do|what do you do|are you (an? )?(ai|bot|robot|"
                            r"human|real person|real|chatgpt|gpt|a person)|who (made|created|built|developed) you|"
                            r"what('?s| is) your name|your name|who is this|what is this|who am i (talking|chatting|speaking) "
                            r"(to|with)|tell me about (yourself|you)|introduce yourself|what should i call you|"
                            r"what can i ask( you)?|how can you help( me)?|"
                            r"(aap|ap|tum|tu|tame|tamey|tu) (kaun|kon|koun) (ho|hai|hain|cho|che|chho|chhe)|"
                            r"kaun (ho|hai) (aap|tum)|"
                            # Hindi / Marathi, Gujarati, Malayalam, Tamil, Telugu, Kannada, Bengali, Punjabi
                            r"(आप|तुम|तू|तुम्ही) कौन (हैं|हो|है)|(आप|तुम्ही|तू) कोण (आहात|आहेस)|"
                            r"(તમે|તું) કોણ (છો|છે)|"
                            r"(നിങ്ങൾ|നീ) ആരാണ്|"
                            r"(நீங்கள்|நீ) யார்|"
                            r"(మీరు|నువ్వు) ఎవరు|"
                            r"(ನೀವು|ನೀನು) ಯಾರು|"
                            r"(আপনি|তুমি) কে|"
                            r"(ਤੁਸੀਂ|ਤੂੰ) ਕੌਣ (ਹੋ|ਹੈਂ))"
                            + _END, re.I)),
    ("ack", re.compile(r"^(ok+a*y*|okie|k+|kk|alright|all right|fine|got it|understood|i see|cool|great|nice|good|"
                       r"awesome|perfect|hmm+|ah+|oh+|sure|yes|yeah|yep|no|nope|noted|done|right|haan|ha|theek hai|"
                       r"thik hai|achha|acha|barabar|saru)( then| thanks?| thank you)?" + _END, re.I)),
]
# "hi, who are you?" / "hello there, what can you do"
_LEADING_GREETING = re.compile(r"^(hi+|hello+|hey+|helo|hlo|namaste|namaskar|good (morning|afternoon|evening))"
                               r"( there| bot)?[\s,.!-]+(?=\w)", re.I)
# Words people tack on that don't change the question: "who are you bro", "what can you do exactly?"
_FILLER_END = re.compile(r"(\s+(bro|bruh|buddy|dude|man|sir|madam|ma'?am|please|pls|plz|exactly|actually|really|"
                         r"again|then|yaar|bhai|ji))+(?=[\s!.?,:)🙂😊👍🙏]*$)", re.I)
# A leading acknowledgement before a real question: "ok, and what is the BCA fee?"
_LEADING_ACK = re.compile(r"^(ok+a*y*|okie|alright|fine|got it|cool|great|thanks?|thank you|hmm+|sure|yes|no|acha|achha)"
                          r"[\s,.!-]+(?=\w)", re.I)


@dataclass
class GuardResult:
    kind: Literal["ok", "misconduct", "off_topic", "abuse_only", "small_talk"]
    question: str  # cleaned question (profanity removed)
    abusive: bool = False
    small_talk: str = ""  # greeting | thanks | bye | how_are_you | identity | ack


# How an insult is addressed ("you ___ bot", "…, bot") — removed together with the insult.
_ADDRESSING = re.compile(r"[,;:]?\s*\b(?:(?:you|u|this|hey)\s+)?(?:chat)?bot\b[!?.]*", re.IGNORECASE)


def _strip_profanity(text: str) -> str:
    cleaned = _ADDRESSING.sub(" ", PROFANITY.sub(" ", text))
    cleaned = re.sub(r"[!?.]{2,}", lambda m: m.group(0)[0], cleaned)  # "???!!!" → "?"
    return re.sub(r"\s{2,}", " ", cleaned).strip(" ,.!-")


def small_talk_kind(question: str) -> str:
    """greeting | thanks | bye | how_are_you | identity | ack, or "" for a real question. Chat spelling and a
    greeting in front are allowed: "hi, who r u bro?" is identity."""
    q = question.strip()
    for text in dict.fromkeys((q, _casual(q))):
        for kind, pattern in SMALL_TALK:
            if pattern.match(text):
                return kind
        rest = _LEADING_GREETING.sub("", text)
        if rest != text:
            for kind, pattern in SMALL_TALK:
                if pattern.match(rest):
                    return "greeting" if kind == "ack" else kind  # "hi, ok" is a greeting
    return ""


def _casual(q: str) -> str:
    q = re.sub(r"\s+", " ", q.lower())
    q = re.sub(r"\br\b", "are", q)
    q = re.sub(r"\b(u|ya|yu)\b", "you", q)
    q = re.sub(r"\bur\b", "your", q)
    q = re.sub(r"\bwat\b|\bwht\b", "what", q)
    return _FILLER_END.sub("", q)


def check_input(question: str) -> GuardResult:
    q = question.strip()
    kind = small_talk_kind(q)
    if kind:
        return GuardResult(kind="small_talk", question=q, small_talk=kind)
    q = _LEADING_ACK.sub("", q)  # "ok, what about hostel?" → "what about hostel?"
    if MISCONDUCT.search(q):
        return GuardResult(kind="misconduct", question=q)
    abusive = bool(PROFANITY.search(q))
    cleaned = _strip_profanity(q) if abusive else q
    if abusive:
        leftover = _FILLER.sub(" ", cleaned)
        if len(re.findall(r"\w{2,}", leftover)) < 2:  # nothing factual left to answer
            return GuardResult(kind="abuse_only", question=cleaned, abusive=True)
    # Off-topic unless the rest of the question (outside the off-topic words) is about the college:
    # "film club" is a college question, "write a Python program" is not.
    if OFF_TOPIC.search(cleaned) and not COLLEGE_TOPIC.search(OFF_TOPIC.sub(" ", cleaned)):
        return GuardResult(kind="off_topic", question=cleaned, abusive=abusive)
    return GuardResult(kind="ok", question=cleaned, abusive=abusive)


def is_about_college(text: str) -> bool:
    """Heuristic: does the question mention a college topic? (Non-English text is assumed to.)"""
    return bool(COLLEGE_TOPIC.search(text)) or bool(_NON_LATIN.search(text))
