import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


MessageRole = Literal["user", "assistant"]

MEETING_FORMATS = ("онлайн", "в офисе", "по телефону")
BOOKING_FIELDS = ("name", "contact", "topic", "preferred_datetime", "meeting_format")

_NAME_BLOCKERS = (
    "консультац",
    "запис",
    "хочу",
    "нужн",
    "привет",
    "здравств",
    "онлайн",
    "офис",
    "телефон",
    "завтра",
    "сегодня",
    "тема",
    "вопрос",
)
_TOPIC_STOPWORDS = {
    "да",
    "нет",
    "ок",
    "хорошо",
    "не знаю",
    "потом",
    "любое",
    "неважно",
}
_DAY_TOKENS = (
    "сегодня",
    "завтра",
    "послезавтра",
    "понедельник",
    "вторник",
    "среда",
    "среду",
    "четверг",
    "пятниц",
    "суббот",
    "воскресен",
    "январ",
    "феврал",
    "март",
    "апрел",
    "мая",
    "июн",
    "июл",
    "август",
    "сентябр",
    "октябр",
    "ноябр",
    "декабр",
)
_TIME_TOKENS = ("утром", "днём", "днем", "вечером", "ночью", "после обеда", "в обед")
_PAST_TOKENS = ("вчера", "позавчера", "прошл")
_FORMAT_ALIASES = {
    "онлайн": "онлайн",
    "online": "онлайн",
    "видео": "онлайн",
    "zoom": "онлайн",
    "в офисе": "в офисе",
    "офис": "в офисе",
    "очно": "в офисе",
    "лично": "в офисе",
    "по телефону": "по телефону",
    "телефон": "по телефону",
    "звонок": "по телефону",
    "созвон": "по телефону",
}


def clean_space(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(str(value).split()).strip()
    return cleaned or None


def has_day(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"\b\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?\b", lowered):
        return True
    if re.search(r"через\s+\d+\s*дн", lowered):
        return True
    return any(token in lowered for token in _DAY_TOKENS)


def has_time(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"\b\d{1,2}[:.]\d{2}\b", lowered):
        return True
    if re.search(
        r"\bв\s+\d{1,2}(?:\s*час(?:а|ов)?)?(?!\s*(?:январ|феврал|март|апрел|ма[яй]|июн|июл|август|сентябр|октябр|ноябр|декабр))",
        lowered,
    ):
        return True
    if re.search(r"через\s+(?:полчаса|час|пару часов|\d+\s*(?:минут|час)(?!а?\s*дн))", lowered):
        return True
    return any(token in lowered for token in _TIME_TOKENS)


def is_past_datetime(text: str) -> bool:
    return any(token in text.lower() for token in _PAST_TOKENS)


def is_valid_name(value: str) -> bool:
    words = value.split()
    if not 1 <= len(words) <= 3:
        return False
    if any(char.isdigit() for char in value):
        return False
    lowered = value.lower()
    if any(token in lowered for token in _NAME_BLOCKERS):
        return False
    return all(re.fullmatch(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё-]{1,40}", word) for word in words)


def is_valid_contact(value: str) -> bool:
    if re.fullmatch(r"@[A-Za-z0-9_]{4,}", value):
        return True
    digits = re.sub(r"\D", "", value)
    return 10 <= len(digits) <= 15


def is_valid_topic(value: str) -> bool:
    if value.lower() in _TOPIC_STOPWORDS:
        return False
    if not re.search(r"[A-Za-zА-Яа-яЁё]", value):
        return False
    return len(value) >= 3


def is_valid_datetime(value: str) -> bool:
    if is_past_datetime(value):
        return False
    lowered = value.lower()
    if re.search(r"через\s+(?:полчаса|час|пару часов|\d+\s*(?:минут|час))", lowered):
        return True
    return has_day(value) and has_time(value)


def normalize_meeting_format(value: str | None) -> str | None:
    cleaned = clean_space(value)
    if cleaned is None:
        return None
    lowered = cleaned.lower().replace("ё", "е")
    if lowered in _FORMAT_ALIASES:
        return _FORMAT_ALIASES[lowered]
    if "онлайн" in lowered or "online" in lowered or "zoom" in lowered:
        return "онлайн"
    if "офис" in lowered or "очно" in lowered:
        return "в офисе"
    if "телефон" in lowered or "созвон" in lowered or "звонок" in lowered:
        return "по телефону"
    return None


_NAME_CHUNK = r"[A-Za-zА-Яа-яЁё-]{2,}(?:\s+[A-Za-zА-Яа-яЁё-]{2,}){0,2}"
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\-\s()]{8,20}\d)(?!\d)")


def extract_person_name(message: str) -> str | None:
    cleaned = clean_space(message)
    if cleaned is None:
        return None
    labeled = re.search(
        rf"(?:меня зовут|мо[её] имя)\s+({_NAME_CHUNK})",
        cleaned,
        flags=re.IGNORECASE,
    )
    if labeled:
        candidate = labeled.group(1)
    else:
        short = re.fullmatch(rf"(?:я)\s+({_NAME_CHUNK})", cleaned, flags=re.IGNORECASE)
        candidate = short.group(1) if short else cleaned
    candidate = candidate.strip(" .,!")
    if not is_valid_name(candidate):
        return None
    if candidate.islower():
        candidate = candidate.title()
    return candidate


def _normalize_phone_digits(digits: str) -> str | None:
    if len(digits) == 11 and digits.startswith("8"):
        digits = "7" + digits[1:]
    elif len(digits) == 10:
        digits = "7" + digits
    if not 11 <= len(digits) <= 15:
        return None
    return "+" + digits


def extract_contact_value(message: str) -> str | None:
    for match in _PHONE_RE.finditer(message):
        phone = _normalize_phone_digits(re.sub(r"\D", "", match.group(0)))
        if phone:
            return phone
    username = re.search(r"@[A-Za-z0-9_]{4,}", message)
    if username:
        return username.group(0)
    return None


class SupportTicket(BaseModel):
    name: str | None = None
    contact: str | None = None
    topic: str | None = None
    preferred_datetime: str | None = None
    meeting_format: str | None = None

    @field_validator("name", "contact", "topic", "preferred_datetime", "meeting_format", mode="before")
    @classmethod
    def clean_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return clean_space(str(value))

    def merge(self, other: "SupportTicket") -> None:
        for field_name, value in other.model_dump().items():
            if value not in (None, ""):
                setattr(self, field_name, value)

    def sanitize(self) -> list[str]:
        rejected: list[str] = []

        if self.name is not None and not is_valid_name(self.name):
            self.name = None
            rejected.append("name")

        if self.contact is not None and not is_valid_contact(self.contact):
            self.contact = None
            rejected.append("contact")

        if self.topic is not None and not is_valid_topic(self.topic):
            self.topic = None
            rejected.append("topic")

        if self.preferred_datetime is not None and not is_valid_datetime(self.preferred_datetime):
            self.preferred_datetime = None
            rejected.append("preferred_datetime")

        normalized_format = normalize_meeting_format(self.meeting_format)
        if self.meeting_format is not None and normalized_format is None:
            self.meeting_format = None
            rejected.append("meeting_format")
        else:
            self.meeting_format = normalized_format

        return [field_name for field_name in BOOKING_FIELDS if field_name in rejected]

    def is_complete(self) -> bool:
        return all(
            [
                self.name,
                self.contact,
                self.topic,
                self.preferred_datetime and is_valid_datetime(self.preferred_datetime),
                self.meeting_format in MEETING_FORMATS,
            ]
        )


class DialogueMessage(BaseModel):
    role: MessageRole
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        cleaned = " ".join(value.split()).strip()
        if not cleaned:
            raise ValueError("Dialogue message cannot be empty")
        return cleaned


class AssistantTurn(BaseModel):
    reply: str = Field(min_length=1)
    extracted_ticket: SupportTicket = Field(default_factory=SupportTicket)
    ready_to_submit: bool = False


class SupportSession(BaseModel):
    user_id: int
    chat_id: int
    telegram_username: str | None = None
    telegram_first_name: str | None = None
    started: bool = False
    submitted: bool = False
    ticket: SupportTicket = Field(default_factory=SupportTicket)
    history: list[DialogueMessage] = Field(default_factory=list)

    def add_user_message(self, text: str) -> None:
        self._append_history("user", text)

    def add_assistant_message(self, text: str) -> None:
        self._append_history("assistant", text)

    def recent_history(self, limit: int = 8) -> list[DialogueMessage]:
        return list(self.history[-limit:])

    @property
    def last_assistant_message(self) -> str | None:
        for message in reversed(self.history):
            if message.role == "assistant":
                return message.text
        return None

    def reset(self) -> None:
        self.started = False
        self.submitted = False
        self.ticket = SupportTicket()
        self.history = []

    def _append_history(self, role: MessageRole, text: str) -> None:
        self.history.append(DialogueMessage(role=role, text=text))
        if len(self.history) > 20:
            self.history = self.history[-20:]
