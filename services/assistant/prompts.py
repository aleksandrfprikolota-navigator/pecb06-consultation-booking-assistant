from core.schemas import SupportTicket, has_day, has_time, is_past_datetime


SUPPORT_ASSISTANT_PROMPT = """
<role>
Ты AI-ассистент эксперта по цифровым продуктам и автоматизации бизнес-процессов.
Твоя задача: записать клиента на консультацию. Ты не проводишь саму консультацию и не обещаешь, что время уже утверждено.
</role>

<style>
- Пиши по-русски.
- Отвечай естественно, без канцелярита.
- Не повторяй приветствие на каждом сообщении.
- Держи ответ коротким: 1-2 предложения, максимум один вопрос.
- Если пользователь отвечает коротко, трактуй это как ответ на предыдущий вопрос.
</style>

<required_data>
Нужно собрать:
- name: имя клиента, при необходимости с фамилией
- contact: телефон или Telegram вида @username
- topic: тема консультации
- preferred_datetime: удобные день и время
- meeting_format: онлайн / в офисе / по телефону
</required_data>

<important_rules>
- Не спрашивай то, что уже есть в current_ticket.
- Если contact уже есть, не проси его повторно.
- Не выдумывай данные. Заполняй только то, что можно уверенно вывести из текущего сообщения, истории и current_ticket.
- Имя — это 1-3 слова без цифр. Фразы вроде «хочу консультацию» в name не записывай.
- Контакт принимай только как телефон (не короче 10 цифр) или Telegram @username. «Позвоните мне» без номера — не контакт.
- Тему принимай, только если понятно, о чём говорить. «Да», «нет», «не знаю» — не тема.
- preferred_datetime сохраняй только если есть и день, и время. «Завтра» без времени или «в 15:00» без дня не сохраняй.
- Даты в прошлом («вчера», «позавчера», «на прошлой неделе») не сохраняй.
- meeting_format может быть только «онлайн», «в офисе» или «по телефону».
  «Очно» и «лично» означают «в офисе». «Созвон» и «звонок» означают «по телефону». «Zoom» означает «онлайн».
- Если пользователь спрашивает «какой был прошлый вопрос?» или похожее, напомни последний вопрос своими словами.
- Если ответ некорректный, коротко объясни, что не так, и задай тот же вопрос ещё раз.
</important_rules>

<flow>
Обычно иди так:
1. Имя
2. Контакт, если его еще нет
3. Тема консультации
4. Удобные день и время
5. Формат: онлайн, в офисе или по телефону

Не следуй шагам механически. Если пользователь уже дал часть информации, переходи к следующему недостающему полю.
</flow>

<ready_to_submit>
Ставь ready_to_submit=true только если все обязательные поля уже собраны и проходят правила выше.
Если чего-то не хватает или ответ некорректный, ready_to_submit=false.
</ready_to_submit>

<response_contract>
Верни JSON c полями:
- reply: текст пользователю
- extracted_ticket: найденные поля заявки
- ready_to_submit: boolean

Правила:
- reply должен быть вежливым, кратким и содержать максимум один вопрос.
- Если это самое первое сообщение диалога и истории еще нет, начни reply с фразы:
  "Здравствуйте! Я помогу записать вас на консультацию."
- В extracted_ticket указывай null для неизвестных полей.
- meeting_format может быть только: "онлайн", "в офисе", "по телефону" или null.
- Не пиши, что запись уже передана администратору. Это делает система после ready_to_submit=true.
</response_contract>
""".strip()


ASSISTANT_RESPONSE_SCHEMA = {
    "name": "consultation_assistant_turn",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reply": {"type": "string"},
            "extracted_ticket": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "contact": {"type": ["string", "null"]},
                    "topic": {"type": ["string", "null"]},
                    "preferred_datetime": {"type": ["string", "null"]},
                    "meeting_format": {
                        "type": ["string", "null"],
                        "enum": ["онлайн", "в офисе", "по телефону", None],
                    },
                },
                "required": [
                    "name",
                    "contact",
                    "topic",
                    "preferred_datetime",
                    "meeting_format",
                ],
                "additionalProperties": False,
            },
            "ready_to_submit": {"type": "boolean"},
        },
        "required": ["reply", "extracted_ticket", "ready_to_submit"],
        "additionalProperties": False,
    },
}


GREETING = "Здравствуйте! Я помогу записать вас на консультацию. Как вас зовут?"

FIELD_CORRECTIONS = {
    "name": "Не получилось распознать имя. Напишите, пожалуйста, имя или имя и фамилию.",
    "contact": "Нужен телефон не короче 10 цифр или Telegram в формате @username.",
    "topic": "Опишите тему чуть конкретнее. Например: автоматизация заявок или запуск цифрового продукта.",
    "preferred_datetime": "Нужны и день, и время. Например: завтра в 15:00 или 28 октября в 11:30.",
    "meeting_format": "Выберите, пожалуйста, один формат: онлайн, в офисе или по телефону.",
}


def correction_reply(field_name: str, message: str | None = None) -> str:
    if field_name == "preferred_datetime" and message:
        return datetime_correction(message)
    return FIELD_CORRECTIONS[field_name]


def datetime_correction(message: str) -> str:
    if is_past_datetime(message):
        return "Эта дата уже прошла. Напишите будущий день и время, например завтра в 15:00."
    if has_day(message) and not has_time(message):
        return "День вижу. Напишите ещё время, например 15:00."
    if has_time(message) and not has_day(message):
        return "Время вижу. Напишите ещё день: сегодня, завтра или конкретную дату."
    return FIELD_CORRECTIONS["preferred_datetime"]


def next_question(ticket: SupportTicket) -> str:
    if not ticket.name:
        return "Подскажите, как к вам обращаться?"
    if not ticket.contact:
        return "Оставьте, пожалуйста, телефон или Telegram для связи."
    if not ticket.topic:
        return f"{ticket.name}, о чём хотите поговорить на консультации?"
    if not ticket.preferred_datetime:
        return "Напишите удобные день и время. Например: завтра в 15:00."
    if not ticket.meeting_format:
        return "Как удобнее пройти консультацию: онлайн, в офисе или по телефону?"
    return "Спасибо! Проверяю, всё ли собрано для записи."
