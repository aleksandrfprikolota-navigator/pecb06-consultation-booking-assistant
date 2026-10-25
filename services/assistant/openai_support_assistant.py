import asyncio
import json
import logging

import httpx

from core import AssistantTurn, Settings, SupportTicket
from core.schemas import (
    DialogueMessage,
    clean_space,
    extract_contact_value,
    extract_person_name,
    is_valid_datetime,
    is_valid_topic,
    normalize_meeting_format,
)
from services.assistant.prompts import (
    ASSISTANT_RESPONSE_SCHEMA,
    GREETING,
    SUPPORT_ASSISTANT_PROMPT,
    correction_reply,
    next_question,
)

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class OpenAISupportAssistant:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._omit_temperature = False
        self._client = httpx.AsyncClient(
            base_url=settings.openai_base_url,
            timeout=httpx.Timeout(45.0, connect=12.0),
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
        )

    async def generate_turn(
        self,
        current_ticket: SupportTicket,
        user_message: str,
        is_new_session: bool,
        conversation_history: list[DialogueMessage],
        last_assistant_message: str | None,
        telegram_first_name: str | None,
    ) -> AssistantTurn:
        payload = {
            "model": self._settings.openai_model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SUPPORT_ASSISTANT_PROMPT},
                {
                    "role": "user",
                    "content": self._build_user_prompt(
                        current_ticket=current_ticket,
                        user_message=user_message,
                        is_new_session=is_new_session,
                        conversation_history=conversation_history,
                        last_assistant_message=last_assistant_message,
                        telegram_first_name=telegram_first_name,
                    ),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": ASSISTANT_RESPONSE_SCHEMA,
            },
        }

        try:
            response = await self._post_allowing_default_temperature(payload)
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return AssistantTurn.model_validate(self._load_json_object(content))
        except Exception:
            logger.exception("Falling back to local consultation turn generation")
            return self._build_fallback_turn(
                current_ticket=current_ticket,
                user_message=user_message,
                is_new_session=is_new_session,
                conversation_history=conversation_history,
                last_assistant_message=last_assistant_message,
                telegram_first_name=telegram_first_name,
            )

    async def close(self) -> None:
        await self._client.aclose()

    async def _post_allowing_default_temperature(self, payload: dict) -> httpx.Response:
        request_payload = dict(payload)
        if self._omit_temperature:
            request_payload.pop("temperature", None)
        try:
            return await self._post_with_retries(request_payload)
        except httpx.HTTPStatusError as exc:
            if self._omit_temperature or exc.response.status_code != 400 or "temperature" not in exc.response.text.lower():
                raise
            self._omit_temperature = True
            logger.info("Model accepts only the default temperature; next requests omit temperature=0.2")
            request_payload.pop("temperature", None)
            return await self._post_with_retries(request_payload)

    @staticmethod
    def _load_json_object(content: str) -> dict:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        return json.loads(text)

    async def _post_with_retries(self, payload: dict) -> httpx.Response:
        last_error: Exception | None = None

        for attempt in range(1, 4):
            try:
                response = await self._client.post("/chat/completions", json=payload)
                if response.status_code in RETRYABLE_STATUS_CODES:
                    response.raise_for_status()
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status_code = exc.response.status_code
                if status_code not in RETRYABLE_STATUS_CODES or attempt == 3:
                    raise
            except httpx.RequestError as exc:
                last_error = exc
                if attempt == 3:
                    break

            await asyncio.sleep(0.75 * attempt)

        assert last_error is not None
        raise last_error

    @staticmethod
    def _build_user_prompt(
        current_ticket: SupportTicket,
        user_message: str,
        is_new_session: bool,
        conversation_history: list[DialogueMessage],
        last_assistant_message: str | None,
        telegram_first_name: str | None,
    ) -> str:
        ticket_json = json.dumps(current_ticket.model_dump(), ensure_ascii=False, indent=2)
        history_json = json.dumps(
            [message.model_dump() for message in conversation_history],
            ensure_ascii=False,
            indent=2,
        )
        first_name = telegram_first_name or "null"
        last_bot_message = last_assistant_message or "null"

        return (
            f"is_new_session: {str(is_new_session).lower()}\n"
            f"telegram_first_name: {first_name}\n"
            f"current_ticket:\n{ticket_json}\n\n"
            f"conversation_history:\n{history_json}\n\n"
            f"last_assistant_message:\n{last_bot_message}\n\n"
            f"latest_user_message:\n{user_message}"
        )

    def _build_fallback_turn(
        self,
        current_ticket: SupportTicket,
        user_message: str,
        is_new_session: bool,
        conversation_history: list[DialogueMessage],
        last_assistant_message: str | None,
        telegram_first_name: str | None,
    ) -> AssistantTurn:
        del telegram_first_name
        message = clean_space(user_message) or ""
        message_lower = message.lower()
        extracted = SupportTicket()
        requested_field = self._detect_requested_field(last_assistant_message)
        correction: str | None = None

        if self._is_repeat_question_request(message_lower):
            repeated_question = last_assistant_message or "Пока мы не дошли до следующего вопроса."
            return AssistantTurn(
                reply=f"Последний мой вопрос был таким: {repeated_question}",
                extracted_ticket=extracted,
                ready_to_submit=current_ticket.is_complete(),
            )

        if not current_ticket.name and (requested_field in {None, "name"} or "зовут" in message_lower):
            if requested_field == "name" or "зовут" in message_lower or self._looks_like_only_name(message):
                name = extract_person_name(message)
                if name:
                    extracted.name = name
                elif requested_field == "name":
                    correction = correction_reply("name")

        if not current_ticket.contact and requested_field != "name":
            contact = extract_contact_value(message)
            if contact:
                extracted.contact = contact
            elif requested_field == "contact":
                correction = correction or correction_reply("contact")

        if not current_ticket.topic and requested_field == "topic":
            if self._accept_topic(message):
                extracted.topic = message
            else:
                correction = correction or correction_reply("topic")

        if not current_ticket.preferred_datetime and (
            requested_field == "preferred_datetime" or (requested_field is None and len(message.split()) <= 8)
        ):
            if is_valid_datetime(message):
                extracted.preferred_datetime = message
            elif requested_field == "preferred_datetime":
                correction = correction or correction_reply("preferred_datetime", message)

        if not current_ticket.meeting_format and (
            requested_field == "meeting_format" or (requested_field is None and len(message.split()) <= 3)
        ):
            meeting_format = normalize_meeting_format(message)
            if meeting_format and (requested_field == "meeting_format" or message_lower == meeting_format or message_lower in {"очно", "офис", "online", "zoom", "созвон", "звонок"}):
                extracted.meeting_format = meeting_format
            elif requested_field == "meeting_format":
                correction = correction or correction_reply("meeting_format")

        rejected = extracted.sanitize()
        if rejected and correction is None:
            correction = correction_reply(rejected[0], message)

        merged_ticket = current_ticket.model_copy(deep=True)
        merged_ticket.merge(extracted)

        if correction:
            reply = correction
        elif not conversation_history and is_new_session and not merged_ticket.name:
            reply = GREETING
        else:
            reply = next_question(merged_ticket)

        return AssistantTurn(
            reply=reply,
            extracted_ticket=extracted,
            ready_to_submit=merged_ticket.is_complete() and correction is None,
        )

    @staticmethod
    def _accept_topic(message: str) -> bool:
        if not is_valid_topic(message):
            return False
        if extract_contact_value(message):
            return False
        if is_valid_datetime(message) and len(message.split()) <= 6:
            return False
        if message.lower() in {
            "онлайн",
            "online",
            "в офисе",
            "офис",
            "очно",
            "по телефону",
            "созвон",
            "звонок",
            "zoom",
        }:
            return False
        return True

    @staticmethod
    def _looks_like_only_name(message: str) -> bool:
        return extract_person_name(message) is not None and len(message.split()) <= 3

    @staticmethod
    def _is_repeat_question_request(message_lower: str) -> bool:
        triggers = [
            "какой был прошлый вопрос",
            "какой был предыдущий вопрос",
            "повтори вопрос",
            "повтори последний вопрос",
            "что ты спрашивал",
            "что вы спрашивали",
        ]
        return any(trigger in message_lower for trigger in triggers)

    @staticmethod
    def _detect_requested_field(last_assistant_message: str | None) -> str | None:
        if not last_assistant_message:
            return None

        message = last_assistant_message.lower()
        if any(
            phrase in message
            for phrase in [
                "один формат",
                "онлайн, в офисе",
                "в офисе или по телефону",
            ]
        ):
            return "meeting_format"
        if any(
            phrase in message
            for phrase in [
                "как вас зовут",
                "как к вам обращаться",
                "имя или имя и фамилию",
                "распознать имя",
            ]
        ):
            return "name"
        if any(
            phrase in message
            for phrase in [
                "телефон или telegram",
                "не короче 10",
                "telegram для связи",
                "telegram в формате",
            ]
        ):
            return "contact"
        if any(
            phrase in message
            for phrase in [
                "о чём хотите",
                "о чем хотите",
                "опишите тему",
            ]
        ):
            return "topic"
        if any(
            phrase in message
            for phrase in [
                "день и время",
                "напишите ещё время",
                "напишите еще время",
                "напишите ещё день",
                "напишите еще день",
                "дата уже прошла",
                "будущий день",
            ]
        ):
            return "preferred_datetime"
        return None
