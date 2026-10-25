from core import SupportTicket


def format_collected_ticket(ticket: SupportTicket) -> str:
    lines = [
        f"Имя: {ticket.name or '-'}",
        f"Контакт: {ticket.contact or '-'}",
        f"Тема: {ticket.topic or '-'}",
        f"Дата и время: {ticket.preferred_datetime or '-'}",
        f"Формат: {ticket.meeting_format or '-'}",
    ]
    return "\n".join(lines)
