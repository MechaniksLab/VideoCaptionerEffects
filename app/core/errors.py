from __future__ import annotations


class AppError(Exception):
    """Базовая ошибка приложения с дружелюбным сообщением для UI."""


class ConfigError(AppError):
    pass


class AuthenticationError(AppError):
    pass


class RateLimitError(AppError):
    pass


class NetworkError(AppError):
    pass


class ParsingError(AppError):
    pass


class ProcessingError(AppError):
    pass


def map_exception(exc: Exception) -> AppError:
    """Нормализует типовые ошибки провайдеров к единым ошибкам приложения."""
    msg = str(exc)
    lower = msg.lower()

    if "api key" in lower or "unauthorized" in lower or "authentication" in lower:
        return AuthenticationError(msg)
    if "rate limit" in lower or "quota" in lower or "too many requests" in lower:
        return RateLimitError(msg)
    if "timeout" in lower or "connection" in lower or "network" in lower:
        return NetworkError(msg)
    if "json" in lower or "parse" in lower:
        return ParsingError(msg)
    return ProcessingError(msg)
