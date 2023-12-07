from typing import TypeVar

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


BaseClass = TypeVar("BaseClass", bound=Base)
