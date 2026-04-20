"""Датаклассы и строковые константы для результата парсинга."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

MISSING = "отсутствует"
GARBAGE = "неразборчиво"


@dataclass
class FieldConfidence:
    """Уверенность по каждому полю.

    Шкала:
        1.0 — поле найдено в своём разделе и прошло валидацию
        0.7 — поле найдено в своём разделе, валидации нет
        0.5 — поле найдено эвристикой (не по разделу)
        0.0 — поле отсутствует или явный мусор
    """

    date: float = 0.0
    number: float = 0.0
    shipper: float = 0.0
    consignee: float = 0.0
    cargo: float = 0.0
    volume: float = 0.0
    driver: float = 0.0
    vehicle: float = 0.0
    reception: float = 0.0

    def overall(self) -> float:
        values = [
            self.date, self.number, self.shipper, self.consignee, self.cargo,
            self.volume, self.driver, self.vehicle, self.reception,
        ]
        return round(sum(values) / len(values), 2) if values else 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class ParsedRow:
    waybill: str = ""
    date: str = ""
    number: str = ""
    shipper: str = ""
    consignee: str = ""
    cargo: str = ""
    volume: str = ""
    driver: str = ""
    vehicle: str = ""
    reception: str = ""
    source: str = ""
    note: str = ""
    confidence: FieldConfidence = field(default_factory=FieldConfidence)

    def to_excel_tuple(self):
        return (
            self.waybill,
            self.date,
            self.number,
            self.shipper,
            self.consignee,
            self.cargo,
            self.volume,
            self.driver,
            self.vehicle,
            self.reception,
            self.source,
            self.note,
        )

    @classmethod
    def empty_missing(cls, source: str, note: str = "") -> ParsedRow:
        return cls(
            waybill="Транспортная накладная",
            date=MISSING,
            number=MISSING,
            shipper=MISSING,
            consignee=MISSING,
            cargo=MISSING,
            volume=MISSING,
            driver=MISSING,
            vehicle=MISSING,
            reception=MISSING,
            source=source,
            note=note,
        )

    def to_json_dict(self) -> dict:
        d = asdict(self)
        # confidence уже сериализуется через asdict
        return d

    @classmethod
    def from_json_dict(cls, d: dict) -> ParsedRow:
        conf = d.pop("confidence", None)
        row = cls(**d)
        if isinstance(conf, dict):
            row.confidence = FieldConfidence(**conf)
        return row
