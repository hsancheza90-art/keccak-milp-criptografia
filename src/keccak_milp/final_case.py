"""
Parámetros dinámicos del caso final de Keccak reducido.

Los parámetros definidos en este módulo son públicos y seleccionan
configuraciones experimentales previamente validadas. No deben depender
de la clave ni de información secreta.
"""

from __future__ import annotations

from typing import Final, Literal, cast


SecurityLevel = Literal[0, 1, 2]
DomainId = Literal[0, 1, 2, 3]
WordSize = Literal[4, 8]


SUPPORTED_SECURITY_LEVELS: Final[tuple[int, ...]] = (
    0,
    1,
    2,
)

SUPPORTED_DOMAIN_IDS: Final[tuple[int, ...]] = (
    0,
    1,
    2,
    3,
)

SUPPORTED_WORD_SIZES: Final[tuple[int, ...]] = (
    4,
    8,
)

SECURITY_LEVEL_ROUNDS: Final[dict[SecurityLevel, int]] = {
    0: 1,
    1: 2,
    2: 3,
}


def _validate_plain_integer(
    value: object,
    parameter_name: str,
) -> int:
    """
    Valida que un parámetro sea un entero y no un booleano.

    En Python, ``bool`` es una subclase de ``int``. Se rechaza de forma
    explícita para impedir que ``True`` y ``False`` se interpreten como
    configuraciones numéricas.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"El parámetro {parameter_name} debe ser un entero."
        )

    return value


def validate_security_level(
    security_level: object,
) -> SecurityLevel:
    """
    Valida el nivel público de seguridad experimental.

    Los valores admitidos son:

    - 0: perfil ligero.
    - 1: perfil equilibrado.
    - 2: perfil reforzado.
    """

    validated = _validate_plain_integer(
        security_level,
        "security_level",
    )

    if validated not in SUPPORTED_SECURITY_LEVELS:
        raise ValueError(
            "security_level debe ser 0, 1 o 2. "
            f"Se recibió security_level={validated}."
        )

    return cast(SecurityLevel, validated)


def validate_domain_id(
    domain_id: object,
) -> DomainId:
    """
    Valida el identificador público del dominio de aplicación.

    Los valores admitidos son:

    - 0: hash general.
    - 1: autenticación.
    - 2: derivación de claves.
    - 3: integridad de firmware.
    """

    validated = _validate_plain_integer(
        domain_id,
        "domain_id",
    )

    if validated not in SUPPORTED_DOMAIN_IDS:
        raise ValueError(
            "domain_id debe ser 0, 1, 2 o 3. "
            f"Se recibió domain_id={validated}."
        )

    return cast(DomainId, validated)


def _validate_word_size(
    z: object,
) -> WordSize:
    """Valida los tamaños de palabra incluidos en el experimento."""

    validated = _validate_plain_integer(
        z,
        "z",
    )

    if validated not in SUPPORTED_WORD_SIZES:
        raise ValueError(
            "El tamaño de palabra z debe ser 4 u 8. "
            f"Se recibió z={validated}."
        )

    return cast(WordSize, validated)


def security_level_to_rounds(
    security_level: object,
) -> int:
    """
    Traduce el nivel público de seguridad al número de rondas.

    Esta correspondencia pertenece al experimento reducido y no
    representa una recomendación de seguridad para Keccak-f[1600].
    """

    validated = validate_security_level(
        security_level
    )

    return SECURITY_LEVEL_ROUNDS[validated]


def encode_dynamic_parameters(
    security_level: object,
    domain_id: object,
    z: object,
) -> int:
    """
    Codifica los dos parámetros públicos en una constante no nula.

    La codificación utilizada es:

        E(s, d) = 4 * (s + 1) + d

    con:

        s en {0, 1, 2}
        d en {0, 1, 2, 3}

    Se obtienen doce valores distintos en el intervalo [4, 15].
    Todos caben en los tamaños de palabra z = 4 y z = 8.
    """

    validated_security_level = validate_security_level(
        security_level
    )
    validated_domain_id = validate_domain_id(
        domain_id
    )
    validated_z = _validate_word_size(z)

    encoded = (
        4 * (validated_security_level + 1)
        + validated_domain_id
    )

    if encoded >= (1 << validated_z):
        raise RuntimeError(
            "La codificación dinámica no cabe en el lane."
        )

    return encoded


__all__ = [
    "DomainId",
    "SECURITY_LEVEL_ROUNDS",
    "SUPPORTED_DOMAIN_IDS",
    "SUPPORTED_SECURITY_LEVELS",
    "SUPPORTED_WORD_SIZES",
    "SecurityLevel",
    "WordSize",
    "encode_dynamic_parameters",
    "security_level_to_rounds",
    "validate_domain_id",
    "validate_security_level",
]
