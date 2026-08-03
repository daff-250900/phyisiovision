"""Autenticación de la aplicación.

PhysioVision enseña nombres de pacientes y su historial, así que exponerla sin
control de acceso no es una opción. Este módulo aporta lo mínimo para cerrarla:
usuarios con contraseña, verificados contra un resumen derivado, nunca contra la
contraseña en claro.

**Dónde viven los usuarios.** En la variable `PHYSIOVISION_USUARIOS` (una línea
por usuario, o separadas por comas) o en el archivo que indique
`PHYSIOVISION_USUARIOS_ARCHIVO`, por defecto `data/usuarios.txt`. El formato de
cada línea es:

    usuario:pbkdf2_sha256$240000$<sal_hex>$<resumen_hex>

**Cómo se crea uno.** Sin escribir la contraseña en ningún archivo ni en el
historial de la terminal:

    python -m src.auth dafne

que pide la contraseña por teclado e imprime la línea que hay que pegar en
`.env` o en `data/usuarios.txt`.

**Qué más hace**: frena los intentos repetidos (cinco fallos cierran la puerta
un rato, y el rato crece si se insiste) y deja constancia de cada acceso en el
registro, acertado o no, sin escribir nunca la contraseña.

**Qué NO hace**: roles ni caducidad de contraseñas. Para uso clínico real hacen
falta las dos, y una auditoría persistente en vez de un registro a `stdout`.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import threading
import time
from pathlib import Path

from src.config import BASE_DIR

logger = logging.getLogger(__name__)

ALGORITMO = "pbkdf2_sha256"
#: Coste de derivación. Sube con los años; 240 000 es un valor razonable para
#: 2026 y mantiene la comprobación por debajo de la décima de segundo.
ITERACIONES = 240_000
LONGITUD_SAL = 16

#: Archivo por defecto. Está en `.gitignore`: una contraseña, aunque sea un
#: resumen, no pinta nada en el historial del repositorio.
ARCHIVO_POR_DEFECTO = BASE_DIR / "data" / "usuarios.txt"

#: Resumen de descarte, para gastar el mismo tiempo cuando el usuario no existe
#: que cuando existe. Sin esto, la diferencia de tiempos delata qué nombres son
#: válidos.
_RESUMEN_SENUELO = None

# --------------------------------------------------------------------------- #
# Freno a la fuerza bruta
#
# Una contraseña de ocho caracteres se agota en horas si se pueden probar miles
# por minuto. Con esto, cinco fallos cierran la puerta un rato, y el rato crece
# si se insiste: cinco minutos, diez, veinte… hasta una hora. El contador es por
# nombre de usuario y vive en memoria del proceso, que es donde vive también la
# sesión: reiniciar la app lo borra, y es un precio asumible frente a añadir una
# tabla para esto.
# --------------------------------------------------------------------------- #

MAX_INTENTOS = 5
VENTANA_S = 300.0
BLOQUEO_BASE_S = 300.0
BLOQUEO_MAXIMO_S = 3600.0

_fallos: dict[str, list[float]] = {}
_bloqueos: dict[str, tuple[float, int]] = {}      # usuario -> (hasta, veces)
_candado = threading.Lock()


def _segundos_de_bloqueo(usuario: str, ahora: float) -> float:
    """Segundos que le quedan de castigo, 0 si puede intentarlo."""
    hasta, _ = _bloqueos.get(usuario, (0.0, 0))
    return max(0.0, hasta - ahora)


def _anotar_fallo(usuario: str, ahora: float) -> None:
    recientes = [t for t in _fallos.get(usuario, []) if ahora - t < VENTANA_S]
    recientes.append(ahora)
    _fallos[usuario] = recientes
    if len(recientes) < MAX_INTENTOS:
        logger.warning("credenciales incorrectas para %r (%d/%d)",
                       usuario, len(recientes), MAX_INTENTOS)
        return

    _, veces = _bloqueos.get(usuario, (0.0, 0))
    espera = min(BLOQUEO_BASE_S * (2 ** veces), BLOQUEO_MAXIMO_S)
    _bloqueos[usuario] = (ahora + espera, veces + 1)
    _fallos[usuario] = []
    logger.warning("acceso bloqueado para %r durante %.0f s tras %d fallos",
                   usuario, espera, MAX_INTENTOS)


def _limpiar(usuario: str) -> None:
    _fallos.pop(usuario, None)
    _bloqueos.pop(usuario, None)


def reiniciar_intentos() -> None:
    """Olvida fallos y bloqueos. Para las pruebas y para desbloquear a mano."""
    with _candado:
        _fallos.clear()
        _bloqueos.clear()


def resumir(contrasena: str, sal: bytes | None = None,
            iteraciones: int = ITERACIONES) -> str:
    """Deriva el resumen de una contraseña. Formato `algoritmo$iter$sal$hash`."""
    sal = sal or secrets.token_bytes(LONGITUD_SAL)
    derivado = hashlib.pbkdf2_hmac("sha256", contrasena.encode("utf-8"), sal,
                                   iteraciones)
    return f"{ALGORITMO}${iteraciones}${sal.hex()}${derivado.hex()}"


def verificar(contrasena: str, resumen: str) -> bool:
    """`True` si la contraseña corresponde al resumen. No lanza nunca."""
    try:
        algoritmo, iteraciones, sal_hex, esperado = resumen.split("$")
        if algoritmo != ALGORITMO:
            return False
        candidato = hashlib.pbkdf2_hmac("sha256", contrasena.encode("utf-8"),
                                        bytes.fromhex(sal_hex), int(iteraciones))
    except (ValueError, TypeError):
        return False
    # compare_digest y no `==`: comparar cadena a cadena filtra por tiempo
    # cuántos caracteres iniciales se acertaron.
    return hmac.compare_digest(candidato.hex(), esperado)


def _lineas(texto: str) -> list[str]:
    crudas = texto.replace(",", "\n").splitlines()
    return [l.strip() for l in crudas if l.strip() and not l.strip().startswith("#")]


def cargar_usuarios(valor: str | None = None,
                    archivo: str | Path | None = None) -> dict[str, str]:
    """Usuarios configurados, como `{nombre: resumen}`.

    Precedencia: el argumento, luego `PHYSIOVISION_USUARIOS`, luego el archivo.
    Las entradas mal formadas se ignoran en silencio: una línea rota no puede
    tumbar el arranque de la aplicación, y admitirla sería peor.
    """
    if valor is None:
        valor = os.environ.get("PHYSIOVISION_USUARIOS", "")
    if not valor.strip():
        ruta = Path(archivo or os.environ.get("PHYSIOVISION_USUARIOS_ARCHIVO")
                    or ARCHIVO_POR_DEFECTO)
        valor = ruta.read_text(encoding="utf-8") if ruta.exists() else ""

    usuarios: dict[str, str] = {}
    for linea in _lineas(valor):
        nombre, separador, resumen = linea.partition(":")
        if separador and nombre.strip() and resumen.strip().startswith(ALGORITMO):
            usuarios[nombre.strip()] = resumen.strip()
    return usuarios


def hay_usuarios() -> bool:
    return bool(cargar_usuarios())


def autenticar(usuario: str, contrasena: str) -> bool:
    """Función que consume `demo.launch(auth=...)`.

    Además de comprobar la contraseña, frena los intentos repetidos y deja
    constancia en el registro de cada acceso, acertado o no. El registro es el
    primer escalón de la auditoría que pide una herramienta con datos de salud.
    """
    global _RESUMEN_SENUELO
    nombre = (usuario or "").strip()
    ahora = time.monotonic()

    with _candado:
        restante = _segundos_de_bloqueo(nombre, ahora)
    if restante > 0:
        logger.warning("intento de acceso de %r bloqueado, faltan %.0f s",
                       nombre, restante)
        return False

    resumen = cargar_usuarios().get(nombre)
    if resumen is None:
        if _RESUMEN_SENUELO is None:
            _RESUMEN_SENUELO = resumir("señuelo")
        verificar(contrasena or "", _RESUMEN_SENUELO)   # mismo coste
        with _candado:
            _anotar_fallo(nombre, ahora)
        return False

    if verificar(contrasena or "", resumen):
        with _candado:
            _limpiar(nombre)
        logger.info("acceso concedido a %r", nombre)
        return True

    with _candado:
        _anotar_fallo(nombre, ahora)
    return False


def exige_login(host: str) -> bool:
    """¿Hay que exigir credenciales para escuchar en `host`?

    En `localhost` no: es la máquina de quien desarrolla y pedirle usuario en
    cada arranque solo invita a desactivarlo. En cualquier otra dirección sí,
    porque ahí la app deja de ser privada.
    """
    return host not in ("127.0.0.1", "localhost", "::1")


def _cli() -> int:
    import getpass
    import sys

    nombre = sys.argv[1] if len(sys.argv) > 1 else input("Usuario: ").strip()
    if not nombre:
        print("Hace falta un nombre de usuario.")
        return 1
    contrasena = getpass.getpass("Contraseña: ")
    if len(contrasena) < 8:
        print("Usa al menos 8 caracteres.")
        return 1
    if contrasena != getpass.getpass("Repítela: "):
        print("No coinciden.")
        return 1

    print("\nPega esta línea en data/usuarios.txt (o en PHYSIOVISION_USUARIOS):\n")
    print(f"{nombre}:{resumir(contrasena)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
