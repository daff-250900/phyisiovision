"""Autenticación de la aplicación.

PhysioVision enseña nombres de pacientes y su historial, así que exponerla sin
control de acceso no es una opción. Este módulo aporta lo mínimo para cerrarla:
usuarios con contraseña, verificados contra un resumen derivado, nunca contra la
contraseña en claro.

**Dónde viven los usuarios.** En la tabla `usuarios` de la base de datos, con
su perfil: `fisioterapeuta` o `paciente`. Antes vivían en un archivo de texto y
ese camino sigue abierto, pero solo como **semilla**: `PHYSIOVISION_USUARIOS` o
el archivo que indique `PHYSIOVISION_USUARIOS_ARCHIVO` —por defecto
`data/usuarios.txt`— se leen si la tabla está vacía, para que una instalación
antigua arranque sin quedarse fuera. `src/migracion.py` los pasa a la base. El
formato de cada línea es:

    usuario:pbkdf2_sha256$240000$<sal_hex>$<resumen_hex>

**Cómo se crea uno.** Sin escribir la contraseña en ningún archivo ni en el
historial de la terminal:

    python -m src.auth crear dafne --rol fisioterapeuta
    python -m src.auth crear maria --rol paciente --paciente "María Gómez"

que piden la contraseña por teclado y dan de alta la cuenta en la base.

**Qué más hace**: frena los intentos repetidos (cinco fallos cierran la puerta
un rato, y el rato crece si se insiste) y deja constancia de cada acceso en el
registro, acertado o no, sin escribir nunca la contraseña.

**Qué NO hace**: caducidad de contraseñas ni segundo factor. Para uso clínico
real hacen falta ambas, y una auditoría persistente en vez de un registro a
`stdout`.
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


def usuarios_de_archivo(valor: str | None = None,
                        archivo: str | Path | None = None) -> dict[str, str]:
    """Usuarios del archivo o de la variable de entorno, `{nombre: resumen}`.

    Precedencia: el argumento, luego `PHYSIOVISION_USUARIOS`, luego el archivo.
    Las entradas mal formadas se ignoran en silencio: una línea rota no puede
    tumbar el arranque de la aplicación, y admitirla sería peor.

    Esta es la fuente **heredada**. La actual es la tabla `usuarios`; esta se
    conserva como semilla para migrar y para no dejar fuera a una instalación
    que aún no haya migrado.
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


def _repositorio():
    """Repositorio de usuarios, o `None` si la base no está disponible.

    No se cachea a propósito: las pruebas cambian `PHYSIOVISION_BD` y la ruta
    de la base entre casos, y un repositorio guardado en una global las haría
    escribir todas en la primera que se hubiera abierto.
    """
    try:
        from src.storage import UsuarioRepository

        return UsuarioRepository()
    except Exception:                      # base bloqueada, disco de solo lectura…
        logger.exception("no se pudo abrir la tabla de usuarios")
        return None


def cargar_usuarios(valor: str | None = None,
                    archivo: str | Path | None = None) -> dict[str, str]:
    """Usuarios que pueden entrar, como `{nombre: resumen}`.

    Se **unen las dos fuentes**, y no es un arreglo: son dos cosas distintas.
    `PHYSIOVISION_USUARIOS` y el archivo son configuración del despliegue —en
    Cloud Run llegan como secreto, y ahí no hay disco donde guardar una base—,
    mientras que la tabla `usuarios` es lo que se da de alta desde la propia
    aplicación. Quedarse solo con una dejaría fuera a alguien: si mandara la
    base, un despliegue configurado por secreto no podría entrar; si mandara el
    archivo, las cuentas creadas con `python -m src.auth crear` no valdrían.

    En caso de coincidencia de nombre manda la base, que es donde aterriza un
    cambio de contraseña.

    Los argumentos fuerzan la fuente heredada, y existen para las pruebas y
    para `src/migracion.py`.
    """
    if valor is not None or archivo is not None:
        return usuarios_de_archivo(valor, archivo)

    usuarios = usuarios_de_archivo()
    repositorio = _repositorio()
    if repositorio is not None:
        usuarios.update(repositorio.resumenes())
    return usuarios


def perfil(usuario: str) -> dict[str, object] | None:
    """Ficha de la cuenta: usuario, rol, nombre y ficha de paciente si la hay.

    Es lo que la interfaz consulta en cada petición para decidir qué enseñar.
    Devuelve `None` si la cuenta no existe.

    Un usuario que solo esté en el archivo heredado —todavía sin migrar— se
    considera **fisioterapeuta**: es lo que era antes de que hubiera perfiles,
    porque quien manejaba la aplicación era el profesional.
    """
    from src.storage import ROL_FISIO

    nombre = (usuario or "").strip()
    if not nombre:
        return None

    repositorio = _repositorio()
    if repositorio is not None:
        cuenta = repositorio.obtener(nombre)
        if cuenta is not None:
            ficha = None
            if cuenta["rol"] != ROL_FISIO:
                from src.storage import PatientRepository

                ficha = PatientRepository().por_usuario(int(cuenta["id"]))
            return {"id": int(cuenta["id"]), "usuario": cuenta["usuario"],
                    "rol": cuenta["rol"], "nombre": cuenta["nombre"],
                    "paciente_id": ficha["id"] if ficha else None,
                    "paciente_nombre": ficha["nombre"] if ficha else None}

    if nombre in usuarios_de_archivo():
        return {"id": None, "usuario": nombre, "rol": ROL_FISIO, "nombre": nombre,
                "paciente_id": None, "paciente_nombre": None}
    return None


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


def _pedir_contrasena() -> str | None:
    import getpass

    contrasena = getpass.getpass("Contraseña: ")
    if len(contrasena) < 8:
        print("Usa al menos 8 caracteres.")
        return None
    if contrasena != getpass.getpass("Repítela: "):
        print("No coinciden.")
        return None
    return contrasena


def _cli() -> int:
    """Alta y listado de cuentas.

        python -m src.auth crear <usuario> --rol fisioterapeuta
        python -m src.auth crear <usuario> --rol paciente --paciente "María Gómez"
        python -m src.auth listar

    La forma antigua —`python -m src.auth <usuario>`, que imprimía una línea
    para pegar en el archivo— se conserva porque es la que está escrita en el
    README y en la documentación del proyecto.
    """
    import argparse
    import sys

    from src.storage import PatientRepository, ROL_FISIO, ROL_PACIENTE, \
        UsuarioRepository

    analizador = argparse.ArgumentParser(prog="python -m src.auth")
    ordenes = analizador.add_subparsers(dest="orden")

    crear = ordenes.add_parser("crear", help="da de alta una cuenta en la base")
    crear.add_argument("usuario")
    crear.add_argument("--rol", choices=[ROL_FISIO, ROL_PACIENTE], default=ROL_FISIO)
    crear.add_argument("--nombre", help="nombre para mostrar; por defecto, el usuario")
    crear.add_argument("--paciente", help="ficha de paciente a la que se asocia "
                                          "la cuenta (solo con --rol paciente)")
    crear.add_argument("--fisio", help="usuario del fisioterapeuta que lo atiende "
                                       "(solo con --rol paciente)")

    ordenes.add_parser("listar", help="cuentas activas y su perfil")
    ordenes.add_parser(
        "exportar", help="vuelca las cuentas en el formato de PHYSIOVISION_USUARIOS")

    resumen_cmd = ordenes.add_parser(
        "resumen", help="imprime la línea para data/usuarios.txt (forma antigua)")
    resumen_cmd.add_argument("usuario")

    # Forma antigua: `python -m src.auth dafne` sin subcomando.
    argumentos = sys.argv[1:]
    if argumentos and argumentos[0] not in ("crear", "listar", "resumen",
                                            "exportar", "-h", "--help"):
        argumentos = ["resumen", *argumentos]
    opciones = analizador.parse_args(argumentos)

    if opciones.orden == "exportar":
        # Semilla para un despliegue sin disco persistente: se vuelcan **las dos
        # fuentes**, base y archivo, porque es lo mismo que admite el login. Sale
        # el resumen derivado, nunca la contraseña.
        usuarios = cargar_usuarios()
        if not usuarios:
            print("No hay ninguna cuenta que exportar.", file=sys.stderr)
            return 1
        print(",".join(f"{nombre}:{resumen}"
                       for nombre, resumen in sorted(usuarios.items())))
        return 0

    if opciones.orden in (None, "listar"):
        usuarios = UsuarioRepository().listar()
        if not usuarios:
            print("No hay cuentas en la base. Crea una con:\n"
                  "  python -m src.auth crear <usuario> --rol fisioterapeuta")
            return 0
        print(f"{'usuario':20s} {'perfil':16s} nombre")
        for cuenta in usuarios:
            print(f"{cuenta['usuario']:20s} {cuenta['rol']:16s} {cuenta['nombre']}")
        return 0

    if opciones.orden == "resumen":
        contrasena = _pedir_contrasena()
        if contrasena is None:
            return 1
        print("\nPega esta línea en data/usuarios.txt "
              "(o en PHYSIOVISION_USUARIOS):\n")
        print(f"{opciones.usuario}:{resumir(contrasena)}")
        print("\nLo recomendable es dar de alta la cuenta en la base:\n"
              f"  python -m src.auth crear {opciones.usuario} --rol fisioterapeuta")
        return 0

    # crear
    usuarios = UsuarioRepository()
    if usuarios.obtener(opciones.usuario):
        print(f"Ya existe una cuenta activa para {opciones.usuario!r}.")
        return 1

    fisio_id = None
    if opciones.fisio:
        cuenta_fisio = usuarios.obtener(opciones.fisio)
        if cuenta_fisio is None or cuenta_fisio["rol"] != ROL_FISIO:
            print(f"No hay ningún fisioterapeuta llamado {opciones.fisio!r}.")
            return 1
        fisio_id = int(cuenta_fisio["id"])

    contrasena = _pedir_contrasena()
    if contrasena is None:
        return 1

    identificador = usuarios.crear(opciones.usuario, resumir(contrasena),
                                   opciones.rol, opciones.nombre)
    print(f"Alta de {opciones.usuario!r} como {opciones.rol}.")

    if opciones.rol == ROL_PACIENTE:
        pacientes = PatientRepository()
        nombre_ficha = opciones.paciente or opciones.nombre or opciones.usuario
        ficha = pacientes.obtener_o_crear(nombre_ficha, fisio_id=fisio_id)
        pacientes.asignar_usuario(int(ficha["id"]), identificador)
        if fisio_id is not None and ficha.get("fisio_id") is None:
            pacientes.asignar_fisio(int(ficha["id"]), fisio_id)
        print(f"Asociada a la ficha {nombre_ficha!r} (id {ficha['id']}).")
        if fisio_id is None and ficha.get("fisio_id") is None:
            print("Aviso: la ficha no tiene fisioterapeuta asignado, así que no "
                  "aparecerá en la vista de Pacientes de nadie. Asígnalo con "
                  "--fisio <usuario>.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
