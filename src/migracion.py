"""Traslado de los datos anteriores a los perfiles.

Antes de que existieran `usuarios` y `pacientes`, PhysioVision guardaba dos
cosas por separado y sin relación entre ellas:

- **Quién entraba**, en `data/usuarios.txt`: una línea por persona, sin perfil,
  porque solo entraba el profesional.
- **Qué se midió**, en la tabla `sessions`: el nombre del paciente escrito a
  mano en la cabecera, como texto suelto y repetido en cada serie.

Esta migración las une. Es **idempotente**: ejecutarla dos veces no duplica
nada, y por eso puede correr en cada arranque sin que nadie se acuerde de
lanzarla. Lo que hace, en orden:

1. Cada usuario del archivo pasa a la tabla `usuarios` con perfil
   `fisioterapeuta`. Es lo que eran: la aplicación la manejaba el profesional.
2. Cada nombre distinto de `sessions.patient_name` pasa a ser una ficha en
   `pacientes`, sin credenciales —nadie eligió una contraseña por ellos—.
3. Cada serie se ata a su ficha (`paciente_id`) y al profesional que se le
   asigne (`fisio_id`).

**A qué fisioterapeuta se asignan las fichas.** Si solo hay uno, a ese. Si hay
varios, al más antiguo, porque no hay ningún dato en la base que diga quién
atendió cada serie: esa columna no existía. Se avisa por pantalla y se reasigna
con:

    python -m src.migracion asignar "Dafne" --fisio otra_persona

Ejecutarla a mano, con informe:

    python -m src.migracion
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.storage import (PatientRepository, ROL_FISIO, SessionRepository,
                         UsuarioRepository)

logger = logging.getLogger(__name__)


@dataclass
class Informe:
    """Lo que la migración ha hecho. Vacío significa que no había nada que hacer."""

    usuarios_migrados: list[str] = field(default_factory=list)
    fichas_creadas: list[str] = field(default_factory=list)
    series_atadas: int = 0
    fisio_asignado: str | None = None
    avisos: list[str] = field(default_factory=list)

    @property
    def hubo_cambios(self) -> bool:
        return bool(self.usuarios_migrados or self.fichas_creadas
                    or self.series_atadas)

    def __str__(self) -> str:
        if not self.hubo_cambios and not self.avisos:
            return "Nada que migrar: la base ya está al día."
        lineas = []
        if self.usuarios_migrados:
            lineas.append(f"Usuarios pasados a la base como fisioterapeutas: "
                          f"{', '.join(self.usuarios_migrados)}")
        if self.fichas_creadas:
            lineas.append(f"Fichas de paciente creadas: "
                          f"{', '.join(self.fichas_creadas)}")
        if self.series_atadas:
            destino = f" (fisioterapeuta: {self.fisio_asignado})" if \
                self.fisio_asignado else ""
            lineas.append(f"Series atadas a su ficha: {self.series_atadas}{destino}")
        lineas.extend(f"Aviso: {a}" for a in self.avisos)
        return "\n".join(lineas)


def _fisio_por_defecto(usuarios: UsuarioRepository,
                       informe: Informe) -> dict | None:
    """A quién se le asignan las fichas huérfanas.

    Se prefiere una cuenta **venida del archivo heredado**: son las que
    existían cuando se grabaron esas series, así que es lo más parecido a un
    dato y no a una suposición. Entre varias, o si no hay ninguna, la más
    antigua.

    En cualquier caso se dice en voz alta cuando hay más de un candidato: la
    base antigua no guardaba quién atendió cada serie —esa columna no existía—,
    así que cualquier reparto fino sería inventado.
    """
    fisios = sorted(usuarios.listar(ROL_FISIO), key=lambda c: c["id"])
    del_archivo = [c for c in fisios if c["usuario"] in informe.usuarios_migrados]
    fisios = del_archivo + [c for c in fisios if c not in del_archivo]
    if not fisios:
        informe.avisos.append(
            "no hay ningún fisioterapeuta en la base, así que las fichas quedan "
            "sin asignar y no aparecerán en la vista de Pacientes. Crea uno con "
            "`python -m src.auth crear <usuario> --rol fisioterapeuta` y vuelve "
            "a ejecutar la migración")
        return None
    if len(fisios) > 1:
        informe.avisos.append(
            f"hay {len(fisios)} fisioterapeutas y la base antigua no guardaba "
            f"quién atendió cada serie: las fichas se asignan a "
            f"{fisios[0]['usuario']!r}. Reasigna con "
            f"`python -m src.migracion asignar \"<paciente>\" --fisio <usuario>`")
    return fisios[0]


def migrar_usuarios(usuarios: UsuarioRepository, informe: Informe) -> None:
    """Pasa `data/usuarios.txt` a la tabla, como fisioterapeutas."""
    from src.auth import usuarios_de_archivo

    for nombre, resumen in usuarios_de_archivo().items():
        if usuarios.obtener(nombre) is not None:
            continue
        usuarios.crear(nombre, resumen, ROL_FISIO)
        informe.usuarios_migrados.append(nombre)


def migrar_pacientes(sesiones: SessionRepository, pacientes: PatientRepository,
                     fisio_id: int | None, informe: Informe) -> None:
    """Crea una ficha por nombre distinto y ata cada serie a la suya.

    Se recorren las series **sin ficha**: las que ya la tienen se dejan como
    están, que es lo que hace la segunda ejecución idempotente.
    """
    antes = sesiones.contar_sin_ficha()
    for nombre in sesiones.nombres_sin_ficha():
        nombre = nombre.strip()
        if not nombre:
            informe.avisos.append(
                "hay series guardadas sin nombre de paciente; se quedan sin "
                "ficha, porque inventarles una sería peor que dejarlas visibles "
                "solo en la base")
            continue

        existente = pacientes.por_nombre(nombre, fisio_id)
        ficha = existente or pacientes.obtener_o_crear(nombre, fisio_id=fisio_id)
        if existente is None:
            informe.fichas_creadas.append(nombre)

        # `fisio_id` se escribe también en la serie: es quien la atendió, y
        # aunque mañana el paciente cambie de profesional, la serie no cambia
        # de manos.
        sesiones.atar_a_ficha(nombre, int(ficha["id"]), fisio_id)

    informe.series_atadas = antes - sesiones.contar_sin_ficha()


def migrar(db_path=None, url: str | None = None) -> Informe:
    """Ejecuta la migración completa. Idempotente y segura de repetir."""
    informe = Informe()
    usuarios = UsuarioRepository(db_path=db_path, url=url)
    pacientes = PatientRepository(db_path=db_path, url=url)
    sesiones = SessionRepository(db_path=db_path, url=url)

    migrar_usuarios(usuarios, informe)

    if sesiones.contar_sin_ficha() == 0:
        return informe

    fisio = _fisio_por_defecto(usuarios, informe)
    informe.fisio_asignado = fisio["usuario"] if fisio else None
    migrar_pacientes(sesiones, pacientes,
                     int(fisio["id"]) if fisio else None, informe)
    return informe


def migrar_en_arranque() -> Informe | None:
    """Migración silenciosa al arrancar la app. Nunca impide el arranque.

    Si algo falla —disco de solo lectura, base bloqueada por otro proceso— se
    anota en el registro y la aplicación sigue: quedarse sin arrancar por una
    migración sería peor que arrancar con los perfiles a medio poblar, porque
    el fisioterapeuta tiene un paciente esperando delante de la cámara.
    """
    try:
        informe = migrar()
    except Exception:
        logger.exception("la migración de perfiles no se pudo completar")
        return None
    if informe.hubo_cambios:
        logger.info("migración de perfiles: %s", str(informe).replace("\n", " | "))
    for aviso in informe.avisos:
        logger.warning("migración de perfiles: %s", aviso)
    return informe


#: Orden de copia: primero lo que otras tablas referencian.
TABLAS_EN_ORDEN = (
    ("usuarios", ("id", "usuario", "resumen", "rol", "nombre", "activo",
                  "creado_en")),
    ("pacientes", ("id", "nombre", "usuario_id", "fisio_id", "alta", "notas")),
    ("sessions", ("id", "patient_name", "exercise_id", "classification",
                  "confidence", "max_rom", "repetitions", "shoulder_angle",
                  "elbow_angle", "trunk_inclination", "movement_speed",
                  "created_at", "rom_medio", "correctas", "lado",
                  "cobertura_pose", "fuente", "resumen_json", "paciente_id",
                  "fisio_id")),
)


def copiar(destino: str, origen=None, forzar: bool = False) -> dict[str, int]:
    """Copia la base local a otra, **conservando los identificadores**.

    Es lo que hace falta al pasar de SQLite a Cloud SQL: el secreto de
    despliegue solo transporta `usuario:resumen`, así que reseminar desde él
    perdería los perfiles, las fichas y todo el historial. Aquí viaja la base
    entera.

    Los `id` se conservan a propósito: `pacientes.fisio_id` y
    `sessions.paciente_id` apuntan a ellos, y renumerar obligaría a reescribir
    las referencias, que es justo donde se cuelan los errores silenciosos.

    Args:
        destino: URL de PostgreSQL.
        origen: ruta del SQLite de partida. Por defecto, el de la aplicación.
        forzar: escribir aunque el destino ya tenga datos. Sin esto se niega,
            porque copiar sobre una base viva duplicaría o chocaría con lo que
            hubiera.

    Returns:
        `{tabla: filas copiadas}`.
    """
    lector = SessionRepository(db_path=origen) if origen else SessionRepository()
    escritor = SessionRepository(url=destino)
    if not escritor.postgres:
        raise ValueError("El destino tiene que ser una URL de PostgreSQL.")

    # Las tres tablas del destino se crean solas al instanciar los repositorios.
    UsuarioRepository(url=destino)
    PatientRepository(url=destino)

    existentes = {t: len(escritor._consultar(f"SELECT id FROM {t}"))
                  for t, _ in TABLAS_EN_ORDEN}
    if any(existentes.values()) and not forzar:
        raise RuntimeError(
            f"El destino ya tiene datos ({existentes}). Copiar encima "
            "duplicaría filas o chocaría con los identificadores. Usa "
            "forzar=True solo si sabes que se puede vaciar.")

    copiadas: dict[str, int] = {}
    with escritor._connect() as conexion:
        if forzar:
            for tabla, _ in reversed(TABLAS_EN_ORDEN):
                conexion.execute(f"DELETE FROM {tabla}")

        for tabla, columnas in TABLAS_EN_ORDEN:
            filas = lector._consultar(
                f"SELECT {', '.join(columnas)} FROM {tabla} ORDER BY id")
            marcadores = ", ".join(["%s"] * len(columnas))
            for fila in filas:
                conexion.execute(
                    f"INSERT INTO {tabla} ({', '.join(columnas)}) "
                    f"VALUES ({marcadores})",
                    tuple(fila[c] for c in columnas))
            copiadas[tabla] = len(filas)

            # Al insertar los `id` a mano, la secuencia de PostgreSQL se queda
            # en 1 y el primer INSERT posterior chocaría con la fila 1. Esto la
            # coloca detrás del último identificador copiado.
            conexion.execute(
                f"SELECT setval(pg_get_serial_sequence('{tabla}', 'id'), "
                f"COALESCE((SELECT MAX(id) FROM {tabla}), 1))")
        conexion.commit()
    return copiadas


def _cli() -> int:
    import argparse
    import sys

    analizador = argparse.ArgumentParser(prog="python -m src.migracion")
    ordenes = analizador.add_subparsers(dest="orden")

    asignar = ordenes.add_parser(
        "asignar", help="cambia el fisioterapeuta de una ficha de paciente")
    asignar.add_argument("paciente")
    asignar.add_argument("--fisio", required=True)

    ordenes.add_parser("estado", help="qué hay en la base ahora mismo")

    copia = ordenes.add_parser(
        "copiar", help="lleva la base local a PostgreSQL (Cloud SQL) tal cual")
    copia.add_argument("--a", required=True, metavar="URL",
                       help="postgresql://usuario:clave@host/base")
    copia.add_argument("--desde", metavar="RUTA",
                       help="SQLite de partida; por defecto, el de la app")
    copia.add_argument("--forzar", action="store_true",
                       help="vaciar el destino antes de copiar")

    opciones = analizador.parse_args(sys.argv[1:])

    if opciones.orden == "copiar":
        try:
            copiadas = copiar(opciones.a, origen=opciones.desde,
                              forzar=opciones.forzar)
        except (RuntimeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 1
        for tabla, n in copiadas.items():
            print(f"  {tabla:10s} {n:4d} filas copiadas")
        print("\nComprueba el resultado con la URL de destino:")
        print("  PHYSIOVISION_BD='<url>' python -m src.migracion estado")
        return 0

    if opciones.orden == "asignar":
        usuarios, pacientes = UsuarioRepository(), PatientRepository()
        cuenta = usuarios.obtener(opciones.fisio)
        if cuenta is None or cuenta["rol"] != ROL_FISIO:
            print(f"No hay ningún fisioterapeuta llamado {opciones.fisio!r}.")
            return 1
        ficha = pacientes.por_nombre(opciones.paciente)
        if ficha is None:
            print(f"No hay ninguna ficha para {opciones.paciente!r}.")
            return 1
        pacientes.asignar_fisio(int(ficha["id"]), int(cuenta["id"]))
        # Las series ya guardadas siguen a su ficha.
        SessionRepository().reasignar_fisio(int(ficha["id"]), int(cuenta["id"]))
        print(f"{opciones.paciente!r} pasa a {opciones.fisio!r}.")
        return 0

    if opciones.orden == "estado":
        usuarios, pacientes = UsuarioRepository(), PatientRepository()
        sesiones = SessionRepository()
        print(f"usuarios  : {len(usuarios.listar())} "
              f"({len(usuarios.listar(ROL_FISIO))} fisioterapeutas)")
        for ficha in pacientes.listar():
            fisio = (usuarios.por_id(int(ficha["fisio_id"]))
                     if ficha["fisio_id"] else None)
            series = sesiones.contar_de_ficha(int(ficha["id"]))
            print(f"  paciente {ficha['nombre']!r}: {series} series, "
                  f"fisio {fisio['usuario'] if fisio else '(sin asignar)'}, "
                  f"acceso {'sí' if ficha['usuario_id'] else 'no'}")
        return 0

    print(migrar())
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
