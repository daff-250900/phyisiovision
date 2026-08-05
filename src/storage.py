"""Persistencia de perfiles, pacientes y series terminadas.

Dos motores detrás de las mismas clases, y la elección la hace el entorno:

- **SQLite** (por defecto). Un archivo, cero configuración. Es lo que usan el
  equipo de desarrollo y el despliegue en la clínica, donde el disco es del
  centro y no se va a ninguna parte.
- **PostgreSQL**, cuando `PHYSIOVISION_BD` trae una URL de conexión. Hace falta
  allí donde el disco del contenedor es efímero —Cloud Run, sin ir más lejos—,
  porque SQLite sobre un bucket o un NFS no tiene bloqueo POSIX real y se
  corrompe al escribir.

Lo que **no** cambia entre motores es el esquema ni las consultas: se escriben
una vez con marcadores `?` y se traducen a `%s` para Postgres. Mantener dos
juegos de SQL sería la forma segura de que uno se quedara atrás.

    # local, sin nada que configurar
    SessionRepository()

    # Cloud SQL
    PHYSIOVISION_BD=postgresql://usuario:clave@/physiovision?host=/cloudsql/...

**Tres tablas y dos perfiles.** `usuarios` guarda a quien entra en la
aplicación con su rol —`fisioterapeuta` o `paciente`—; `pacientes` es la ficha
clínica, que existe tenga o no credenciales asociadas; y `sessions` es cada
serie terminada, atada ya a la ficha y a quien la atendió.

La separación entre `usuarios` y `pacientes` no es ceremonia: un paciente puede
tener ficha e historial mucho antes de que alguien le dé acceso, y de hecho así
llegaron los que ya estaban en la base cuando esto se introdujo (ver
`src/migracion.py`).
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings

#: Los dos perfiles. No hay más, y la lista está aquí para que añadir uno sea
#: un cambio consciente y no una cadena suelta escrita en tres sitios.
ROL_FISIO = "fisioterapeuta"
ROL_PACIENTE = "paciente"
ROLES = (ROL_FISIO, ROL_PACIENTE)

#: Columnas que se añadieron después de la primera versión. La migración es
#: idempotente y se ejecuta al arrancar, así que actualizar la aplicación no
#: exige tocar la base a mano.
COLUMNAS_NUEVAS = (("rom_medio", "REAL"), ("correctas", "INTEGER"),
                   ("lado", "TEXT"), ("cobertura_pose", "REAL"),
                   ("fuente", "TEXT"), ("resumen_json", "TEXT"),
                   # Perfiles: quién hizo la serie y quién la atendió. Quedan
                   # nulas en las series anteriores hasta que `src/migracion.py`
                   # las rellena.
                   ("paciente_id", "INTEGER"), ("fisio_id", "INTEGER"))

#: `patient_name` se conserva a propósito aunque ya exista `paciente_id`: es lo
#: que se escribió el día de la serie. Si mañana se corrige un nombre mal
#: tecleado en la ficha, el historial antiguo sigue diciendo con qué nombre se
#: registró, que es lo que espera cualquiera que audite.
ESQUEMA = (
    """
    CREATE TABLE IF NOT EXISTS usuarios (
        id {serial},
        usuario TEXT NOT NULL,
        resumen TEXT NOT NULL,
        rol TEXT NOT NULL,
        nombre TEXT NOT NULL,
        activo INTEGER NOT NULL DEFAULT 1,
        creado_en TEXT NOT NULL
    )
    """,
    # El índice va sobre `lower(usuario)` porque el login no distingue
    # mayúsculas: sin esto, "Dafne" y "dafne" serían dos cuentas capaces de
    # entrar con la misma contraseña escrita distinta.
    """
    CREATE UNIQUE INDEX IF NOT EXISTS ix_usuarios_usuario
        ON usuarios (lower(usuario))
    """,
    """
    CREATE TABLE IF NOT EXISTS pacientes (
        id {serial},
        nombre TEXT NOT NULL,
        usuario_id INTEGER,
        fisio_id INTEGER,
        alta TEXT NOT NULL,
        notas TEXT
    )
    """,
    # Unicidad **por fisioterapeuta**, no global: dos consultas distintas
    # pueden tener cada una a su María García, y son dos personas.
    """
    CREATE UNIQUE INDEX IF NOT EXISTS ix_pacientes_nombre
        ON pacientes (lower(nombre), fisio_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id {serial},
        patient_name TEXT NOT NULL,
        exercise_id TEXT NOT NULL,
        classification TEXT NOT NULL,
        confidence {real} NOT NULL,
        max_rom {real} NOT NULL,
        repetitions INTEGER NOT NULL,
        shoulder_angle {real},
        elbow_angle {real},
        trunk_inclination {real},
        movement_speed {real},
        created_at TEXT NOT NULL
    )
    """,
)


def _es_postgres(url: str) -> bool:
    return url.startswith(("postgres://", "postgresql://"))


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


class _RepositorioBase:
    """Conexión, dialecto y esquema. No se instancia directamente.

    Las tres tablas se crean siempre, instancie quien instancie: son parte de
    la misma base y crearlas por separado abriría la puerta a arrancar con
    media estructura según qué vista se visitara primero.

    Args:
        db_path: ruta del archivo SQLite. Si se indica, **manda sobre
            `PHYSIOVISION_BD`**: es lo que usan las pruebas para trabajar sobre
            una base temporal aunque el entorno apunte a Postgres.
        url: URL de conexión de Postgres. Por defecto, `PHYSIOVISION_BD`.
    """

    def __init__(self, db_path: str | Path | None = None,
                 url: str | None = None) -> None:
        self.url = "" if db_path is not None else (
            url if url is not None else os.environ.get("PHYSIOVISION_BD", "")).strip()
        self.postgres = _es_postgres(self.url)

        if self.postgres:
            self.db_path = None
        else:
            self.db_path = Path(db_path or settings.database_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    # -- conexión ------------------------------------------------------------ #

    def _connect(self):
        if not self.postgres:
            conexion = sqlite3.connect(self.db_path)
            conexion.row_factory = sqlite3.Row
            return conexion

        # El import va aquí dentro: quien use SQLite —la mayoría— no tiene por
        # qué instalar el controlador de Postgres.
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self.url, row_factory=dict_row)

    def _sql(self, consulta: str) -> str:
        """Traduce los marcadores al dialecto del motor."""
        return consulta.replace("?", "%s") if self.postgres else consulta

    # -- esquema ------------------------------------------------------------- #

    def _columnas_existentes(self, conexion, tabla: str) -> set[str]:
        if self.postgres:
            filas = conexion.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = %s", (tabla,)).fetchall()
            return {f["column_name"] for f in filas}
        return {f["name"] for f in conexion.execute(f"PRAGMA table_info({tabla})")}

    def _create_tables(self) -> None:
        formato = {
            "serial": "SERIAL PRIMARY KEY" if self.postgres
                      else "INTEGER PRIMARY KEY AUTOINCREMENT",
            "real": "DOUBLE PRECISION" if self.postgres else "REAL",
        }
        with self._connect() as conexion:
            for sentencia in ESQUEMA:
                conexion.execute(sentencia.format(**formato))

            # Migración idempotente de columnas. Las cuatro columnas angulares
            # del esquema base pertenecen al contrato antiguo de 4 variables; el
            # modelo actual usa 26 y el conjunto puede volver a cambiar al
            # reentrenar. Guardar el resumen como JSON evita migrar el esquema
            # en cada reentreno.
            existentes = self._columnas_existentes(conexion, "sessions")
            for columna, tipo in COLUMNAS_NUEVAS:
                if columna not in existentes:
                    if tipo == "REAL" and self.postgres:
                        tipo = "DOUBLE PRECISION"
                    conexion.execute(
                        f"ALTER TABLE sessions ADD COLUMN {columna} {tipo}")
            if self.postgres:
                conexion.commit()

    # -- utilidades ---------------------------------------------------------- #

    def _consultar(self, consulta: str, valores: tuple = ()) -> list[dict[str, Any]]:
        with self._connect() as conexion:
            filas = conexion.execute(self._sql(consulta), valores).fetchall()
        return [dict(fila) for fila in filas]

    def _uno(self, consulta: str, valores: tuple = ()) -> dict[str, Any] | None:
        filas = self._consultar(consulta, valores)
        return filas[0] if filas else None

    def _insertar(self, consulta: str, valores: tuple) -> int:
        with self._connect() as conexion:
            if self.postgres:
                fila = conexion.execute(
                    self._sql(consulta + " RETURNING id"), valores).fetchone()
                conexion.commit()
                return int(fila["id"])
            cursor = conexion.execute(consulta, valores)
            return int(cursor.lastrowid)

    def _ejecutar(self, consulta: str, valores: tuple = ()) -> None:
        with self._connect() as conexion:
            conexion.execute(self._sql(consulta), valores)
            if self.postgres:
                conexion.commit()


class UsuarioRepository(_RepositorioBase):
    """Quién puede entrar en la aplicación y con qué perfil.

    Guarda el **resumen** de la contraseña, nunca la contraseña: derivarlo es
    cosa de `src/auth.py`, que es quien sabe de PBKDF2. Aquí solo se almacena y
    se devuelve la cadena.
    """

    def crear(self, usuario: str, resumen: str, rol: str,
              nombre: str | None = None) -> int:
        """Da de alta una cuenta. Lanza `ValueError` si el rol no existe."""
        if rol not in ROLES:
            raise ValueError(f"Rol desconocido: {rol!r}. Usa uno de {ROLES}.")
        usuario = usuario.strip()
        if not usuario:
            raise ValueError("El nombre de usuario no puede estar vacío.")
        return self._insertar(
            "INSERT INTO usuarios (usuario, resumen, rol, nombre, activo, creado_en) "
            "VALUES (?, ?, ?, ?, 1, ?)",
            (usuario, resumen, rol, (nombre or usuario).strip(), _ahora()))

    def obtener(self, usuario: str) -> dict[str, Any] | None:
        """Cuenta activa con ese nombre, sin distinguir mayúsculas."""
        return self._uno(
            "SELECT * FROM usuarios WHERE lower(usuario) = lower(?) AND activo = 1",
            (usuario.strip(),))

    def por_id(self, usuario_id: int) -> dict[str, Any] | None:
        return self._uno("SELECT * FROM usuarios WHERE id = ?", (usuario_id,))

    def listar(self, rol: str | None = None) -> list[dict[str, Any]]:
        if rol is None:
            return self._consultar(
                "SELECT * FROM usuarios WHERE activo = 1 ORDER BY lower(usuario)")
        return self._consultar(
            "SELECT * FROM usuarios WHERE activo = 1 AND rol = ? "
            "ORDER BY lower(usuario)", (rol,))

    def resumenes(self) -> dict[str, str]:
        """`{usuario: resumen}` de las cuentas activas, como espera el login."""
        return {f["usuario"]: f["resumen"] for f in self.listar()}

    def cambiar_clave(self, usuario: str, resumen: str) -> None:
        self._ejecutar(
            "UPDATE usuarios SET resumen = ? WHERE lower(usuario) = lower(?)",
            (resumen, usuario.strip()))

    def desactivar(self, usuario: str) -> None:
        """Baja lógica. No se borra: las series ya guardadas lo referencian."""
        self._ejecutar(
            "UPDATE usuarios SET activo = 0 WHERE lower(usuario) = lower(?)",
            (usuario.strip(),))


class PatientRepository(_RepositorioBase):
    """Fichas de paciente, con o sin credenciales asociadas."""

    def crear(self, nombre: str, fisio_id: int | None = None,
              usuario_id: int | None = None, notas: str | None = None) -> int:
        nombre = nombre.strip()
        if not nombre:
            raise ValueError("El nombre del paciente no puede estar vacío.")
        return self._insertar(
            "INSERT INTO pacientes (nombre, usuario_id, fisio_id, alta, notas) "
            "VALUES (?, ?, ?, ?, ?)",
            (nombre, usuario_id, fisio_id, _ahora(), notas))

    def por_nombre(self, nombre: str,
                   fisio_id: int | None = None) -> dict[str, Any] | None:
        """Ficha por nombre. Con `fisio_id`, solo entre las de ese profesional.

        Sin `fisio_id` se devuelve la primera coincidencia: vale para el
        arranque local sin login, donde no hay a quién atribuir nada.
        """
        if fisio_id is None:
            return self._uno(
                "SELECT * FROM pacientes WHERE lower(nombre) = lower(?) "
                "ORDER BY id LIMIT 1", (nombre.strip(),))
        return self._uno(
            "SELECT * FROM pacientes WHERE lower(nombre) = lower(?) AND fisio_id = ?",
            (nombre.strip(), fisio_id))

    def por_id(self, paciente_id: int) -> dict[str, Any] | None:
        return self._uno("SELECT * FROM pacientes WHERE id = ?", (paciente_id,))

    def por_usuario(self, usuario_id: int) -> dict[str, Any] | None:
        """Ficha de la persona que ha entrado con perfil de paciente."""
        return self._uno("SELECT * FROM pacientes WHERE usuario_id = ?",
                         (usuario_id,))

    def obtener_o_crear(self, nombre: str,
                        fisio_id: int | None = None) -> dict[str, Any]:
        """La ficha de ese nombre, creándola si es la primera vez.

        Es lo que llama la sesión al guardar: escribir el nombre en la cabecera
        da de alta al paciente si no existía, que es como se trabajaba antes de
        que hubiera fichas y sigue siendo lo que espera quien usa la app.
        """
        existente = self.por_nombre(nombre, fisio_id)
        if existente:
            return existente
        self.crear(nombre, fisio_id=fisio_id)
        # Se relee en vez de construir el dict a mano: así el `alta` y el `id`
        # son los que quedaron en la base, no los que creemos que quedaron.
        return self.por_nombre(nombre, fisio_id)

    def asignar_usuario(self, paciente_id: int, usuario_id: int | None) -> None:
        self._ejecutar("UPDATE pacientes SET usuario_id = ? WHERE id = ?",
                       (usuario_id, paciente_id))

    def asignar_fisio(self, paciente_id: int, fisio_id: int | None) -> None:
        self._ejecutar("UPDATE pacientes SET fisio_id = ? WHERE id = ?",
                       (fisio_id, paciente_id))

    def listar(self, fisio_id: int | None = None) -> list[dict[str, Any]]:
        if fisio_id is None:
            return self._consultar("SELECT * FROM pacientes ORDER BY lower(nombre)")
        return self._consultar(
            "SELECT * FROM pacientes WHERE fisio_id = ? ORDER BY lower(nombre)",
            (fisio_id,))


class SessionRepository(_RepositorioBase):
    """Guarda y consulta las series de un paciente."""

    # -- escritura ----------------------------------------------------------- #

    def save_summary(self, resumen: dict[str, object],
                     paciente_id: int | None = None,
                     fisio_id: int | None = None) -> int:
        """Guarda el resumen de una serie devuelto por `SesionEnVivo.resumen()`.

        Args:
            resumen: lo que devuelve la sesión en vivo.
            paciente_id: ficha a la que se ata la serie. Puede faltar en el
                arranque local sin login, y entonces la serie se guarda igual:
                perder la medición por no saber a quién atribuirla sería el peor
                de los intercambios.
            fisio_id: quién atendió la serie.
        """
        consulta = """
            INSERT INTO sessions (
                patient_name, exercise_id, classification, confidence,
                max_rom, repetitions, rom_medio, correctas, lado,
                cobertura_pose, fuente, resumen_json, paciente_id, fisio_id,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        valores = (
            str(resumen.get("paciente", "")),
            str(resumen.get("ejercicio", "")),
            str(resumen.get("clasificacion") or "sin_datos"),
            float(resumen.get("confianza") or 0.0),
            float(resumen.get("rom_max") or 0.0),
            int(resumen.get("repeticiones") or 0),
            float(resumen.get("rom_medio") or 0.0),
            int(resumen.get("correctas") or 0),
            resumen.get("lado"),
            float(resumen.get("cobertura_pose") or 0.0),
            str(resumen.get("fuente", "desconocido")),
            json.dumps(resumen, ensure_ascii=False, default=str),
            paciente_id,
            fisio_id,
            _ahora(),
        )
        return self._insertar(consulta, valores)

    # -- vínculo con las fichas (lo usa src/migracion.py) -------------------- #

    def nombres_sin_ficha(self) -> list[str]:
        """Nombres distintos de series que todavía no apuntan a una ficha."""
        return [str(f["patient_name"]) for f in self._consultar(
            "SELECT DISTINCT patient_name FROM sessions WHERE paciente_id IS NULL")]

    def contar_sin_ficha(self) -> int:
        filas = self._consultar(
            "SELECT COUNT(*) AS n FROM sessions WHERE paciente_id IS NULL")
        return int(filas[0]["n"]) if filas else 0

    def contar_de_ficha(self, paciente_id: int) -> int:
        filas = self._consultar(
            "SELECT COUNT(*) AS n FROM sessions WHERE paciente_id = ?",
            (paciente_id,))
        return int(filas[0]["n"]) if filas else 0

    def atar_a_ficha(self, nombre: str, paciente_id: int,
                     fisio_id: int | None) -> None:
        """Ata a una ficha las series de ese nombre que aún no la tuvieran.

        Solo toca las que están sueltas: las ya atadas se dejan como están, que
        es lo que hace idempotente volver a ejecutar la migración.
        """
        self._ejecutar(
            "UPDATE sessions SET paciente_id = ?, fisio_id = ? "
            "WHERE paciente_id IS NULL AND lower(patient_name) = lower(?)",
            (paciente_id, fisio_id, nombre.strip()))

    def reasignar_fisio(self, paciente_id: int, fisio_id: int) -> None:
        """Pasa a otro profesional las series ya guardadas de una ficha."""
        self._ejecutar("UPDATE sessions SET fisio_id = ? WHERE paciente_id = ?",
                       (fisio_id, paciente_id))

    # -- lectura ------------------------------------------------------------- #

    def list_patients(self, limit: int = 200,
                      fisio_id: int | None = None) -> list[dict[str, object]]:
        """Un registro por paciente, con su actividad acumulada.

        Se agrupa por nombre en minúsculas porque es lo mismo que hace
        `get_patient_history` al buscar: si no, "Dafne" y "dafne" saldrían como
        dos personas en la lista y como una sola en el historial.

        Con `fisio_id` se limita a los pacientes de ese profesional, que es lo
        que ve la vista de Pacientes cuando hay login.
        """
        if fisio_id is None:
            return self._consultar(
                """
                SELECT MIN(patient_name) AS paciente,
                       COUNT(*) AS series,
                       MAX(created_at) AS ultima,
                       SUM(repetitions) AS repeticiones,
                       SUM(correctas) AS correctas
                FROM sessions
                GROUP BY lower(patient_name)
                ORDER BY ultima DESC
                LIMIT ?
                """,
                (limit,))
        # Se filtra por la ficha y no por `sessions.fisio_id`: si un compañero
        # cubre una baja y atiende una serie, esa serie sigue siendo del
        # historial del paciente, y el paciente sigue siendo de su fisio.
        return self._consultar(
            """
            SELECT MIN(s.patient_name) AS paciente,
                   COUNT(*) AS series,
                   MAX(s.created_at) AS ultima,
                   SUM(s.repetitions) AS repeticiones,
                   SUM(s.correctas) AS correctas
            FROM sessions s
            JOIN pacientes p ON p.id = s.paciente_id
            WHERE p.fisio_id = ?
            GROUP BY lower(s.patient_name)
            ORDER BY ultima DESC
            LIMIT ?
            """,
            (fisio_id, limit))

    def get_patient_history(self, patient_name: str, limit: int = 50,
                            fisio_id: int | None = None,
                            paciente_id: int | None = None
                            ) -> list[dict[str, object]]:
        """Series de un paciente, de la más reciente a la más antigua.

        Args:
            patient_name: nombre tal y como se escribió en la cabecera.
            limit: cuántas series como mucho.
            fisio_id: si se indica, solo se devuelven series de pacientes de
                ese profesional. Es lo que impide que escribir un nombre en el
                buscador abra el historial de la consulta de al lado.
            paciente_id: consulta directa por ficha. Manda sobre el nombre, y
                es lo que usa el perfil de paciente para ver **lo suyo**.
        """
        columnas = ("created_at, exercise_id, classification, confidence, "
                    "max_rom, repetitions, correctas, lado, fuente")
        if paciente_id is not None:
            return self._consultar(
                f"SELECT {columnas} FROM sessions WHERE paciente_id = ? "
                "ORDER BY created_at DESC LIMIT ?", (paciente_id, limit))
        if fisio_id is None:
            return self._consultar(
                f"SELECT {columnas} FROM sessions "
                "WHERE lower(patient_name) = lower(?) "
                "ORDER BY created_at DESC LIMIT ?",
                (patient_name.strip(), limit))
        return self._consultar(
            f"SELECT {', '.join('s.' + c.strip() for c in columnas.split(','))} "
            "FROM sessions s JOIN pacientes p ON p.id = s.paciente_id "
            "WHERE lower(s.patient_name) = lower(?) AND p.fisio_id = ? "
            "ORDER BY s.created_at DESC LIMIT ?",
            (patient_name.strip(), fisio_id, limit))
