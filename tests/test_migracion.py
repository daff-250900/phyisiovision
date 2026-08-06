"""La migración de los datos anteriores a los perfiles.

Se parte de una base **como las de antes**: la tabla `sessions` con el nombre
del paciente escrito a mano y ni rastro de `usuarios` ni de `pacientes`. Es el
caso real que había en `data/physiovision.db` cuando esto se introdujo, así que
se reconstruye aquí en vez de darlo por bueno con una base ya migrada.
"""

from __future__ import annotations

import dataclasses
import os
import sqlite3

import pytest

from src import auth, storage
from src.migracion import migrar
from src.storage import (PatientRepository, ROL_FISIO, SessionRepository,
                         UsuarioRepository)

URL_PRUEBA = os.environ.get("PHYSIOVISION_BD_PRUEBA", "")

#: El esquema tal y como estaba antes de los perfiles, en los dos dialectos.
#: Se escribe entero y a mano: reconstruir la base antigua es el punto de la
#: prueba, y generarla con el código actual la haría trivialmente compatible.
ESQUEMA_ANTIGUO = """
CREATE TABLE sessions (
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
"""

SERIES_ANTIGUAS = [("Dafne", 150.0, "2026-07-28T19:34:12+00:00"),
                   ("Dafne", 160.0, "2026-07-29T19:34:12+00:00"),
                   ("dafne", 155.0, "2026-07-30T19:34:12+00:00"),
                   ("Daf", 120.0, "2026-08-01T17:16:07+00:00")]

INSERTAR = ("INSERT INTO sessions (patient_name, exercise_id, classification, "
            "confidence, max_rom, repetitions, created_at) "
            "VALUES ({m}, 'elevacion_lateral_hombro', 'correcto', 0.9, {m}, 5, {m})")


class _Antigua:
    """Dónde vive la base de la prueba, sea cual sea el motor.

    Se pasa como `db_path=` o como `url=` según el motor, que es exactamente lo
    que hace la aplicación al arrancar.
    """

    def __init__(self, ruta=None, url=None):
        self.ruta, self.url = ruta, url

    @property
    def argumentos(self) -> dict:
        return {"url": self.url} if self.url else {"db_path": self.ruta}


@pytest.fixture(params=["sqlite", "postgres"])
def base_antigua(request, tmp_path, monkeypatch):
    """Base anterior a los perfiles: solo `sessions`, y un archivo de usuarios.

    Se prueba en los dos motores porque la migración corre **en cada arranque**,
    y en Cloud Run ese arranque es contra PostgreSQL.
    """
    if request.param == "postgres":
        if not URL_PRUEBA:
            pytest.skip("sin PHYSIOVISION_BD_PRUEBA: se omite PostgreSQL")
        import psycopg

        with psycopg.connect(URL_PRUEBA) as conexion:
            for tabla in ("sessions", "pacientes", "usuarios"):
                conexion.execute(f"DROP TABLE IF EXISTS {tabla} CASCADE")
            conexion.execute(ESQUEMA_ANTIGUO.format(
                serial="SERIAL PRIMARY KEY", real="DOUBLE PRECISION"))
            for fila in SERIES_ANTIGUAS:
                conexion.execute(INSERTAR.format(m="%s"), fila)
            conexion.commit()
        antigua = _Antigua(url=URL_PRUEBA)
        monkeypatch.setenv("PHYSIOVISION_BD", URL_PRUEBA)
    else:
        ruta = tmp_path / "antigua.db"
        conexion = sqlite3.connect(ruta)
        conexion.execute(ESQUEMA_ANTIGUO.format(
            serial="INTEGER PRIMARY KEY AUTOINCREMENT", real="REAL"))
        for fila in SERIES_ANTIGUAS:
            conexion.execute(INSERTAR.format(m="?"), fila)
        conexion.commit()
        conexion.close()
        antigua = _Antigua(ruta=ruta)
        monkeypatch.setenv("PHYSIOVISION_BD", "")
        monkeypatch.setattr(
            storage, "settings",
            dataclasses.replace(storage.settings, database_path=ruta))

    archivo = tmp_path / "usuarios.txt"
    archivo.write_text(f"dafne:{auth.resumir('clave-de-dafne')}\n",
                       encoding="utf-8")
    monkeypatch.setenv("PHYSIOVISION_USUARIOS", "")
    monkeypatch.setenv("PHYSIOVISION_USUARIOS_ARCHIVO", str(archivo))
    return antigua


def test_el_usuario_del_archivo_pasa_a_ser_fisioterapeuta(base_antigua):
    informe = migrar(**base_antigua.argumentos)
    assert informe.usuarios_migrados == ["dafne"]
    cuenta = UsuarioRepository(**base_antigua.argumentos).obtener("dafne")
    assert cuenta["rol"] == ROL_FISIO


def test_la_contrasena_sigue_valiendo_despues_de_migrar(base_antigua):
    """Se traslada el resumen, no se pide una contraseña nueva."""
    migrar(**base_antigua.argumentos)
    cuenta = UsuarioRepository(**base_antigua.argumentos).obtener("dafne")
    assert auth.verificar("clave-de-dafne", cuenta["resumen"])


def test_cada_nombre_distinto_se_convierte_en_una_ficha(base_antigua):
    migrar(**base_antigua.argumentos)
    fichas = PatientRepository(**base_antigua.argumentos).listar()
    assert sorted(f["nombre"].lower() for f in fichas) == ["daf", "dafne"]


def test_las_variantes_de_mayusculas_son_la_misma_persona(base_antigua):
    """"Dafne" y "dafne" eran una sola persona escribiendo con prisa."""
    migrar(**base_antigua.argumentos)
    pacientes = PatientRepository(**base_antigua.argumentos)
    sesiones = SessionRepository(**base_antigua.argumentos)
    ficha = pacientes.por_nombre("Dafne")
    assert sesiones.contar_de_ficha(int(ficha["id"])) == 3


def test_no_se_pierde_ninguna_serie(base_antigua):
    informe = migrar(**base_antigua.argumentos)
    sesiones = SessionRepository(**base_antigua.argumentos)
    assert informe.series_atadas == 4
    assert sesiones.contar_sin_ficha() == 0


def test_las_series_conservan_sus_datos(base_antigua):
    """Migrar ata filas, no las reescribe."""
    antes = SessionRepository(**base_antigua.argumentos).get_patient_history("Dafne")
    migrar(**base_antigua.argumentos)
    despues = SessionRepository(**base_antigua.argumentos).get_patient_history("Dafne")
    assert [f["max_rom"] for f in antes] == [f["max_rom"] for f in despues]
    assert [f["created_at"] for f in antes] == [f["created_at"] for f in despues]


def test_las_fichas_quedan_asignadas_al_fisioterapeuta(base_antigua):
    migrar(**base_antigua.argumentos)
    usuarios = UsuarioRepository(**base_antigua.argumentos)
    pacientes = PatientRepository(**base_antigua.argumentos)
    dafne = usuarios.obtener("dafne")
    assert all(f["fisio_id"] == dafne["id"] for f in pacientes.listar())
    # Y por tanto las ve en su vista de Pacientes.
    lista = SessionRepository(**base_antigua.argumentos).list_patients(
        fisio_id=int(dafne["id"]))
    assert len(lista) == 2


def test_migrar_dos_veces_no_duplica_nada(base_antigua):
    migrar(**base_antigua.argumentos)
    segundo = migrar(**base_antigua.argumentos)
    assert not segundo.hubo_cambios
    assert len(PatientRepository(**base_antigua.argumentos).listar()) == 2
    assert len(UsuarioRepository(**base_antigua.argumentos).listar()) == 1


def test_sin_fisioterapeuta_las_fichas_se_crean_pero_se_avisa(tmp_path,
                                                              monkeypatch):
    """Base con series y sin ningún usuario: no hay a quién asignarlas."""
    ruta = tmp_path / "huerfana.db"
    conexion = sqlite3.connect(ruta)
    conexion.execute(ESQUEMA_ANTIGUO.format(
        serial="INTEGER PRIMARY KEY AUTOINCREMENT", real="REAL"))
    conexion.execute(
        "INSERT INTO sessions (patient_name, exercise_id, classification, "
        "confidence, max_rom, repetitions, created_at) "
        "VALUES ('Nadie', 'elevacion_lateral_hombro', 'correcto', 0.9, 90, 3, "
        "'2026-07-28T00:00:00+00:00')")
    conexion.commit()
    conexion.close()

    monkeypatch.setenv("PHYSIOVISION_USUARIOS", "")
    monkeypatch.setenv("PHYSIOVISION_USUARIOS_ARCHIVO", str(tmp_path / "no-hay.txt"))
    monkeypatch.setenv("PHYSIOVISION_BD", "")

    informe = migrar(db_path=ruta)
    assert any("no hay ningún fisioterapeuta" in a for a in informe.avisos)
    ficha = PatientRepository(db_path=ruta).por_nombre("Nadie")
    assert ficha is not None and ficha["fisio_id"] is None
    # La serie no se pierde: queda atada a la ficha aunque nadie la reclame.
    assert SessionRepository(db_path=ruta).contar_sin_ficha() == 0


def test_con_varios_fisioterapeutas_gana_el_que_venia_del_archivo(base_antigua):
    """"otra" se crea antes, pero "dafne" es quien existía cuando se midió.

    Es lo más parecido a un dato que hay: la cuenta del archivo heredado estaba
    ahí cuando se grabaron esas series; una creada después, no.
    """
    UsuarioRepository(**base_antigua.argumentos).crear(
        "otra", auth.resumir("clave-de-otra"), ROL_FISIO)
    informe = migrar(**base_antigua.argumentos)
    assert informe.fisio_asignado == "dafne"
    assert any("2 fisioterapeutas" in a for a in informe.avisos)


def test_una_base_nueva_no_tiene_nada_que_migrar(tmp_path, monkeypatch):
    monkeypatch.setenv("PHYSIOVISION_USUARIOS", "")
    monkeypatch.setenv("PHYSIOVISION_USUARIOS_ARCHIVO", str(tmp_path / "no-hay.txt"))
    monkeypatch.setenv("PHYSIOVISION_BD", "")
    informe = migrar(db_path=tmp_path / "nueva.db")
    assert not informe.hubo_cambios
    assert str(informe) == "Nada que migrar: la base ya está al día."


# --------------------------------------------------------------------------- #
# Copia de la base local a PostgreSQL (el paso previo a Cloud SQL)
#
# El secreto de despliegue solo transporta `usuario:resumen`, así que reseminar
# desde él perdería los perfiles, las fichas y todo el historial. Estas pruebas
# fijan que la copia los conserva.
# --------------------------------------------------------------------------- #

@pytest.fixture
def origen_completo(tmp_path):
    """SQLite con las tres tablas pobladas y relacionadas entre sí."""
    from src.migracion import copiar  # noqa: F401  (se usa en las pruebas)
    from src.storage import ROL_PACIENTE

    ruta = tmp_path / "origen.db"
    usuarios = UsuarioRepository(db_path=ruta)
    pacientes = PatientRepository(db_path=ruta)
    sesiones = SessionRepository(db_path=ruta)

    ana = usuarios.crear("ana", auth.resumir("clave-de-ana"), ROL_FISIO)
    cuenta = usuarios.crear("marta", auth.resumir("clave-de-marta"), ROL_PACIENTE)
    ficha = pacientes.crear("Marta", fisio_id=ana)
    pacientes.asignar_usuario(ficha, cuenta)
    for rom in (140.0, 155.0):
        sesiones.save_summary(
            {"paciente": "Marta", "ejercicio": "elevacion_lateral_hombro",
             "clasificacion": "correcto", "confianza": 0.9, "rom_max": rom,
             "repeticiones": 5, "correctas": 5, "lado": "right",
             "cobertura_pose": 0.97, "fuente": "xgboost"},
            paciente_id=ficha, fisio_id=ana)
    return ruta


def _postgres_vacio():
    import psycopg

    with psycopg.connect(URL_PRUEBA) as conexion:
        for tabla in ("sessions", "pacientes", "usuarios"):
            conexion.execute(f"DROP TABLE IF EXISTS {tabla} CASCADE")
        conexion.commit()


@pytest.mark.skipif(not URL_PRUEBA, reason="sin PHYSIOVISION_BD_PRUEBA")
def test_la_copia_conserva_perfiles_fichas_e_historial(origen_completo):
    from src.migracion import copiar

    _postgres_vacio()
    copiadas = copiar(URL_PRUEBA, origen=origen_completo)
    assert copiadas == {"usuarios": 2, "pacientes": 1, "sessions": 2}

    usuarios = UsuarioRepository(url=URL_PRUEBA)
    pacientes = PatientRepository(url=URL_PRUEBA)
    sesiones = SessionRepository(url=URL_PRUEBA)

    # El perfil de paciente es justo lo que un reseminado desde el secreto
    # perdería: allí toda cuenta llega como fisioterapeuta.
    assert usuarios.obtener("marta")["rol"] == "paciente"
    ficha = pacientes.por_nombre("Marta")
    assert ficha["usuario_id"] == usuarios.obtener("marta")["id"]
    assert ficha["fisio_id"] == usuarios.obtener("ana")["id"]
    assert sesiones.contar_de_ficha(int(ficha["id"])) == 2
    assert sesiones.contar_sin_ficha() == 0


@pytest.mark.skipif(not URL_PRUEBA, reason="sin PHYSIOVISION_BD_PRUEBA")
def test_las_contrasenas_siguen_valiendo_tras_copiar(origen_completo):
    from src.migracion import copiar

    _postgres_vacio()
    copiar(URL_PRUEBA, origen=origen_completo)
    cuenta = UsuarioRepository(url=URL_PRUEBA).obtener("ana")
    assert auth.verificar("clave-de-ana", cuenta["resumen"])


@pytest.mark.skipif(not URL_PRUEBA, reason="sin PHYSIOVISION_BD_PRUEBA")
def test_se_puede_seguir_escribiendo_despues_de_copiar(origen_completo):
    """Al copiar los `id` a mano, la secuencia de PostgreSQL se queda atrás.

    Sin corregirla, el primer INSERT posterior choca con la fila 1 y la
    aplicación deja de poder guardar series: un fallo que solo aparece en
    producción, con el paciente delante.
    """
    from src.migracion import copiar

    _postgres_vacio()
    copiar(URL_PRUEBA, origen=origen_completo)
    sesiones = SessionRepository(url=URL_PRUEBA)
    nuevo = sesiones.save_summary(
        {"paciente": "Marta", "ejercicio": "elevacion_lateral_hombro",
         "clasificacion": "correcto", "confianza": 0.9, "rom_max": 150.0,
         "repeticiones": 5, "correctas": 5, "lado": "right",
         "cobertura_pose": 0.97, "fuente": "xgboost"}, paciente_id=1, fisio_id=1)
    assert nuevo > 2
    assert UsuarioRepository(url=URL_PRUEBA).crear(
        "nueva", auth.resumir("clave-larga"), ROL_FISIO) > 2


@pytest.mark.skipif(not URL_PRUEBA, reason="sin PHYSIOVISION_BD_PRUEBA")
def test_se_niega_a_copiar_sobre_una_base_con_datos(origen_completo):
    """Copiar encima duplicaría filas o chocaría con los identificadores."""
    from src.migracion import copiar

    _postgres_vacio()
    copiar(URL_PRUEBA, origen=origen_completo)
    with pytest.raises(RuntimeError, match="ya tiene datos"):
        copiar(URL_PRUEBA, origen=origen_completo)
    # Y con --forzar vuelve a dejarlo exacto, sin duplicar.
    copiado = copiar(URL_PRUEBA, origen=origen_completo, forzar=True)
    assert copiado == {"usuarios": 2, "pacientes": 1, "sessions": 2}


def test_el_destino_tiene_que_ser_postgres(origen_completo, tmp_path):
    from src.migracion import copiar

    with pytest.raises(ValueError, match="PostgreSQL"):
        copiar(str(tmp_path / "otra.db"), origen=origen_completo)
