"""Perfiles: dos roles, y cada fisioterapeuta con sus pacientes.

Lo que se comprueba aquí no es que las consultas devuelvan filas, sino que **no
devuelvan las de otro**. Es la parte que, si se rompe, no lo nota nadie hasta
que un profesional ve el historial de la consulta de al lado.

Se ejecuta contra **los dos motores**. Cloud Run despliega sobre PostgreSQL y
el equipo desarrolla sobre SQLite: probar solo uno es la forma segura de que el
otro se rompa el día del despliegue, y aquí hay dos índices únicos sobre
expresiones (`lower(...)`) que no tienen por qué comportarse igual.

PostgreSQL se omite si no hay `PHYSIOVISION_BD_PRUEBA`, igual que en
`tests/test_storage.py`.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

from src import auth, storage
from src.storage import (PatientRepository, ROL_FISIO, ROL_PACIENTE,
                         SessionRepository, UsuarioRepository)

URL_PRUEBA = os.environ.get("PHYSIOVISION_BD_PRUEBA", "")

#: Las tres tablas, en orden de borrado seguro.
TABLAS = ("sessions", "pacientes", "usuarios")


def _apuntar_a(monkeypatch, ruta) -> None:
    """Hace que los repositorios sin argumentos abran la base de la prueba.

    `settings` es un dataclass congelado, así que no se le asigna un campo: se
    sustituye el objeto entero en el módulo que lo consulta, que es lo que hace
    `dataclasses.replace`.
    """
    monkeypatch.setenv("PHYSIOVISION_BD", "")
    monkeypatch.setattr(
        storage, "settings",
        dataclasses.replace(storage.settings, database_path=ruta))


def _limpiar_postgres() -> None:
    import psycopg

    with psycopg.connect(URL_PRUEBA) as conexion:
        for tabla in TABLAS:
            conexion.execute(f"DROP TABLE IF EXISTS {tabla} CASCADE")
        conexion.commit()


def _serie(nombre: str, **extra):
    base = {"paciente": nombre, "ejercicio": "elevacion_lateral_hombro",
            "clasificacion": "correcto", "confianza": 0.9, "rom_max": 150.0,
            "repeticiones": 5, "correctas": 5, "lado": "right",
            "cobertura_pose": 0.98, "fuente": "xgboost"}
    return {**base, **extra}


@pytest.fixture(params=["sqlite", "postgres"])
def base(request, tmp_path):
    """Una base vacía de cada motor, con dos fisios y un paciente cada uno."""
    if request.param == "postgres":
        if not URL_PRUEBA:
            pytest.skip("sin PHYSIOVISION_BD_PRUEBA: se omite PostgreSQL")
        _limpiar_postgres()
        argumentos = {"url": URL_PRUEBA}
        ruta = None
    else:
        ruta = tmp_path / "perfiles.db"
        argumentos = {"db_path": ruta}

    usuarios = UsuarioRepository(**argumentos)
    pacientes = PatientRepository(**argumentos)
    sesiones = SessionRepository(**argumentos)

    ana = usuarios.crear("ana", auth.resumir("clave-de-ana"), ROL_FISIO)
    luis = usuarios.crear("luis", auth.resumir("clave-de-luis"), ROL_FISIO)

    marta = pacientes.crear("Marta", fisio_id=ana)
    pedro = pacientes.crear("Pedro", fisio_id=luis)

    sesiones.save_summary(_serie("Marta"), paciente_id=marta, fisio_id=ana)
    sesiones.save_summary(_serie("Marta", rom_max=160.0), paciente_id=marta,
                          fisio_id=ana)
    sesiones.save_summary(_serie("Pedro"), paciente_id=pedro, fisio_id=luis)

    def apuntar(monkeypatch):
        """Hace que los repositorios sin argumentos abran **esta** base.

        Es lo que necesitan las pruebas que pasan por `src.auth` o por la
        interfaz, que construyen los repositorios por su cuenta.
        """
        if ruta is None:
            monkeypatch.setenv("PHYSIOVISION_BD", URL_PRUEBA)
        else:
            _apuntar_a(monkeypatch, ruta)

    return {"ruta": ruta, "apuntar": apuntar, "usuarios": usuarios,
            "pacientes": pacientes, "sesiones": sesiones,
            "ana": ana, "luis": luis, "marta": marta, "pedro": pedro}


# --------------------------------------------------------------------------- #
# Cuentas y perfiles
# --------------------------------------------------------------------------- #

def test_el_rol_tiene_que_ser_uno_de_los_dos(base):
    with pytest.raises(ValueError):
        base["usuarios"].crear("otra", auth.resumir("x" * 8), "administrador")


def test_no_se_pueden_repetir_usuarios_cambiando_mayusculas(base):
    """El login no distingue mayúsculas: dos cuentas así serían la misma.

    La excepción concreta la pone el controlador —`IntegrityError` en SQLite,
    `UniqueViolation` en PostgreSQL—, así que se comprueba que la base lo
    rechace, no cómo se llame el error.
    """
    with pytest.raises(Exception) as fallo:
        base["usuarios"].crear("ANA", auth.resumir("otra-clave"), ROL_FISIO)
    assert type(fallo.value).__name__ in ("IntegrityError", "UniqueViolation")


def test_obtener_no_distingue_mayusculas(base):
    assert base["usuarios"].obtener("ANA")["usuario"] == "ana"


def test_la_baja_es_logica(base):
    base["usuarios"].desactivar("luis")
    assert base["usuarios"].obtener("luis") is None
    # La ficha y sus series siguen ahí: son datos clínicos, no del usuario.
    assert base["pacientes"].por_id(base["pedro"]) is not None
    assert base["sesiones"].contar_de_ficha(base["pedro"]) == 1


# --------------------------------------------------------------------------- #
# Aislamiento entre fisioterapeutas
# --------------------------------------------------------------------------- #

def test_cada_fisio_solo_ve_sus_pacientes(base):
    de_ana = base["sesiones"].list_patients(fisio_id=base["ana"])
    de_luis = base["sesiones"].list_patients(fisio_id=base["luis"])
    assert [f["paciente"] for f in de_ana] == ["Marta"]
    assert [f["paciente"] for f in de_luis] == ["Pedro"]


def test_el_historial_de_otro_no_se_alcanza_escribiendo_el_nombre(base):
    """Escribir "Pedro" en el buscador de Ana no puede abrir su historial."""
    assert base["sesiones"].get_patient_history("Pedro", fisio_id=base["ana"]) == []
    assert len(base["sesiones"].get_patient_history("Pedro",
                                                    fisio_id=base["luis"])) == 1


def test_sin_fisio_se_ve_todo(base):
    """Arranque local sin login: no hay a quién filtrar y no se filtra."""
    assert len(base["sesiones"].list_patients()) == 2


def test_dos_fisios_pueden_tener_un_paciente_con_el_mismo_nombre(base):
    """La unicidad es por profesional: son dos personas distintas."""
    otra_marta = base["pacientes"].crear("Marta", fisio_id=base["luis"])
    assert otra_marta != base["marta"]
    assert base["pacientes"].por_nombre("Marta", base["luis"])["id"] == otra_marta
    assert base["pacientes"].por_nombre("Marta", base["ana"])["id"] == base["marta"]


def test_obtener_o_crear_no_duplica(base):
    ficha = base["pacientes"].obtener_o_crear("Marta", fisio_id=base["ana"])
    assert ficha["id"] == base["marta"]
    assert len(base["pacientes"].listar(base["ana"])) == 1


# --------------------------------------------------------------------------- #
# Perfil de paciente
# --------------------------------------------------------------------------- #

def test_el_paciente_consulta_por_ficha_no_por_nombre(base):
    """Su historial sale de su `paciente_id`, no de un texto del navegador."""
    historial = base["sesiones"].get_patient_history("da igual lo que escriba",
                                                     paciente_id=base["marta"])
    assert len(historial) == 2


class _Peticion:
    """Lo único que la interfaz necesita de `gr.Request`."""

    def __init__(self, username):
        self.username = username


def _menu_de(usuario, base, monkeypatch) -> dict[str, bool]:
    """`{entrada: visible}` de la tira lateral para ese usuario."""
    base["apuntar"](monkeypatch)
    from ui.app_ui import NAV
    from ui.callbacks import preparar_perfil

    _, _, *entradas = preparar_perfil(_Peticion(usuario))
    return {clave: entrada.get("visible", True)
            for (clave, _), entrada in zip(NAV, entradas)}


@pytest.fixture
def con_cuenta_de_paciente(base):
    cuenta = base["usuarios"].crear("marta", auth.resumir("clave-de-marta"),
                                    ROL_PACIENTE, "Marta Ruiz")
    base["pacientes"].asignar_usuario(base["marta"], cuenta)
    return base


def test_el_paciente_no_ve_la_lista_de_otros_pacientes(con_cuenta_de_paciente,
                                                       monkeypatch):
    menu = _menu_de("marta", con_cuenta_de_paciente, monkeypatch)
    assert menu["pacientes"] is False


def test_el_paciente_si_entra_en_configuracion(con_cuenta_de_paciente,
                                               monkeypatch):
    """No guarda nada de nadie: dice si hay modelo, si hay voz y qué tema hay.

    Comprobar por qué la sesión va en silencio, o pasar a modo oscuro, es usar
    la aplicación, no administrarla.
    """
    menu = _menu_de("marta", con_cuenta_de_paciente, monkeypatch)
    assert menu["configuracion"] is not False


def test_el_paciente_conserva_lo_suyo(con_cuenta_de_paciente, monkeypatch):
    menu = _menu_de("marta", con_cuenta_de_paciente, monkeypatch)
    for entrada in ("sesion", "progreso", "historial", "ejercicios", "ayuda"):
        assert menu[entrada] is not False


def test_el_fisioterapeuta_lo_ve_todo(con_cuenta_de_paciente, monkeypatch):
    menu = _menu_de("ana", con_cuenta_de_paciente, monkeypatch)
    assert all(visible is not False for visible in menu.values())


def test_el_perfil_de_paciente_trae_su_ficha(base, monkeypatch):
    cuenta = base["usuarios"].crear("marta", auth.resumir("clave-de-marta"),
                                    ROL_PACIENTE, "Marta Ruiz")
    base["pacientes"].asignar_usuario(base["marta"], cuenta)
    base["apuntar"](monkeypatch)

    ficha = auth.perfil("marta")
    assert ficha["rol"] == ROL_PACIENTE
    assert ficha["paciente_id"] == base["marta"]
    assert ficha["paciente_nombre"] == "Marta"


def test_el_login_admite_las_dos_fuentes_a_la_vez(base, monkeypatch):
    """Cuenta de la base y usuario configurado por secreto: entran los dos.

    En Cloud Run los usuarios llegan en `PHYSIOVISION_USUARIOS` y no hay disco
    donde guardar una base; en la clínica, al revés. Quedarse con una sola
    fuente dejaría fuera a alguien en uno de los dos despliegues.
    """
    monkeypatch.setenv("PHYSIOVISION_USUARIOS",
                       f"desplegada:{auth.resumir('clave-del-secreto')}")
    base["apuntar"](monkeypatch)

    assert set(auth.cargar_usuarios()) == {"ana", "luis", "desplegada"}
    assert auth.autenticar("desplegada", "clave-del-secreto")
    assert auth.autenticar("ana", "clave-de-ana")


def test_si_el_nombre_coincide_manda_la_base(base, monkeypatch):
    """Es donde aterriza un cambio de contraseña, así que es la buena."""
    monkeypatch.setenv("PHYSIOVISION_USUARIOS",
                       f"ana:{auth.resumir('clave-vieja-del-archivo')}")
    base["apuntar"](monkeypatch)

    assert auth.autenticar("ana", "clave-de-ana")
    auth.reiniciar_intentos()
    assert not auth.autenticar("ana", "clave-vieja-del-archivo")


def test_un_usuario_solo_del_archivo_se_considera_fisioterapeuta(tmp_path,
                                                                 monkeypatch):
    """Instalación sin migrar todavía: quien entraba era el profesional."""
    archivo = tmp_path / "usuarios.txt"
    archivo.write_text(f"vieja:{auth.resumir('clave-vieja')}\n", encoding="utf-8")
    monkeypatch.setenv("PHYSIOVISION_USUARIOS", "")
    monkeypatch.setenv("PHYSIOVISION_USUARIOS_ARCHIVO", str(archivo))
    _apuntar_a(monkeypatch, tmp_path / "vacia.db")

    assert auth.perfil("vieja")["rol"] == ROL_FISIO
    assert auth.perfil("inexistente") is None
    # Y con la tabla vacía, el login sigue admitiéndolo: nadie se queda fuera
    # por no haber migrado todavía.
    assert auth.autenticar("vieja", "clave-vieja")
