"""Autenticación: resúmenes, carga de usuarios y la puerta de arranque."""

from __future__ import annotations

import pytest

from src import auth


# -- resumen y verificación --------------------------------------------------- #

def test_ida_y_vuelta():
    resumen = auth.resumir("una-contraseña-larga")
    assert auth.verificar("una-contraseña-larga", resumen)


def test_contrasena_incorrecta():
    assert not auth.verificar("otra", auth.resumir("una-contraseña-larga"))


def test_la_contrasena_no_aparece_en_el_resumen():
    resumen = auth.resumir("secreto-en-claro")
    assert "secreto-en-claro" not in resumen
    assert resumen.startswith("pbkdf2_sha256$")


def test_dos_resumenes_de_la_misma_contrasena_difieren():
    """Cada uno lleva su sal: si coincidieran, se verían las repetidas."""
    assert auth.resumir("misma") != auth.resumir("misma")


@pytest.mark.parametrize("resumen", ["", "loquesea", "md5$1$aa$bb",
                                     "pbkdf2_sha256$no-es-un-numero$aa$bb"])
def test_resumen_corrupto_no_revienta(resumen):
    assert auth.verificar("x", resumen) is False


# -- carga de usuarios -------------------------------------------------------- #

def test_carga_desde_la_variable(monkeypatch):
    linea = f"dafne:{auth.resumir('contraseña1')}"
    monkeypatch.setenv("PHYSIOVISION_USUARIOS", linea)
    assert list(auth.cargar_usuarios()) == ["dafne"]


def test_varios_usuarios_separados_por_comas(monkeypatch):
    valor = ",".join([f"a:{auth.resumir('clave-a-larga')}",
                      f"b:{auth.resumir('clave-b-larga')}"])
    monkeypatch.setenv("PHYSIOVISION_USUARIOS", valor)
    assert set(auth.cargar_usuarios()) == {"a", "b"}


def test_carga_desde_archivo(tmp_path, monkeypatch):
    ruta = tmp_path / "usuarios.txt"
    ruta.write_text(f"# comentario\n\nluis:{auth.resumir('clave-larga-1')}\n",
                    encoding="utf-8")
    monkeypatch.setenv("PHYSIOVISION_USUARIOS", "")
    monkeypatch.setenv("PHYSIOVISION_USUARIOS_ARCHIVO", str(ruta))
    assert list(auth.cargar_usuarios()) == ["luis"]


def test_lineas_rotas_se_ignoran(monkeypatch):
    """Una línea mal escrita no puede tumbar el arranque ni colarse."""
    monkeypatch.setenv("PHYSIOVISION_USUARIOS",
                       f"sin-resumen,otro:texto-plano,ok:{auth.resumir('clave-larga')}")
    assert list(auth.cargar_usuarios()) == ["ok"]


def test_sin_configuracion_no_hay_usuarios(monkeypatch, tmp_path):
    monkeypatch.setenv("PHYSIOVISION_USUARIOS", "")
    monkeypatch.setenv("PHYSIOVISION_USUARIOS_ARCHIVO", str(tmp_path / "no-existe"))
    assert auth.cargar_usuarios() == {}
    assert auth.hay_usuarios() is False


# -- autenticación ------------------------------------------------------------ #

@pytest.fixture
def con_usuario(monkeypatch):
    monkeypatch.setenv("PHYSIOVISION_USUARIOS", f"dafne:{auth.resumir('clave-buena')}")


def test_autenticar_correcto(con_usuario):
    assert auth.autenticar("dafne", "clave-buena")


@pytest.mark.parametrize("usuario,clave", [
    ("dafne", "clave-mala"),
    ("otro", "clave-buena"),
    ("", ""),
    ("dafne", ""),
])
def test_autenticar_rechaza(con_usuario, usuario, clave):
    assert not auth.autenticar(usuario, clave)


def test_usuario_desconocido_no_lanza(monkeypatch, tmp_path):
    monkeypatch.setenv("PHYSIOVISION_USUARIOS", "")
    monkeypatch.setenv("PHYSIOVISION_USUARIOS_ARCHIVO", str(tmp_path / "no-existe"))
    assert auth.autenticar("cualquiera", "cosa") is False


# -- puerta de arranque ------------------------------------------------------- #

@pytest.mark.parametrize("host,exige", [
    ("127.0.0.1", False),
    ("localhost", False),
    ("::1", False),
    ("0.0.0.0", True),
    ("192.168.1.40", True),
])
def test_solo_se_exige_login_fuera_de_loopback(host, exige):
    assert auth.exige_login(host) is exige


# -- interfaz ----------------------------------------------------------------- #

def test_la_interfaz_dice_quien_ha_entrado():
    from ui.callbacks import quien_ha_entrado

    peticion = type("Peticion", (), {"username": "dafne"})()
    assert "dafne" in quien_ha_entrado(peticion)


def test_sin_login_la_interfaz_lo_declara():
    """Una app abierta no puede parecerse a una cerrada."""
    from ui.callbacks import quien_ha_entrado

    assert "Sin autenticación" in quien_ha_entrado(None)


def test_salir_cierra_el_navegador_solo_con_login():
    from ui.app_ui import create_app

    con = create_app(con_login=True).get_config_file()["dependencies"]
    sin = create_app(con_login=False).get_config_file()["dependencies"]
    assert sum(1 for d in con if "logout" in (d.get("js") or "")) == 1
    assert sum(1 for d in sin if "logout" in (d.get("js") or "")) == 0


# -- freno a la fuerza bruta -------------------------------------------------- #

@pytest.fixture(autouse=True)
def _sin_bloqueos():
    """Cada prueba arranca sin memoria de intentos previos."""
    auth.reiniciar_intentos()
    yield
    auth.reiniciar_intentos()


def test_cinco_fallos_bloquean_al_usuario(con_usuario):
    for _ in range(auth.MAX_INTENTOS):
        assert not auth.autenticar("dafne", "mala")
    # Con la contraseña correcta tampoco entra: está castigado.
    assert not auth.autenticar("dafne", "clave-buena")


def test_el_bloqueo_no_alcanza_a_los_demas(con_usuario, monkeypatch):
    monkeypatch.setenv("PHYSIOVISION_USUARIOS",
                       f"dafne:{auth.resumir('clave-buena')},"
                       f"luis:{auth.resumir('otra-clave-larga')}")
    for _ in range(auth.MAX_INTENTOS):
        auth.autenticar("dafne", "mala")
    assert auth.autenticar("luis", "otra-clave-larga")


def test_acertar_antes_del_limite_borra_el_contador(con_usuario):
    for _ in range(auth.MAX_INTENTOS - 1):
        auth.autenticar("dafne", "mala")
    assert auth.autenticar("dafne", "clave-buena")
    for _ in range(auth.MAX_INTENTOS - 1):
        assert not auth.autenticar("dafne", "mala")
    assert auth.autenticar("dafne", "clave-buena")


def test_el_castigo_crece_al_insistir(con_usuario, monkeypatch):
    """Cinco minutos, luego diez, luego veinte… hasta el tope."""
    reloj = {"t": 1000.0}
    monkeypatch.setattr(auth.time, "monotonic", lambda: reloj["t"])

    esperas = []
    for ronda in range(3):
        for _ in range(auth.MAX_INTENTOS):
            auth.autenticar("dafne", "mala")
        hasta, _veces = auth._bloqueos["dafne"]
        esperas.append(hasta - reloj["t"])
        reloj["t"] = hasta + 1          # se cumple el castigo y vuelve a fallar

    assert esperas == sorted(esperas) and esperas[0] < esperas[-1]
    assert esperas[-1] <= auth.BLOQUEO_MAXIMO_S


def test_pasado_el_castigo_se_puede_volver_a_entrar(con_usuario, monkeypatch):
    reloj = {"t": 500.0}
    monkeypatch.setattr(auth.time, "monotonic", lambda: reloj["t"])
    for _ in range(auth.MAX_INTENTOS):
        auth.autenticar("dafne", "mala")
    assert not auth.autenticar("dafne", "clave-buena")

    reloj["t"] += auth.BLOQUEO_BASE_S + 1
    assert auth.autenticar("dafne", "clave-buena")


def test_los_fallos_caducan_con_la_ventana(con_usuario, monkeypatch):
    """Cuatro fallos hoy y uno mañana no suman cinco."""
    reloj = {"t": 0.0}
    monkeypatch.setattr(auth.time, "monotonic", lambda: reloj["t"])
    for _ in range(auth.MAX_INTENTOS - 1):
        auth.autenticar("dafne", "mala")
    reloj["t"] += auth.VENTANA_S + 1
    auth.autenticar("dafne", "mala")
    assert auth.autenticar("dafne", "clave-buena")


def test_queda_registro_de_lo_que_pasa(con_usuario, caplog):
    with caplog.at_level("INFO", logger="src.auth"):
        auth.autenticar("dafne", "clave-buena")
        auth.autenticar("dafne", "mala")
    registro = caplog.text
    assert "acceso concedido" in registro and "incorrectas" in registro
    assert "clave-buena" not in registro and "mala" not in registro


# -- pantalla de acceso ------------------------------------------------------- #

def test_la_portada_lleva_marca_y_aviso():
    from ui.app_ui import MENSAJE_ACCESO

    assert "PhysioVision".replace("PhysioVision", "Physio") in MENSAJE_ACCESO
    assert "data:image/png;base64" in MENSAJE_ACCESO      # logotipo incrustado
    assert "datos de pacientes" in MENSAJE_ACCESO
    assert "style=" in MENSAJE_ACCESO                     # sin CSS de la app
