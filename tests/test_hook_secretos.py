"""Pruebas del hook que bloquea credenciales en el índice.

El hook existe por un incidente real: una clave de API acabó en `.env.example`
—el archivo que sí se versiona— y viajó en cuatro commits hasta que GitHub
rechazó el push, cuando ya había que reescribir el historial.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

RUTA_HOOK = Path(__file__).resolve().parent.parent / ".githooks" / "pre-commit"


def _cargar():
    especificacion = importlib.util.spec_from_loader(
        "hook_secretos",
        importlib.machinery.SourceFileLoader("hook_secretos", str(RUTA_HOOK)))
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


hook = _cargar()


# --------------------------------------------------------------------------- #
# Credenciales por su forma
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("linea, esperado", [
    ("GEMINI_API_KEY=AQ.Ab8RN6JxK2mPqR7sT9vW1yZ3aB5cD7eF9gH0iJ2k",  # secreto-de-prueba
     "Google Cloud"),
    ("clave = 'AIza" + "B" * 35 + "'", "Google"),
    ("token = ya29.a0AfH6SMBx1234567890abcdefghij", "OAuth"),  # secreto-de-prueba
    ("-----BEGIN PRIVATE KEY-----", "clave privada"),  # secreto-de-prueba
    ('  "private_key": "-----BEGIN PRIVATE KEY-----\\n"',  # secreto-de-prueba
     "cuenta de servicio"),
    ("aws = AKIAIOSFODNN7EXAMPLE", "AWS"),  # secreto-de-prueba
    ("openai = sk-" + "a" * 40, "OpenAI"),
    ("gh = ghp_" + "b" * 36, "GitHub"),
])
def test_detecta_credenciales(linea, esperado) -> None:
    problemas = hook.revisar("src/algo.py", linea)
    assert problemas, f"no detectó: {linea[:40]}"
    assert esperado.lower() in problemas[0].lower()


def test_no_marca_codigo_normal() -> None:
    codigo = "\n".join([
        "MODELO = 'gemini-flash-lite-latest'",
        "api_key = os.environ.get('GEMINI_API_KEY')",
        "TIMEOUT_S = 10.0",
        "def sintetizar(texto): return None",
    ])
    assert hook.revisar("src/gemini_feedback.py", codigo) == []


# --------------------------------------------------------------------------- #
# Plantillas con valores rellenos
# --------------------------------------------------------------------------- #

def test_plantilla_con_valor_en_variable_sensible() -> None:
    """El caso exacto del incidente."""
    problemas = hook.revisar(".env.example", "GEMINI_API_KEY=una-clave-cualquiera")
    assert problemas and "déjalo vacío" in problemas[0]


def test_plantilla_con_variable_sensible_vacia_es_correcta() -> None:
    assert hook.revisar(".env.example", "GEMINI_API_KEY=") == []
    assert hook.revisar(".env.example", "GOOGLE_APPLICATION_CREDENTIALS=") == []


def test_plantilla_admite_valores_no_sensibles() -> None:
    """La plantilla documenta configuración; solo los secretos van vacíos."""
    plantilla = "\n".join([
        "PHYSIOVISION_GEMINI_MODELO=gemini-flash-lite-latest",
        "PHYSIOVISION_TTS_VELOCIDAD=1.05",
        "PHYSIOVISION_HOST=127.0.0.1",
        "PHYSIOVISION_MEDIAPIPE=heavy",
    ])
    assert hook.revisar(".env.example", plantilla) == []


def test_mediapipe_no_es_falso_positivo() -> None:
    """PHYSIOVISION_MEDIAPIPE contiene 'API' como subcadena: no cuenta."""
    assert not hook.es_nombre_sensible("PHYSIOVISION_MEDIAPIPE")
    assert hook.es_nombre_sensible("GEMINI_API_KEY")
    assert hook.es_nombre_sensible("GOOGLE_APPLICATION_CREDENTIALS")
    assert hook.es_nombre_sensible("DB_PASSWORD")
    assert not hook.es_nombre_sensible("PHYSIOVISION_PORT")


def test_marcadores_de_plantilla_no_cuentan() -> None:
    for valor in ("<tu-clave>", "tu-clave-aqui", "changeme", "xxxx",
                  "/ruta/a/credenciales.json", "path/to/key.json"):
        assert hook.revisar(".env.example", f"API_KEY={valor}") == [], valor


def test_solo_se_aplica_a_plantillas() -> None:
    """En un .py, `TOKEN = "abc"` es código normal, no una plantilla mal puesta."""
    assert hook.revisar("src/config.py", 'TOKEN = "abc123"') == []


# --------------------------------------------------------------------------- #
# Archivos que no deben versionarse
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ruta", [
    ".env", "subdir/.env", "credenciales.json",
    "service_account.json", "service-account-fisio.json",
])
def test_rutas_prohibidas(ruta) -> None:
    problemas = hook.revisar(ruta, "contenido")
    assert problemas and "no debe versionarse" in problemas[0]


def test_env_example_no_esta_prohibido() -> None:
    assert hook.revisar(".env.example", "") == []


# --------------------------------------------------------------------------- #
# Integración: el hook real sobre un repositorio de prueba
# --------------------------------------------------------------------------- #

def test_el_hook_bloquea_de_verdad(tmp_path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)

    ganchos = tmp_path / ".githooks"
    ganchos.mkdir()
    destino = ganchos / "pre-commit"
    destino.write_text(RUTA_HOOK.read_text(encoding="utf-8"), encoding="utf-8")
    destino.chmod(0o755)
    subprocess.run(["git", "config", "core.hooksPath", ".githooks"],
                   cwd=tmp_path, check=True)

    (tmp_path / ".env.example").write_text(          # secreto-de-prueba
        "GEMINI_API_KEY=AQ.Ab8RN6JxK2mPqR7sT9vW1yZ3aB5cD7eF9gH0iJ2k\n")  # secreto-de-prueba
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)

    r = subprocess.run(["git", "commit", "-m", "con secreto"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode != 0, "el hook dejó pasar el secreto"
    assert "BLOQUEADO" in r.stderr

    # Con el valor vacío, el mismo commit pasa.
    (tmp_path / ".env.example").write_text("GEMINI_API_KEY=\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    r = subprocess.run(["git", "commit", "-m", "limpio"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, f"bloqueó un commit limpio: {r.stderr}"


def test_no_verify_permite_saltarlo(tmp_path) -> None:
    """Debe haber una salida para falsos positivos."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    ganchos = tmp_path / ".githooks"
    ganchos.mkdir()
    (ganchos / "pre-commit").write_text(RUTA_HOOK.read_text(encoding="utf-8"))
    (ganchos / "pre-commit").chmod(0o755)
    subprocess.run(["git", "config", "core.hooksPath", ".githooks"],
                   cwd=tmp_path, check=True)

    (tmp_path / ".env.example").write_text("API_KEY=algo-que-parece-secreto\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    r = subprocess.run(["git", "commit", "--no-verify", "-m", "forzado"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0


def test_la_marca_permite_excluir_una_linea() -> None:
    """Sin esto, este mismo archivo de pruebas no se podría commitear."""
    linea = "clave = AKIAIOSFODNN7EXAMPLE"  # secreto-de-prueba
    assert hook.revisar("x.py", linea)
    assert hook.revisar("x.py", f"{linea}  # {hook.PERMITIDO}") == []


def test_la_marca_es_por_linea_no_por_archivo() -> None:
    """Excluir tests/ entero dejaría pasar una clave real en un test."""
    contenido = (f"falsa = AKIAIOSFODNN7EXAMPLE  # {hook.PERMITIDO}\n"  # secreto-de-prueba
                 "de_verdad = AKIAIOSFODNN7REALKEY\n")  # secreto-de-prueba
    problemas = hook.revisar("tests/algo.py", contenido)
    assert len(problemas) == 1
    assert ":2:" in problemas[0], "debe señalar la línea sin marcar"
