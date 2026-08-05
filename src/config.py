"""Configuración del proyecto.

Importar este módulo carga `.env` si existe, así que basta con hacerlo en
cualquier punto de entrada para que las variables del archivo estén disponibles.
"""

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

#: Dónde se escribe lo que la aplicación produce: base de pacientes, caché de
#: audio y vídeos procesados. Es configurable porque no siempre puede vivir
#: junto al código: en Hugging Face Spaces el almacenamiento persistente se
#: monta en `/data`, y en un contenedor conviene que sea un volumen aparte.
DATOS_DIR = Path(os.environ.get("PHYSIOVISION_DATOS", BASE_DIR / "data"))


def cargar_dotenv(ruta: Path | None = None) -> dict[str, str]:
    """Vuelca `.env` en el entorno. Devuelve lo que haya cargado.

    Implementación mínima a propósito, para no añadir `python-dotenv` por doce
    líneas. Admite `CLAVE=valor`, comentarios con `#`, líneas en blanco, el
    prefijo `export` y comillas alrededor del valor.

    **No pisa variables ya definidas**: lo que exportes en la terminal manda
    sobre el archivo, que es lo que se espera al depurar.
    """
    ruta = ruta or BASE_DIR / ".env"
    cargadas: dict[str, str] = {}
    if not ruta.exists():
        return cargadas
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        clave = clave.removeprefix("export ").strip()
        valor = valor.strip().strip("'\"")
        if not clave or not valor or clave in os.environ:
            continue
        os.environ[clave] = valor
        cargadas[clave] = valor
    return cargadas


#: Se carga al importar, antes de que nadie lea os.environ.
DOTENV_CARGADO = cargar_dotenv()


@dataclass(frozen=True)
class Settings:
    model_path: Path = BASE_DIR / "models" / "xgboost_model.json"
    feature_contract_path: Path = BASE_DIR / "models" / "feature_contract.json"
    knowledge_base_path: Path = BASE_DIR / "knowledge_base" / "ejercicios.json"
    database_path: Path = DATOS_DIR / "physiovision.db"
    processed_dir: Path = DATOS_DIR / "processed"
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # --- MediaPipe Pose (API Tasks) ---
    mediapipe_dir: Path = BASE_DIR / "models" / "mediapipe"
    #: Variante de pesos. **Tiene que ser la misma con la que se entrenó.**
    #:
    #: El contrato del modelo no es solo la lista de variables: incluye el
    #: extractor de landmarks. Medido sobre PM_000, mismo video y mismo
    #: reescalado, comparando contra el CSV que generó el entrenamiento:
    #:
    #:     heavy -> diferencia media 0.00 grados (reproduce el CSV exactamente)
    #:     full  -> diferencia media 13.05 grados, abduccion maxima 168 vs 147
    #:
    #: Con `full` las cuatro repeticiones de PM_000 se clasificaban como
    #: compensacion_tronco; con `heavy`, como en el entrenamiento. Cambiar de
    #: variante exige reextraer los landmarks y reentrenar.
    mediapipe_variant: str = os.environ.get("PHYSIOVISION_MEDIAPIPE", "heavy")

    # --- Redacción con Gemini (opcional) ---
    #: Verificado contra la API real (2026-07-28). Se eligió la variante *lite*
    #: por latencia: reformular una frase no necesita razonamiento, y el modelo
    #: con "pensamiento" gastaba 1489 tokens de reflexión para 43 de salida.
    #:
    #:     gemini-flash-latest      7.4 s
    #:     gemini-flash-lite-latest 0.6 s   <- este
    #:
    #: Con repeticiones cada ~5 s, 7 s por llamada haría que las redacciones se
    #: acumularan en la cola. Se usa el alias `-latest` en vez de una versión
    #: fija para que no caduque: `gemini-2.5-flash` ya devuelve 404 para claves
    #: nuevas ("no longer available to new users").
    gemini_model: str = os.environ.get("PHYSIOVISION_GEMINI_MODELO",
                                       "gemini-flash-lite-latest")
    gemini_timeout: float = float(os.environ.get("PHYSIOVISION_GEMINI_TIMEOUT", "6"))

    # --- Historial -------------------------------------------------------- #
    #: Si es falso, la aplicación **no escribe nada en disco sobre pacientes**:
    #: ni nombres, ni series, ni resultados. Se pone a `0` en despliegues donde
    #: el almacenamiento es efímero, y no por comodidad: guardar datos de salud
    #: en un disco que se va a borrar reúne lo peor de las dos opciones, porque
    #: mientras el contenedor vive esos datos existen y pueden acabar en una
    #: instantánea o en un registro.
    guarda_historial: bool = os.environ.get(
        "PHYSIOVISION_HISTORIAL", "1").strip().lower() not in ("0", "false", "no")

    # --- Voz con Google Cloud Text-to-Speech (opcional) ---
    #: Caché de audio. Las consignas son un conjunto cerrado, así que tras la
    #: primera síntesis la reproducción es una lectura de archivo.
    audio_dir: Path = DATOS_DIR / "audio"


settings = Settings()
settings.processed_dir.mkdir(parents=True, exist_ok=True)
settings.database_path.parent.mkdir(parents=True, exist_ok=True)
settings.mediapipe_dir.mkdir(parents=True, exist_ok=True)
settings.audio_dir.mkdir(parents=True, exist_ok=True)
