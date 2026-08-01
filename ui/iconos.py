"""Iconos de la interfaz, como SVG en línea.

Dos decisiones que explican la forma de este módulo:

1. **Nada de CDN ni de archivos sueltos.** La app tiene que arrancar sin red y
   sin depender de `allowed_paths`, así que los iconos viajan dentro del CSS
   como `data:` URI.
2. **Se aplican con `mask-image`, no con `background-image`.** Una máscara usa
   solo el canal alfa del SVG y toma el color de `background-color`, de modo que
   el icono cambia de color al pasar de modo claro a oscuro sin duplicar
   archivos. Un `background-image` se quedaría con el color con el que se dibujó.

Cada nombre genera la regla `.pv-ico--<nombre>::before`, que `styles.css`
completa con el tamaño y la posición.
"""

from __future__ import annotations

from urllib.parse import quote

_ENVOLTORIO = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
    "stroke='black' stroke-width='1.8' stroke-linecap='round' "
    "stroke-linejoin='round'>{}</svg>"
)

#: Trazos de cada icono. Sin `fill`: son siluetas de línea, como en la maqueta.
TRAZOS: dict[str, str] = {
    "sesion": "<path d='M3 10.5 12 3l9 7.5'/>"
              "<path d='M5 9.5V20a1 1 0 0 0 1 1h4v-6h4v6h4a1 1 0 0 0 1-1V9.5'/>",
    "progreso": "<path d='M4 20V11'/><path d='M10 20V4'/><path d='M16 20v-6'/>"
                "<path d='M21 20H3'/>",
    "historial": "<path d='M3.2 12a8.8 8.8 0 1 0 2.8-6.4L3 8'/>"
                 "<path d='M3 3.5V8h4.5'/><path d='M12 7.5V12l3 1.8'/>",
    "ejercicios": "<path d='M9 6h11'/><path d='M9 12h11'/><path d='M9 18h11'/>"
                  "<path d='M4.5 6h.01'/><path d='M4.5 12h.01'/>"
                  "<path d='M4.5 18h.01'/>",
    "pacientes": "<circle cx='12' cy='8' r='3.5'/>"
                 "<path d='M5 20c0-3.3 3.1-6 7-6s7 2.7 7 6'/>",
    "configuracion": "<circle cx='12' cy='12' r='3.2'/>"
                     "<path d='M12 2.5v2.6M12 18.9v2.6M2.5 12h2.6M18.9 12h2.6"
                     "M5.2 5.2l1.9 1.9M16.9 16.9l1.9 1.9M18.8 5.2l-1.9 1.9"
                     "M7.1 16.9l-1.9 1.9'/>",
    "ayuda": "<circle cx='12' cy='12' r='9'/>"
             "<path d='M9.6 9.4a2.5 2.5 0 1 1 3.2 2.4c-.8.3-1.2.9-1.2 1.7v.4'/>"
             "<path d='M11.6 17.4h.01'/>",
    "salir": "<path d='M14 4h4a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-4'/>"
             "<path d='M9 16l4-4-4-4'/><path d='M13 12H3'/>",
    "plegar": "<rect x='3' y='4' width='18' height='16' rx='2.5'/>"
              "<path d='M9.5 4v16'/>",
    "claro": "<circle cx='12' cy='12' r='4'/>"
             "<path d='M12 2v2.2M12 19.8V22M2 12h2.2M19.8 12H22M5.1 5.1l1.6 1.6"
             "M17.3 17.3l1.6 1.6M18.9 5.1l-1.6 1.6M6.7 17.3l-1.6 1.6'/>",
    "oscuro": "<path d='M20.5 14.6A8.6 8.6 0 0 1 9.4 3.5a8.6 8.6 0 1 0 11.1 11.1z'/>",
    "iniciar": "<path d='M7.5 4.8 19 12 7.5 19.2z'/>",
    "terminar": "<rect x='6.5' y='6.5' width='11' height='11' rx='2'/>",
    # El logotipo es una figura corriendo, como en la maqueta. Va con trazo mas
    # grueso porque se dibuja a 26 px y con 1.8 se veria anemico.
    "logo": "<circle cx='14.5' cy='4.6' r='2.1' fill='black' stroke='none'/>"
            "<path d='M15.4 9.1 11 11.4l1.9 3.3-1.4 5.9' stroke-width='2.4'/>"
            "<path d='M12.9 14.7l4.3 1.5 1.6 4.4' stroke-width='2.4'/>"
            "<path d='M11 11.4 6.6 9.9' stroke-width='2.4'/>",
}


def _uri(trazos: str) -> str:
    return "data:image/svg+xml," + quote(_ENVOLTORIO.format(trazos), safe="")


def reglas_css() -> str:
    """Una regla por icono, más las variables que necesitan los iconos que
    cambian con el estado (el del botón de tema).
    """
    reglas = []
    for nombre, trazos in TRAZOS.items():
        uri = _uri(trazos)
        reglas.append(
            f'.pv-ico--{nombre}::before {{'
            f' -webkit-mask-image: url("{uri}");'
            f' mask-image: url("{uri}"); }}'
        )
    # El botón de tema alterna entre sol y luna sin cambiar de clase, así que
    # sus dos iconos viajan como variables y los intercambia `styles.css`.
    reglas.append(
        ":root {\n"
        f'  --pv-ico-claro: url("{_uri(TRAZOS["claro"])}");\n'
        f'  --pv-ico-oscuro: url("{_uri(TRAZOS["oscuro"])}");\n'
        "}"
    )
    return "\n".join(reglas)
