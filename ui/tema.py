"""Paleta y tema de la interfaz.

Fuente única de los colores. De aquí salen las dos cosas que tienen que ir
coordinadas:

- las variables CSS que usan las reglas de `styles.css`;
- el tema de los componentes propios de Gradio (botones, tablas, desplegables),
  que no se pueden estilar desde nuestro CSS sin pelearse con el suyo.

Cambiar un color aquí lo cambia en los dos sitios. Cada modo se define entero,
en vez de derivar el oscuro del claro: el verde de la maqueta (`#22C55E`) sobre
fondo blanco no llega a contraste AA para texto, así que el modo claro usa una
variante más oscura del mismo tono.
"""

from __future__ import annotations

import gradio as gr

#: Tipografías del sistema. No se usa `GoogleFont`: descargar la fuente de un
#: CDN filtra la visita a un tercero y deja la interfaz a merced de la red.
FUENTES = ("system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif")

PALETA: dict[str, dict[str, str]] = {
    "claro": {
        "fondo": "#F4F6F9",
        "superficie": "#FFFFFF",
        "superficie-2": "#ECF0F5",
        "superficie-3": "#E2E8F0",
        "borde": "#D8DFE7",
        "texto": "#0F172A",
        "texto-tenue": "#55606E",
        "acento": "#15803D",
        "acento-fuerte": "#166534",
        "acento-suave": "#DCFCE7",
        "atencion": "#B45309",
        "atencion-suave": "#FEF3C7",
        "peligro": "#DC2626",
        "sombra": "0 1px 2px rgba(15, 23, 42, .06), 0 8px 24px rgba(15, 23, 42, .06)",
    },
    "oscuro": {
        "fondo": "#070A0E",
        "superficie": "#111820",
        "superficie-2": "#18212B",
        "superficie-3": "#22303D",
        "borde": "#1E2A36",
        "texto": "#E6EDF3",
        "texto-tenue": "#93A1B0",
        "acento": "#22C55E",
        "acento-fuerte": "#4ADE80",
        "acento-suave": "#0D2A1A",
        "atencion": "#F59E0B",
        "atencion-suave": "#2A1F07",
        "peligro": "#EF4444",
        "sombra": "0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px rgba(0, 0, 0, .35)",
    },
}


def variables_css() -> str:
    """Bloques `:root` y `body.dark` con la paleta.

    Gradio marca el modo oscuro añadiendo la clase `dark` a `document.body`, así
    que basta con colgar de ahí las sobrescrituras: el mismo interruptor sirve
    para sus componentes y para los nuestros.
    """
    def bloque(selector: str, colores: dict[str, str]) -> str:
        cuerpo = "\n".join(f"  --pv-{nombre}: {valor};"
                           for nombre, valor in colores.items())
        return f"{selector} {{\n{cuerpo}\n}}"

    return "\n".join([bloque(":root", PALETA["claro"]),
                      bloque("body.dark", PALETA["oscuro"])])


def crear_tema() -> gr.themes.Base:
    """Tema de Gradio alineado con la paleta.

    Los sufijos `_dark` son el mecanismo nativo de Gradio para el modo oscuro:
    los aplica con la misma clase `dark` del `body`, de modo que el cambio de
    tema no necesita recargar la página.
    """
    claro, oscuro = PALETA["claro"], PALETA["oscuro"]
    return gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        secondary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.slate,
        font=list(FUENTES),
    ).set(
        body_background_fill=claro["fondo"],
        body_background_fill_dark=oscuro["fondo"],
        body_text_color=claro["texto"],
        body_text_color_dark=oscuro["texto"],
        body_text_color_subdued=claro["texto-tenue"],
        body_text_color_subdued_dark=oscuro["texto-tenue"],
        background_fill_primary=claro["superficie"],
        background_fill_primary_dark=oscuro["superficie"],
        background_fill_secondary=claro["superficie-2"],
        background_fill_secondary_dark=oscuro["superficie-2"],
        block_background_fill=claro["superficie"],
        block_background_fill_dark=oscuro["superficie"],
        block_border_color=claro["borde"],
        block_border_color_dark=oscuro["borde"],
        block_label_background_fill=claro["superficie-2"],
        block_label_background_fill_dark=oscuro["superficie-2"],
        block_label_text_color=claro["texto-tenue"],
        block_label_text_color_dark=oscuro["texto-tenue"],
        block_title_text_color=claro["texto"],
        block_title_text_color_dark=oscuro["texto"],
        border_color_primary=claro["borde"],
        border_color_primary_dark=oscuro["borde"],
        input_background_fill=claro["superficie"],
        input_background_fill_dark=oscuro["superficie-2"],
        input_border_color=claro["borde"],
        input_border_color_dark=oscuro["borde"],
        button_primary_background_fill=claro["acento"],
        button_primary_background_fill_dark=oscuro["acento"],
        button_primary_background_fill_hover=claro["acento-fuerte"],
        button_primary_background_fill_hover_dark=oscuro["acento-fuerte"],
        button_primary_text_color="#FFFFFF",
        button_primary_text_color_dark="#04120A",
        button_secondary_background_fill=claro["superficie-2"],
        button_secondary_background_fill_dark=oscuro["superficie-2"],
        button_secondary_text_color=claro["texto"],
        button_secondary_text_color_dark=oscuro["texto"],
        panel_background_fill=claro["superficie"],
        panel_background_fill_dark=oscuro["superficie"],
        table_odd_background_fill=claro["superficie-2"],
        table_odd_background_fill_dark=oscuro["superficie-2"],
        block_radius="14px",
        container_radius="14px",
        input_radius="10px",
        button_large_radius="10px",
        button_small_radius="8px",
    )
