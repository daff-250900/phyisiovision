"""Aislamiento común de las pruebas.

Desde que los usuarios viven en la base de datos, cualquier prueba que roce la
autenticación abriría `data/physiovision.db` —la de verdad, con los pacientes
de quien esté desarrollando—. Eso hace que el resultado dependa de qué haya en
esa base, y a la larga que una prueba escriba en ella.

Este fixture apunta los repositorios a un archivo temporal por prueba. Quien
necesite una base concreta la pasa por `db_path`, como hasta ahora.
"""

from __future__ import annotations

import dataclasses

import pytest


@pytest.fixture(autouse=True)
def base_aislada(tmp_path, monkeypatch, request):
    """Cada prueba, con su propia base vacía.

    Las pruebas de Postgres se saltan el aislamiento: tienen su propia URL y
    saber a qué servidor apuntan es justo lo que están comprobando.
    """
    if "postgres" in request.node.name.lower():
        yield None
        return

    from src import storage

    monkeypatch.setenv("PHYSIOVISION_BD", "")
    monkeypatch.setattr(
        storage, "settings",
        dataclasses.replace(storage.settings,
                            database_path=tmp_path / "aislada.db"))
    yield tmp_path / "aislada.db"
