# PhysioVision en Google Cloud Run

Cuatro guiones, listos para usar. Por defecto el servicio se despliega **sin
historial**; con `cloudsql.sh` se le añade persistencia de verdad (ver más
abajo).

```
deploy/cloudrun/
  secretos.sh      sube usuarios y clave de Gemini a Secret Manager
  cloudsql.sh      crea la base gestionada y su secreto (opcional)
  desplegar.sh     construye en Cloud Build y despliega el servicio
  aceptacion.sh    comprueba lo desplegado
```

## Antes de empezar

Hace falta `gcloud` instalado y un proyecto con facturación activa. En este
equipo no está: `brew install --cask google-cloud-sdk` y `gcloud auth login`.

## Los tres pasos

```bash
python -m src.auth dafne                       # si aún no hay usuarios
./deploy/cloudrun/secretos.sh   <proyecto>
./deploy/cloudrun/desplegar.sh  <proyecto> [region]
./deploy/cloudrun/aceptacion.sh <url> dafne '<contraseña>'
```

La imagen se construye en **Cloud Build**, no en local: es `amd64` y en un Mac
con Apple Silicon habría que emularla, además de subir 1 GB por la red de casa.

## Por qué estas opciones y no las de fábrica

Cada una responde a una medida de este proyecto:

| Opción | Por qué |
|---|---|
| `--cpu 4 --memory 4Gi` | una sesión satura un núcleo (48 fps nativos aquí, 30 necesarios) y ocupa ~0,7 GB |
| `--no-cpu-throttling` | por defecto solo hay CPU durante la petición; la redacción de Gemini y la voz corren en hilos de fondo y se quedarían a medias |
| `--concurrency 2` | con el valor por defecto (80) el autoescalado mete varias sesiones en una instancia y ninguna llega a 30 Hz |
| `--timeout 3600` | el vídeo va por una conexión larga; con el valor por defecto se corta la sesión a mitad de serie. 3600 s es el techo del servicio |
| `--session-affinity` | la sesión vive en la memoria del proceso; sin afinidad, una petición puede aterrizar en otra instancia donde no existe |
| `--max-instances 2` | **techo de gasto**. Cada instancia cuesta CPU y tráfico mientras vive |

## Historial: dos modos, y el despliegue elige solo

**Sin Cloud SQL** (por defecto), el servicio arranca con
`PHYSIOVISION_HISTORIAL=0`: no escribe **nada** de pacientes en disco, ni
siquiera crea la base. No es una carencia disimulada — el disco de Cloud Run es
efímero, así que guardar y perder reuniría lo peor de las dos opciones: mientras
la instancia vive, esos datos existen y pueden acabar en una instantánea o en un
registro.

**Con Cloud SQL**, el historial se guarda de verdad:

```bash
./deploy/cloudrun/cloudsql.sh <proyecto> [region]     # ~10 min la primera vez
CLOUDSQL_INSTANCIA=<proyecto:region:instancia> \
  ./deploy/cloudrun/desplegar.sh <proyecto> [region]
```

El primer guion crea la instancia, la base y el usuario, genera una clave
aleatoria y deja la **URL completa en Secret Manager** (no la imprime). El
segundo detecta la variable `CLOUDSQL_INSTANCIA` y añade la conexión, el secreto
y `PHYSIOVISION_HISTORIAL=1`.

La aplicación elige motor por la variable `PHYSIOVISION_BD`:

| `PHYSIOVISION_BD` | Motor | Dónde se usa |
|---|---|---|
| sin definir | SQLite en `data/` | local y clínica |
| `postgresql://…` | PostgreSQL | Cloud Run con Cloud SQL |

Las mismas pruebas se ejecutan contra los dos motores (`tests/test_storage.py`),
porque tener dos deja de valer en cuanto uno se comporta distinto.

Cloud Run habla con Cloud SQL por **socket Unix**, no por TCP, de ahí la forma
de la URL: `postgresql://usuario:clave@/base?host=/cloudsql/proyecto:region:instancia`.

Lo que añade a la factura: la instancia más pequeña (`db-f1-micro`, 10 GB) se
paga por estar encendida, no por uso. Son unas pocas filas por serie, así que el
tamaño sobra de largo.

## Lo que cuesta

Órdenes de magnitud con tarifas de referencia; cambian por región y con el
tiempo, así que confírmalos en la calculadora.

| Escenario | Coste aproximado |
|---|---|
| Capa gratuita mensual | ~180.000 vCPU-s ≈ **75 sesiones de 10 min** |
| Sesión de 10 min, pasada la capa gratuita | ~0,06 $ cómputo + ~0,04 $ tráfico |
| Siempre encendida (`--min-instances 1`) | **~210 $/mes** antes de tráfico |

El **tráfico de salida** pesa más de lo que parece porque la app devuelve el
vídeo anotado. Medido sobre un fotograma real, a 30 Hz: 4,7 GB/hora a 1920 px.
Por eso la aplicación ahora **devuelve 960 px** (`PHYSIOVISION_ANCHO_SALIDA`) y
baja a 1,8 GB/hora, sin diferencia visible: la zona de vídeo en pantalla mide
unos 800 px.

## Abierta a todo el mundo

`--allow-unauthenticated` abre la URL, no la aplicación: el login sigue delante,
y es justamente lo que impide que cualquiera queme CPU y tráfico. Aun así:

- Deja `--max-instances` bajo. Es el único techo de gasto real.
- Cada sesión activa cuesta ~0,4 $/hora entre cómputo y tráfico.
- Si de verdad quieres una demo sin credenciales, ten presente que sin historial
  no queda rastro en disco, pero el vídeo sí pasa por el servicio mientras dura
  la sesión.

## Volver atrás

Cada despliegue crea una revisión. Volver es instantáneo y no reconstruye nada:

```bash
gcloud run revisions list --service physiovision --region <region>
gcloud run services update-traffic physiovision --region <region> \
    --to-revisions <revision-anterior>=100
```
