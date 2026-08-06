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
python -m src.auth crear dafne --rol fisioterapeuta   # si aún no hay cuentas
./deploy/cloudrun/secretos.sh   <proyecto>
./deploy/cloudrun/desplegar.sh  <proyecto> [region]
./deploy/cloudrun/aceptacion.sh <url> dafne '<contraseña>'
```

`secretos.sh` sube las cuentas con `python -m src.auth exportar`, que une la
tabla `usuarios` y el archivo heredado. Sube el **resumen** derivado, nunca la
contraseña.

La imagen se construye en **Cloud Build**, no en local: es `amd64` y en un Mac
con Apple Silicon habría que emularla, además de subir 1 GB por la red de casa.

## Por qué estas opciones y no las de fábrica

Cada una responde a una medida de este proyecto:

| Opción | Por qué |
|---|---|
| `--cpu 4 --memory 4Gi` | una sesión satura un núcleo (48 fps nativos aquí, 30 necesarios) y ocupa ~0,7 GB |
| `--no-cpu-throttling` | por defecto solo hay CPU durante la petición; la redacción de Gemini y la voz corren en hilos de fondo y se quedarían a medias |
| `--concurrency 20` | **no limita sesiones, limita peticiones HTTP.** Con 2, un navegador que pide en paralelo las decenas de CSS y JS de Gradio recibía 429 en la mitad y la página se pintaba sin el componente de cámara. Quien protege los 30 Hz es la cola de Gradio (`default_concurrency_limit=4`), no este número |
| `--timeout 3600` | el vídeo va por una conexión larga; con el valor por defecto se corta la sesión a mitad de serie. 3600 s es el techo del servicio |
| `--session-affinity` | la sesión vive en la memoria del proceso; sin afinidad, una petición puede aterrizar en otra instancia donde no existe |
| `--max-instances 1` | **no es techo de gasto, es corrección.** Ver abajo |

### Por qué una sola instancia

Gradio guarda en la memoria del proceso las dos cosas que sostienen una sesión:
el token de acceso (`app.tokens`) y los eventos de la cola
(`_queue.event_ids_to_events`). Con varias instancias, una petición que aterriza
en la que no es devuelve `401` en `/gradio_api/queue/join` o revienta con

    KeyError en gradio/routes.py:1093 -> event_ids_to_events[event_id]

y el navegador recarga. Eso es lo que se veía como «la app se reinicia a las tres
repeticiones»: no era una caída del contenedor —no hubo ni un OOM ni un reinicio
en los registros— sino la sesión aterrizando en otra instancia. `--session-affinity`
es «best effort» y no basta.

Con techo de 1 no hay ninguna otra instancia a la que ir. El precio es que no
escala horizontalmente: para eso habría que sacar sesión y cola fuera del proceso
(Redis o equivalente), que es harina de otro costal. `--min-instances 0` se
mantiene, así que sigue sin costar nada en reposo.

### Por qué la cámara iba a 4 fps

No era el modelo ni la CPU: durante una sesión real el contenedor estaba al 1,5 %
de uso. Gradio envía cada frame como `canvas.toDataURL("image/jpeg")` **al tamaño
nativo de la cámara** y en base64, y no captura el siguiente hasta que vuelve el
anterior; a 1280x720 son ~160 KB por frame, que sobre una subida doméstica de
~1 MB/s dan 156 ms y, con ellos, 4-6 fps por mucho que la app pida 30.

El arreglo está en la app, no aquí: `RESOLUCION_CAMARA` en `ui/app_ui.py` acota la
cámara a 640x480 (~76 KB, ~13 fps). Se puede regular sin tocar código con
`PHYSIOVISION_CAMARA_ANCHO` y `PHYSIOVISION_CAMARA_ALTO` si la conexión del sitio
manda otra cosa.

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

Las mismas pruebas se ejecutan contra los dos motores —`test_storage.py`,
`test_perfiles.py` y `test_migracion.py`—, porque tener dos deja de valer en
cuanto uno se comporta distinto. Para incluir PostgreSQL hace falta levantarlo:

```bash
docker run -d --name pv-pg -e POSTGRES_PASSWORD=prueba \
    -e POSTGRES_DB=physiovision -p 55432:5432 postgres:16-alpine
PHYSIOVISION_BD_PRUEBA=postgresql://postgres:prueba@127.0.0.1:55432/physiovision \
    python -m pytest tests/
```

Sin esa variable los casos de PostgreSQL se omiten, y el número de pruebas baja
de 318 a 281: **conviene ejecutarlas con Postgres antes de desplegar**, porque
los índices únicos van sobre expresiones (`lower(usuario)`) y ahí los dos
motores no tienen por qué coincidir.

Cloud Run habla con Cloud SQL por **socket Unix**, no por TCP, de ahí la forma
de la URL: `postgresql://usuario:clave@/base?host=/cloudsql/proyecto:region:instancia`.

### Perfiles: el modo sin Cloud SQL solo admite fisioterapeutas

No es una limitación arbitraria, es la consecuencia de dónde vive cada cosa:

| | Sin Cloud SQL | Con Cloud SQL |
|---|---|---|
| Cuentas | solo las del secreto | las del secreto **y** las creadas en la app |
| Perfil de las del secreto | `fisioterapeuta` | `fisioterapeuta` (las migra el primer arranque) |
| Perfil `paciente` | **no es posible** | sí |
| Fichas de paciente e historial | no se guardan | se guardan |

Un usuario que solo existe en el secreto se trata como fisioterapeuta: es lo que
era antes de que hubiera perfiles, cuando quien manejaba la aplicación era el
profesional. Y un paciente sin historial que consultar tampoco tendría mucho que
hacer dentro, así que las dos cosas van juntas.

Las cuentas de paciente se crean **contra la base ya desplegada**, con el proxy
de Cloud SQL delante:

```bash
cloud-sql-proxy <proyecto:region:instancia> &
PHYSIOVISION_BD='postgresql://physiovision:<clave>@127.0.0.1:5432/physiovision' \
    python -m src.auth crear marta --rol paciente \
                       --paciente "Marta Gómez" --fisio dafne
```

La clave está en Secret Manager (`physiovision-bd`), que es donde la dejó
`cloudsql.sh`; no se imprime en ningún sitio.

### Llevarte los datos que ya tienes

El secreto solo transporta `usuario:resumen`. Si te limitas a desplegar, el
primer arranque crea esas cuentas **como fisioterapeutas** y la base sale vacía:
se pierden los perfiles de paciente, las fichas y todo el historial.

Para subir la base local tal cual, con el proxy de Cloud SQL delante:

```bash
cloud-sql-proxy <proyecto:region:instancia> &
python -m src.migracion copiar \
    --a 'postgresql://physiovision:<clave>@127.0.0.1:5432/physiovision'
```

Copia `usuarios`, `pacientes` y `sessions` **conservando los identificadores**,
porque `pacientes.fisio_id` y `sessions.paciente_id` apuntan a ellos y
renumerar obligaría a reescribir las referencias. Después coloca las secuencias
de PostgreSQL detrás del último `id`: sin eso, el primer INSERT en producción
chocaría con la fila 1 y la aplicación dejaría de poder guardar series.

Se niega a escribir si el destino ya tiene datos. `--forzar` lo vacía antes,
y es lo que hay que usar para repetir la copia.

Comprueba el resultado:

```bash
PHYSIOVISION_BD='postgresql://…' python -m src.migracion estado
```

### La migración corre en cada arranque

`app.py` llama a `migrar_en_arranque()` antes de construir la interfaz. Es
idempotente, así que en el arranque número mil no hace nada, y **nunca impide
arrancar**: si falla, lo anota en el registro y la aplicación sigue. Quedarse sin
servicio por una migración sería peor que servir con los perfiles a medio
poblar, porque al otro lado hay un paciente esperando delante de la cámara.

Con `PHYSIOVISION_HISTORIAL=0` —el modo por defecto sin Cloud SQL— ni siquiera
se ejecuta: no hay base que migrar.

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

- Deja `--max-instances` bajo. Además de corregir el reinicio de sesión, es el
  único techo de gasto real.
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
