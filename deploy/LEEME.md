# Despliegue de PhysioVision

La aplicación detrás de un proxy con TLS, con la base de pacientes en un volumen
y los secretos fuera de la imagen.

```
deploy/
  docker-compose.yml   la aplicación + el proxy
  Caddyfile            TLS y cabeceras
  .env.ejemplo         plantilla del entorno (el .env real no se versiona)
  secretos/            cuenta de servicio de Cloud TTS (no se versiona)
  comprobar.sh         revisión previa
  copia.sh             copia de la base, en caliente
```

## Puesta en marcha

```bash
cp deploy/.env.ejemplo deploy/.env
python -m src.auth dafne          # y pega la línea en PHYSIOVISION_USUARIOS
$EDITOR deploy/.env

./deploy/comprobar.sh             # revisión previa
docker compose -f deploy/docker-compose.yml up -d --build
```

Con `DOMINIO=midominio.org` el certificado se emite solo con Let's Encrypt, y
hacen falta los puertos 80 y 443 abiertos y el DNS ya apuntando.

## Por qué está montado así

**El TLS no es un extra.** Sin contexto seguro, `getUserMedia` no abre la cámara
y la sesión no arranca nunca. Por eso la aplicación **no publica su puerto**:
solo se llega a ella por el proxy.

**Los usuarios viajan por entorno, no en la imagen.** En `docker inspect` se ve
el resumen derivado (`pbkdf2_sha256$…`), nunca la contraseña. Si falta
`PHYSIOVISION_USUARIOS`, la app se niega a arrancar: escucha en `0.0.0.0` y
arrancar abierta enseñaría el historial de pacientes a quien pase.

**Los secretos se montan como carpeta, no como archivo.** Si se monta
`secretos/google.json` y el archivo no existe, Docker crea un **directorio** con
ese nombre y la ruta de credenciales apunta a una carpeta. Montando la carpeta,
la ausencia del archivo es solo ausencia: la app arranca sin voz, como está
previsto.

**El archivo de entorno se llama `.env` a propósito.** Compose lo usa para dos
cosas a la vez: las variables del contenedor y la interpolación del propio YAML
(dominio y puertos). Con cualquier otro nombre lo segundo no ocurre, y los
puertos se quedan en los valores por defecto sin avisar.

## Copias de seguridad

```bash
./deploy/copia.sh                 # deja deploy/copias/physiovision-<fecha>.db
```

Usa la API de copia de SQLite y no `cp`: copiar el archivo mientras la
aplicación escribe puede dejar una base a medias, y eso no se nota hasta que hace
falta restaurarla. Al terminar abre la copia y cuenta las series, porque una
copia que no se ha probado no es una copia.

Restaurar:

```bash
docker compose -f deploy/docker-compose.yml stop app
docker compose -f deploy/docker-compose.yml cp deploy/copias/<archivo> \
    app:/app/data/physiovision.db
docker compose -f deploy/docker-compose.yml start app
```

Automatizar la copia diaria es una línea de `cron`; guardar las copias en otra
máquina, otra. Las dos hacen falta.

## Medir la tasa antes de abrir a nadie

```bash
./deploy/medir.sh <video con una persona.mp4>
```

El modelo se entrenó a 30 Hz y sus variables temporales se desplazan con el
muestreo: a 10 Hz, una repetición de cada cuatro cambia de clase. Esto no es una
métrica de confort, es una condición de validez, y por eso el guion da un
veredicto en vez de un número suelto.

Medido en este equipo: **48 fps** nativo (macOS, Apple Silicon) y **10,1 fps**
dentro del contenedor `linux/amd64` **emulado**, que es lo que hay al construir
para x86 sobre un Mac ARM. La emulación cuesta unas cinco veces; en un x86 de
verdad la cifra volverá a subir, pero **hay que medirla allí**.

## Vigilar

```bash
./deploy/vigilar.sh [minutos]
```

Cuenta en el registro las cuatro señales que delatan una degradación silenciosa
—la app sigue respondiendo, solo que peor—: tasa de cámara desviada, series con
poca cobertura de pose, clasificación caída a reglas y fallos de Gemini o de voz.
Más la actividad: series terminadas y accesos.

El registro **no lleva nombres de pacientes**: sale de la máquina y no es sitio
para datos identificables.

## Aceptación y vuelta atrás

```bash
./deploy/aceptacion.sh <usuario> <contraseña>
```

Nueve comprobaciones sobre lo desplegado: HTTPS, healthcheck, que la app no
publique su puerto, que el login rechace y acepte, que se sirva la interfaz, que
clasifique con el modelo y no con reglas, que la base responda y que MediaPipe
procese fotogramas.

Cada despliegue va con su etiqueta, y volver atrás es una línea:

```bash
PHYSIOVISION_TAG=<etiqueta anterior> docker compose -f deploy/docker-compose.yml up -d --no-build app
```

Probado: cambiando la etiqueta, la app vuelve a levantar en 20 s y **los datos
siguen ahí**, porque viven en el volumen y no en la imagen.

## Certificado en red local

Con `DOMINIO=localhost` o una IP, Caddy emite el certificado con su CA interna.
Hay que instalar su raíz en cada equipo que abra la aplicación:

```bash
docker compose -f deploy/docker-compose.yml cp \
    proxy:/data/caddy/pki/authorities/local/root.crt .
```

Sin instalarla, el navegador avisa del certificado, y ese aviso no es cosmético:
si no se acepta, no hay origen seguro y la cámara no arranca.

> Al probar con puertos no estándar (`PUERTO_HTTPS=8443`), la redirección desde
> HTTP pierde el puerto y lleva a `https://localhost/`. En producción, con 80 y
> 443, no ocurre.

## Verificado en esta configuración

| Prueba | Resultado |
|---|---|
| HTTPS con la CA interna | 200, emisor `Caddy Local Authority` |
| Redirección HTTP → HTTPS | 308 |
| Cabeceras | HSTS y `Permissions-Policy: camera=(self)` |
| Login a través del proxy | clave mala 400, buena 200, app servida |
| La app no expone su puerto | solo `7860/tcp` interno |
| Persistencia | los pacientes sobreviven a `restart` |
| Copia en caliente | 12 KB, 2 series, restaurada y comprobada |
| Sin `secretos/google.json` | la app sigue en pie, sin voz |
