# Taller Práctico   Procesamiento del lenguaje natural con modelos de secuencias

Este taller permite aplicar los principales contenidos del curso mediante la construcción de una solución que analice solicitudes bancarias escritas en lenguaje natural. Los participantes utilizarán modelos de secuencias para clasificar los mensajes, identificar entidades relevantes y determinar si dos solicitudes expresan la misma necesidad, aunque estén redactadas de forma diferente. La práctica integra redes neuronales recurrentes, modelos de lenguaje neuronales, unidades LSTM, reconocimiento de entidades nombradas y redes siamesas. Estos temas corresponden a las tres unidades del curso: modelos recurrentes para lenguaje, LSTM y reconocimiento de entidades, y redes siamesas para detectar preguntas duplicadas.

## Estructura

- `CapituloXX/README.md`: guía de laboratorio por capítulo.

## Lista de laboratorios

### Capítulo 1

- [Construcción de un sistema para analizar y detectar solicitudes bancarias duplicadas](Capitulo01/README.md#construcción-de-un-sistema-para-analizar-y-detectar-solicitudes-bancarias-duplicadas)
  - Descripción: El software ya está instalado. Los participantes, usuarios de negocio, preparan solicitudes sintéticas, separan los grupos, entrenan una red siamesa y prueban el umbral de duplicado con código Python corto. La prueba queda en grupos no vistos y se compara con las palabras compartidas.
  - Duración estimada: 105 min

## Flujo de colaboración

- Trabajar en `changes_course`.
- Crear Pull Request hacia `main`.
- Merge por `Squash and merge`.
