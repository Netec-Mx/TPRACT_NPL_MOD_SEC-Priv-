# Taller Práctico   Procesamiento del lenguaje natural con modelos de secuencias

Este taller permite aplicar los principales contenidos del curso mediante la construcción de una solución que analice solicitudes bancarias escritas en lenguaje natural. Los participantes utilizarán modelos de secuencias para clasificar los mensajes, identificar entidades relevantes y determinar si dos solicitudes expresan la misma necesidad, aunque estén redactadas de forma diferente. La práctica integra redes neuronales recurrentes, modelos de lenguaje neuronales, unidades LSTM, reconocimiento de entidades nombradas y redes siamesas. Estos temas corresponden a las tres unidades del curso: modelos recurrentes para lenguaje, LSTM y reconocimiento de entidades, y redes siamesas para detectar preguntas duplicadas.

## Estructura

- `CapituloXX/README.md`: guía de laboratorio por capítulo.

## Lista de laboratorios

### Capítulo 1

- [Construcción de un sistema para analizar y detectar solicitudes bancarias duplicadas](Capitulo01/README.md#construcción-de-un-sistema-para-analizar-y-detectar-solicitudes-bancarias-duplicadas)
  - Descripción: Los participantes trabajarán en un notebook con mensajes bancarios sintéticos. Utilizarán un modelo de secuencias para analizar los textos, extraer entidades como productos, canales o tipos de operación y comparar pares de solicitudes mediante una red siamesa. Finalmente, probarán el sistema con nuevos mensajes y revisarán los casos clasificados como duplicados.
  - Duración estimada: 105 min

## Flujo de colaboración

- Trabajar en `changes_course`.
- Crear Pull Request hacia `main`.
- Merge por `Squash and merge`.
