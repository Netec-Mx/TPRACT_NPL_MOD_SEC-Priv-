# Prerrequisitos del laboratorio

Estos pasos los completa el facilitador **antes** de que el participante abra el taller. Durante la sesión no se instala software. El participante solo abre Visual Studio Code, elige el kernel y ejecuta el notebook de [README.md](README.md).

Haz toda la preparación con el mismo usuario de Windows que va a usar el participante. Si creas el entorno como administrador y luego abren Visual Studio Code con otro usuario, el kernel y la carpeta no coinciden.

## Qué debe quedar listo


| Requisito     | Estado esperado                                                                  |
| ------------- | -------------------------------------------------------------------------------- |
| Equipo        | Windows Server 2019 o 2022, 64 bits, con experiencia de escritorio               |
| CPU y memoria | 4 núcleos, 8 GB de RAM (recomendado 16 GB) y 10 GB libres en el disco del perfil |
| Editor        | Visual Studio Code con las extensiones Python y Jupyter                          |
| Carpeta       | `C:\Users\<usuario>\nlp_banking_lab`                                             |
| Entorno       | `nlp_banking_lab\nlp_env`                                                        |
| Kernel        | **Python (nlp_banking_lab)**                                                     |
| Python        | 3.10, 64 bits                                                                    |
| Librerías     | Las versiones de la sección 6                                                    |


Windows Server Core, Python de 32 bits y Python 3.12 no sirven para este laboratorio. TensorFlow 2.15 se usa en CPU: no hace falta GPU.

## 1. Habilitar rutas largas

TensorFlow anida archivos por debajo de los 260 caracteres. En PowerShell **ejecutado como administrador**:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Cierra la sesión y vuelve a entrar antes de instalar las librerías. Si una política de dominio bloquea la clave, habilita *Enable Win32 long paths* en la directiva del equipo.

## 2. Instalar el runtime de Visual C++ y WebView2

1. Instala [Microsoft Visual C++ Redistributable 2015-2022 x64](https://aka.ms/vs/17/release/vc_redist.x64.exe). TensorFlow lo necesita para cargar sus DLL.
2. Instala [WebView2 Runtime Evergreen x64](https://developer.microsoft.com/microsoft-edge/webview2/) si Visual Studio Code no abre la ventana en el servidor.



## 3. Instalar Python 3.10 de 64 bits

1. Descarga el **Windows installer (64-bit)** de [Python 3.10.11](https://www.python.org/downloads/release/python-31011/) o [Python 3.10.12](https://www.python.org/downloads/release/python-31012/).
2. Marca **Add python.exe to PATH**.
3. Marca **Install launcher for all users (py)**.

Cierra PowerShell, ábrelo de nuevo y comprueba:

```powershell
py -3.10 --version
py -3.10 -c "import struct, sys; bits = struct.calcsize('P') * 8; print(bits); assert bits == 64; assert sys.version_info[:2] == (3, 10)"
```

La versión debe empezar por `Python 3.10` y el segundo comando debe imprimir `64`.

## 4. Instalar Visual Studio Code y las extensiones

1. Instala [Visual Studio Code para Windows x64](https://code.visualstudio.com/download) con el **System Installer**. Marca **Add to PATH**.
2. Abre un PowerShell nuevo:

```powershell
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
```

Si `code` no se reconoce, abre Visual Studio Code, pulsa `Ctrl+Shift+P` y ejecuta **Shell Command: Install 'code' command in PATH**. Después abre otra terminal.

## 5. Crear la carpeta del laboratorio

En PowerShell, con el usuario del participante:

```powershell
$Base = Join-Path $env:USERPROFILE "nlp_banking_lab"
$dirs = @(
  "data\raw",
  "data\processed",
  "models",
  "notebooks",
  "embeddings",
  "nltk_data"
)
foreach ($dir in $dirs) {
  New-Item -ItemType Directory -Force -Path (Join-Path $Base $dir) | Out-Null
}
Set-Location $Base
```



## 6. Crear el entorno e instalar las librerías

Sigue en la misma carpeta. El archivo de requisitos se escribe sin BOM: en PowerShell 5.1 un BOM hace que pip no reconozca la primera librería.

No instales el paquete `keras` suelto. Keras entra con TensorFlow 2.15. Gast 0.5.4 y SciPy 1.12.0 van fijos porque las versiones más nuevas que pip elegiría rompen TensorFlow o Gensim.

```powershell
$Base = Join-Path $env:USERPROFILE "nlp_banking_lab"
Set-Location $Base

$utf8NoBom = New-Object System.Text.UTF8Encoding $false
$requirements = @(
  "tensorflow==2.15.0",
  "gast==0.5.4",
  "numpy==1.26.4",
  "pandas==2.2.1",
  "scikit-learn==1.4.1.post1",
  "nltk==3.8.1",
  "gensim==4.3.2",
  "scipy==1.12.0",
  "matplotlib==3.8.3",
  "Unidecode==1.3.8",
  "ipykernel==6.29.4"
)
[System.IO.File]::WriteAllLines((Join-Path $Base "requirements.txt"), $requirements, $utf8NoBom)

py -3.10 -m venv nlp_env
& .\nlp_env\Scripts\python.exe -m pip install --upgrade "pip==24.0"
& .\nlp_env\Scripts\python.exe -m pip install -r requirements.txt
& .\nlp_env\Scripts\python.exe -m ipykernel install --user --name nlp_banking_lab --display-name "Python (nlp_banking_lab)"
```

La instalación de TensorFlow puede tardar varios minutos. El comando usa `nlp_env\Scripts\python.exe`, así que no depende de activar el entorno.

## 7. Dejar lista la lista de palabras vacías

Así el participante no necesita internet durante el taller. En la misma carpeta:

```powershell
& .\nlp_env\Scripts\python.exe -c "import nltk, os; destino = os.path.join(os.getcwd(), 'nltk_data'); nltk.download('stopwords', download_dir=destino)"
```



## 8. Comprobar que el entorno abre

```powershell
& .\nlp_env\Scripts\python.exe -c "import sys, tensorflow as tf, numpy, pandas, sklearn, nltk, gensim, scipy, matplotlib, unidecode; print(sys.executable); print('TensorFlow', tf.__version__); assert tf.__version__.startswith('2.15')"
```

Debes ver la ruta `...\nlp_banking_lab\nlp_env\Scripts\python.exe` y `TensorFlow 2.15.0`. Un aviso de deprecación al importar TensorFlow no impide usar el entorno.

## 9. Probar el kernel en Visual Studio Code

1. Abre Visual Studio Code **con el usuario del participante**, sin "Ejecutar como administrador".
2. **Archivo > Abrir carpeta** y elige `C:\Users\<usuario>\nlp_banking_lab`.
3. Confía en la carpeta.
4. Crea un notebook de prueba en `notebooks`.
5. Elige el kernel **Python (nlp_banking_lab)**. Si no aparece: `Ctrl+Shift+P` → **Python: Select Interpreter** → `nlp_banking_lab\nlp_env\Scripts\python.exe`, y vuelve a elegir el kernel.
6. Ejecuta `import tensorflow as tf; print(tf.__version__)`. Debe imprimir `2.15.0`.
7. Borra el notebook de prueba. El participante crea `notebooks\siamese_banking_duplicates.ipynb` al empezar el taller.



## Si algo falla


| Síntoma                                                                       | Qué hacer                                                                                                                                                        |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `py` no se reconoce                                                           | Vuelve a instalar Python con el py launcher y abre una terminal nueva                                                                                            |
| `python` abre Microsoft Store                                                 | Usa `py -3.10` o `.\nlp_env\Scripts\python.exe`                                                                                                                  |
| La activación del entorno dice que la ejecución de scripts está deshabilitada | No actives el entorno. Usa `nlp_env\Scripts\python.exe`                                                                                                          |
| `DLL load failed while importing _pywrap_tensorflow_internal`                 | Instala el redistribuible de Visual C++ x64 y reinicia la sesión                                                                                                 |
| `FileNotFoundError` al instalar TensorFlow                                    | Confirma que `LongPathsEnabled` quedó en 1 y que cerraste la sesión                                                                                              |
| pip no llega a PyPI por un proxy                                              | Agrega `--proxy http://host:puerto` al comando `pip install`                                                                                                     |
| El certificado TLS del proxy intercepta pip                                   | Agrega `--trusted-host pypi.org --trusted-host files.pythonhosted.org`                                                                                           |
| Visual Studio Code abre en blanco                                             | Instala WebView2 Runtime x64                                                                                                                                     |
| El kernel no aparece                                                          | Repite `python.exe -m ipykernel install --user --name nlp_banking_lab --display-name "Python (nlp_banking_lab)"` con el Python de `nlp_env` y recarga la ventana |


