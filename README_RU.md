# DMH-Net — запуск инференса
Инструкция по запуску предобученной модели DMH-Net для оценки планировки комнаты по панорамному изображению.

* Репозиторий: https://github.com/Starrah/DMH-Net

* Статья: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610630.pdf


# Требования к системе
* Windows 10/11 (или Linux)
* Python 3.7.9:
  
    Установить python: https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe

    GPU не обязателен — инференс работает на CPU;

    ~4 ГБ диска (окружение + веса).
* Проверить наличие Python 3.7.9:
  
    ```powershell
    py -0
    ```

    ```powershell
    py -3.7 --version
    ```

    Если нет — установить:

    ```powershell
    winget install Python.Python.3.7
    ```

    После установки открыть новое окно терминала и проверить:

    ```powershell
    py -3.7 --version
    ```
# Запуск
* Клонировать репозитория
    ```powershell
    git clone [ссылка]
    ```

* Создать виртуальное окружение

    ```powershell
    py -3.7 -m venv venv
    .\venv\Scripts\python.exe
    ```
* Проверить окружение
    ```powershell
    .\venv\Scripts\python.exe --version
    ```
* Обновить pip
    ```powershell
    .\venv\Scripts\python.exe -m pip install --upgrade pip
    ```
* Установить PyTorch 1.9.1 (CPU-версия)
    ```powershell
    .\venv\Scripts\python.exe -m pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
    ```
* Для GPU (NVIDIA, CUDA >= 11.1, проверить nvidia-smi):
    ```powershell 
    .\venv\Scripts\python.exe -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
    ```
* Проверить установку
    ```powershell 
    .\venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
    ```
* Установить остальные зависимости
    ```powershell
    .\venv\Scripts\python.exe -m pip install -r requirements.txt
    ```
* Проверить целостность окружения
    ```powershell
    .\venv\Scripts\python.exe -m pip check
    ```
* Скачать .pth файлы по ссылке из README репозитория и положить в папку ckpt\.
* Создать структуру папок
    ```powershell
    New-Item -ItemType Directory -Force -Path data\layoutnet_dataset\test\img
    New-Item -ItemType Directory -Force -Path data\layoutnet_dataset\test\label_cor
    ```
* Запуск подготовки
    ```powershell
    .\venv\Scripts\python.exe prepare_pano.py путь\к\панораме.jpg
    ```
* Перед каждым новым окном терминала установить кодировку
    ```powershell
    $env:PYTHONUTF8 = "1"
    ```
* Команда запуска (пример для panocontext)
    ```powershell
    .\venv\Scripts\python.exe eval.py --cfg_file cfgs/panocontext.yaml --ckpt ".\ckpt\panocontext_v1 (1).pth" --input_file pano.jpg --no_cuda --num_workers 0 --visu_all --visu_path "D:\temp\dmh_visu" --visu_type "['erw']" --save_json
    ```
    Параметры:

    --cfg_file — конфиг, парный к чекпоинту;

    --input_file — ИМЯ файла из data\layoutnet_dataset\test\img (без пути);

    --no_cuda — только для CPU-версии PyTorch (для GPU убрать);

    --num_workers 0 — обязательно на Windows;

    --visu_type "['erw']" — отрисовка на панораме: e = панорама,r = точки углов, w = контур раскладки; формат строго списком в кавычках;

    --save_json — сохранить координаты в result_json.
* Трехмерная визуализация
    ```powershell
    .\venv\Scripts\python.exe layout_viewer.py --img "путь\к\оригинальной_панораме.jpg" --layout ".\result_json\pano.jpg.json"
    ```
    Флаги: --ignore_ceiling, --ignore_floor, --ignore_wireframe, --ppm 120.
