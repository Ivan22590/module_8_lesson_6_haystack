# Telegram-бот-помощник на Haystack + Pinecone

Telegram-бот с умным персональным помощником (агент на фреймворке
**Haystack 3.x**). Агент сам решает, когда искать в векторной базе Pinecone,
когда изучать присланную ссылку, когда запоминать личные факты о
пользователе, когда достать факт/фото собаки, погоду — а когда просто
ответить.

## Возможности

- **`search_knowledge_base`** — семантический поиск по базе знаний, личной памяти
  пользователя и прошлым диалогам с ним;
- **`process_url_and_save`** — парсинг URL (BeautifulSoup), разбиение на чанки и сохранение в Pinecone;
- **`get_random_cat_fact`** — случайный факт о котиках с catfact.ninja;
- **`get_random_dog_fact`** — случайный факт о собаках с dogapi.dog;
- **`get_random_dog_image`** — случайное фото собаки с dog.ceo: фото скачано и
  проанализировано LLM (vision) — бот определяет породу, рассказывает её
  предысторию и характер, а саму картинку отправляет отдельным сообщением;
- **`get_weather`** — погода по названию города через бесплатное Open-Meteo
  (без ключа);
- **`save_user_memory`** — сохранение важных личных фактов с метаданными
  `user_id` и умной фильтрацией дубликатов (косинусное сходство + порог);
- **контекст диалога в Pinecone**: бот сохраняет в Pinecone **только тексты
  сообщений пользователя** (не свои ответы), с косинусным дедупом, и каждый
  новый запрос учитывает этот контекст — диалог продолжается естественно даже
  после перезапуска бота;
- история переписки по чатам, статусы «Думаю…/Читаю статью…/Проверяю погоду…»,
  понятные сообщения об ошибках.

## Структура проекта

```
.
├── pinecone_manager.py  # ЕДИНЫЙ модуль Pinecone (haystack-интеграция): чтение/запись + умная память
├── config.py            # загрузка .env и настройки (Settings)
├── service_messages.py  # служебные сообщения бота + is_service_text()
├── tools.py             # 7 инструментов агента (haystack.tools.Tool)
├── agent.py             # Haystack Agent, история диалога, контекст из Pinecone
├── bot.py               # логика Telegram-бота (PyTelegramBotAPI)
├── main.py              # точка входа (запуск бота)
├── requirements.txt
├── .env.example         # шаблон переменных окружения
└── .gitignore
```

## Архитектура

```
Пользователь → PyTelegramBotAPI (bot.py)
  → текст сообщения пользователя сохраняется в Pinecone (namespace dialog_context,
    косинусный дедуп; служебные тексты и ответы бота НЕ сохраняются)
  → Haystack Agent (agent.py) + OpenAIChatGenerator (OpenAI-совместимый провайдер)
  → перед запуском в промпт подмешивается контекст из Pinecone
    (namespace user_memories + dialog_context, поиск по косинусному сходству)
  → инструменты (tools.py): поиск, URL, факты, фото собак, погода, память
  → PineconeDocumentStore (haystack-pinecone): knowledge / user_memories / dialog_context
```

- **Хранение контекста**: `PineconeManager` (файл `pinecone_manager.py`) пишет и
  читает через `PineconeDocumentStore` (официальная интеграция haystack-pinecone),
  эмбеддинги — `OpenAITextEmbedder`/`OpenAIDocumentEmbedder` Haystack на
  OpenAI-совместимом провайдере (`OPENAI_BASE_URL`). Метрика — **cosine**.
- **Дедупликация**: перед записью менеджер считает косинусное сходство между
  вектором текста и вектором ближайшего сохранённого фрагмента (эмбеддинги
  векторов хранятся прямо в Pinecone, `include_values=True`).
- `knowledge` — статьи из ссылок; `user_memories` — личные факты о пользователе;
  `dialog_context` — тексты сообщений пользователя (контекст многодневных
  разговоров).

## Умная память (косинусное сходство)

Перед записью в долговременную память (личные факты, сообщения пользователя)
менеджер считает косинусное сходство между текущим сообщением и уже
сохранёнными фрагментами:

- `similarity < COSINE_SIMILARITY_THRESHOLD` — новая информация → `action: created`;
- `similarity >= COSINE_SIMILARITY_THRESHOLD` — дубликат/вариация →
  `DUPLICATE_ACTION = "update"` (перезапись слота, `action: updated`) либо
  `"skip"` (пропуск, `action: skipped`).

Граница и поведение задаются **глобальными переменными в начале
`pinecone_manager.py`**:

```python
COSINE_SIMILARITY_THRESHOLD = 0.75  # граница «низкого» и «высокого» сходства
DUPLICATE_ACTION = "update"         # что делать с дубликатом: "update" | "skip"
DEDUPE_ENABLED = True               # общий выключатель фильтрации
```

### Ручной тест менеджера

```bash
python pinecone_manager.py                # подключение, запись/чтение, демо дедупа
python pinecone_manager.py --delete-index # удалить (тестовый) индекс Pinecone
python pinecone_manager.py --reset-memory # очистить namespace личной памяти
python pinecone_manager.py --reset-dialog # очистить контекст диалогов
```

В логах теста подключения появляется `✅ Подключение успешно`, а демо
фильтрации дубликатов выводит `action: created|updated|skipped` с фактическим
`similarity` — по нему удобно подбирать порог:

```text
action: created | similarity=0.0000 (порог=0.75) | id=... | text='Хочу на Марс'
action: updated | similarity=0.9206 (порог=0.75) | id=... | text='Люблю программировать'
```

## Настройка

### 1. Python

Нужен Python 3.11 или новее.

### 2. Виртуальное окружение

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Переменные окружения

`.env` не хранится в git (см. `.gitignore`). Скопируйте `.env.example`
в `.env` и заполните значения:

```bash
copy .env.example .env     # Windows
cp .env.example .env       # macOS/Linux
```

| Переменная              | Обязательна | Описание                                                       |
|-------------------------|-------------|----------------------------------------------------------------|
| `TELEGRAM_TOKEN`        | да          | токен бота от [@BotFather](https://t.me/BotFather)             |
| `PINECONE_API_KEY`      | да          | ключ Pinecone (Console → API Keys)                             |
| `PINECONE_INDEX_NAME`   | да          | имя индекса (создаётся автоматически, если нет)                |
| `OPENAI_API_KEY`        | да          | ключ OpenAI-совместимого провайдера (например, ProxyAPI)       |
| `OPENAI_BASE_URL`       | нет         | базовый URL API провайдера (по умолчанию — OpenAI)             |
| `OPENAI_MODEL`          | нет         | модель чата; для ProxyAPI — с префиксом `openai/gpt-4o-mini`   |
| `EMBEDDING_MODEL`       | нет         | модель эмбеддингов; для ProxyAPI — `openai/text-embedding-3-small` |
| `EMBEDDING_DIM`         | нет         | размерность эмбеддингов (по умолчанию `1536`)                  |
| `PINECONE_REGION`       | нет         | регион serverless-индекса (`us-east-1`)                        |
| `PINECONE_CLOUD`        | нет         | облако serverless-индекса (`aws`)                              |

> Индекс Pinecone создаётся автоматически при старте (serverless, метрика
> `cosine`, размерность = `EMBEDDING_DIM`). Если меняете модель эмбеддингов —
> измените и `EMBEDDING_DIM`, либо укажите другое имя индекса.
>
> **Важно (ProxyAPI):** имена моделей указываются С префиксом провайдера
> (`openai/gpt-4o-mini`, `openai/text-embedding-3-small`) — код не переименовывает
> их, значения из `.env` передаются в API как есть.

### 4. Проверка подключения

```bash
python pinecone_manager.py   # тест менеджера: подключение, чтение/запись, дедуп
python agent.py              # короткая проверка подключения из агента
python tools.py              # факты (котики/собаки) + погода без ключей
```

В логах должно появиться `✅ ... успешно`, либо `❌ Ошибка ...` с причиной.

### 5. Запуск бота

```bash
python main.py
```

## Использование

- `/start` — приветствие и справка;
- `/help` — справка по возможностям;
- `/reset` — очистить историю диалога;
- «Изучи эту статью и запомни: `<URL>`» — агент распарсит страницу и сохранит её в базу знаний;
- «Что ты знаешь о …?» — поиск по базе знаний, личной памяти и прошлым диалогам;
- «Я живу в Питере» — факт сохранится в память пользователя;
- «Расскажи факт о котиках» / «Расскажи факт о собаке» — случайные факты;
- «Покажи фото собаки» — бот пришлёт фото, определит породу по нему и расскажет
  о происхождении и характере породы;
- «Какая погода в Москве?» — текущая погода по названию города.

## Примечания

- История переписки хранится в памяти процесса (сбрасывается при рестарте бота);
  личные факты и **контекст диалогов** хранятся в Pinecone и переживают рестарт
  — благодаря этому бот «вспоминает» разговоры и факты о пользователе.
- В Pinecone сохраняются **только тексты сообщений пользователя** (`dialog_context`),
  а также изученные статьи (`knowledge`) и личные факты (`user_memories`).
  Служебные ответы бота и статусы в Pinecone не попадают (`is_service_text`).
- Все инструменты перехватывают сетевые и БД-ошибки и возвращают понятный текст —
  агент объяснит проблему пользователю, не роняя диалог.
- Бесплатные API, которые использует бот: catfact.ninja (котики),
  dogapi.dog (факты о собаках), dog.ceo (фото собак),
  Open-Meteo (погода, геокодирование) — ключей не требуют.
