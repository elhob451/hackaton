# Telegram AI Userbot

Юзербот читает последние сообщения из пользовательской папки чатов Telegram, отправляет каждую переписку в `AICore` из `main.py` и выводит саммари в консоль. Он не отправляет сообщения от имени пользователя и не работает в фоновом режиме после завершения команды.

## Настройка

1. Установите зависимости:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Создайте приложение на [my.telegram.org](https://my.telegram.org) и получите **API ID** и **API Hash**.

3. Добавьте настройки в `.env`:

   ```dotenv
   TELEGRAM_API_ID=123456
   TELEGRAM_API_HASH=your_api_hash
   OPENROUTER_API_KEY=your_openrouter_key

   # Необязательно:
   TELEGRAM_PHONE=+1234567890
   TELEGRAM_SESSION=user_bot
   OPENROUTER_MODEL=openai/gpt-4o-mini
   OPENROUTER_APP_NAME=Telegram AI Userbot
   OPENROUTER_SITE_URL=
   ```

При первом запуске Telethon попросит номер телефона (если он не задан в `.env`), код входа и пароль двухфакторной аутентификации, если он включён. Сессия сохраняется в `TELEGRAM_SESSION`; session-файл даёт доступ к аккаунту, поэтому не передавайте его никому. Такие файлы добавлены в `.gitignore`.

## Запуск

Посмотреть ID и названия пользовательских папок:

```bash
python user_bot.py --list-folders
```

Суммировать до 100 последних сообщений в каждом чате папки:

```bash
python user_bot.py --folder "Работа" --limit 100
```

Вместо названия можно передать ID:

```bash
python user_bot.py --folder 2 --limit 50
```

Поддерживаются текстовые сообщения и подписи к медиа. Содержимое переписок отправляется настроенному провайдеру через OpenRouter; саммари выводятся только в консоль.
