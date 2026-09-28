# reddit-telegram-digest

سرویس پایتونی MVP که از فیدهای RSS ساب‌ردیت‌های تعریف‌شده پست جدید می‌گیرد، با **یک تماس LLM**
هم‌زمان ربط، شباهت با پست‌های اخیر، موضوع، اهمیت، خلاصه فارسی و نکات کلیدی را استخراج می‌کند،
نتیجه را در Postgres ذخیره می‌کند و پست‌های مناسب را به یک کانال/چت ثابت تلگرام broadcast می‌کند.

> **منبع حقیقت این پروژه [`AGENTS.md`](./AGENTS.md) است.** این README فقط راهنمای سریع راه‌اندازی
> است و عمداً BRD/FR/NFR/Invariantها را تکرار نمی‌کند (DRY).

## معماری در یک نگاه

```
config/topics.yaml ─► reddit_source (RSS) ─► repository (بررسی تکراری در Postgres)
        ─► analyzer + llm_client (یک تماس LLM: ربط/شباهت/موضوع/اهمیت/خلاصه/نکات)
        ─► repository (ذخیره) ─► formatting + telegram_notifier (ارسال) ─► status=sent
```

هر اجرا ابتدا پست‌های باقی‌مانده با `status='to_send'` از اجرای قبلی را دوباره ارسال می‌کند
(بازیابی پس از کرش، FR-11) و سپس فیدها را واکشی می‌کند.

## راه‌اندازی

```bash
cp .env.example .env
# مقداردهی OPENAI_API_KEY, OPENAI_MODEL (و در صورت نیاز OPENAI_BASE_URL)،
# TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID در .env
# (اختیاری) تکمیل فهرست واقعی موضوعات/فیدها در config/topics.yaml

docker compose up --build
```

سرویس `db` (Postgres 16) در اولین بوت `db/schema.sql` را اعمال می‌کند و سرویس `worker`
هر `POLL_INTERVAL_SECONDS` ثانیه یک دور کامل پایپ‌لاین اجرا می‌کند.

## تست‌ها

```bash
pip install -r requirements.txt
pytest
```

تست‌ها هیچ تماس شبکه‌ای (RSS/LLM/Telegram) نمی‌زنند؛ به‌جای آن‌ها fake تزریق می‌شود. تست‌های `repository` عمداً integration هستند و روی یک Postgres واقعی اجرا می‌شوند:

```bash
docker compose up -d db   # دیتابیس محلی؛ اگر در دسترس نباشد این تست‌ها با پیام روشن skip می‌شوند
pytest
```

## ساختار پروژه

```
app/
  main.py               حلقه اجرای دوره‌ای
  settings.py           تنظیمات env
  models.py             مدل‌های Pydantic (اعتبارسنجی خروجی LLM)
  reddit_source.py      مالک config/topics.yaml + واکشی/parse فید RSS
  repository.py         تمام پرس‌وجوهای Postgres
  llm_client.py         wrapper نازک روی SDK سازگار با OpenAI
  prompts/              متن prompt، جدا از کد
  analyzer.py           ساخت prompt + اعتبارسنجی خروجی LLM
  formatting.py         تبدیل رکورد به پیام فارسی تلگرام
  telegram_notifier.py  ارسال sendMessage
  retry.py              retry/backoff مشترک
  pipeline.py           هماهنگ‌سازی همه مراحل
db/schema.sql           تنها منبع تغییر schema
config/topics.yaml      موضوعات مجاز + فیدها
tests/                  تست‌های واحد
```
