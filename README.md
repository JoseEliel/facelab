---
title: FaceLab
emoji: 😃
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
python_version: 3.10.13
app_file: app.py
pinned: false
license: mit
---

## Cloudflare Turnstile setup

This app now supports optional bot protection with Cloudflare Turnstile.

1. Create a Turnstile widget in Cloudflare.
2. Add your exact Space hostname in the widget settings, for example `your-name-your-space.hf.space`.
3. Set `TURNSTILE_SITE_KEY` and `TURNSTILE_SECRET_KEY`.
4. On Hugging Face Spaces, add `TURNSTILE_SITE_KEY` as a Variable (or Secret) and `TURNSTILE_SECRET_KEY` as a Secret in `Settings` → `Variables and secrets`.
5. Restart the Space.

If both variables are missing, the app still runs without Turnstile.
