# Deployment

LeafDx has two deployable parts:

| Part | What it does | Needed? |
|---|---|---|
| **Web app** (`web/`) | the whole user-facing app; classifies photos **in the browser** with TensorFlow.js and works offline | yes |
| **API** (`api/`) | Grad-CAM heat maps (and server-side prediction) | optional: without it the app works fully and simply hides the "Show heat map" button |

## Recommended free set-up (checked November 2026)

1. **Web app on Vercel, Hobby plan.** Free for personal, non-commercial projects, which a
   research demo is. Vercel builds Next.js natively, serves over HTTPS (required for the
   offline mode and "install app"), and redeploys on every `git push`.
2. **No hosted API at first.** Deploy the web app alone. Show the heat maps from the report
   figures, or run the API locally when you present (section 4).

Why the API is not on a free host. It loads TensorFlow and uses about **470 MB of RAM at idle
and 560 MB after a Grad-CAM request** (measured locally). The usual free options do not fit:

- **Hugging Face Spaces** (the host originally planned): since about July 2026, free accounts
  can no longer create Docker or Gradio Spaces on the free CPU hardware; only static Spaces
  stay free. With HF PRO, the Dockerfile here works unchanged (section 3).
- **Render free:** 512 MB RAM and 0.1 CPU, so the API would run out of memory.
- **Google Cloud Run** has a free monthly allowance but needs a billing account with a card.

Free-tier terms change often. Check the provider's pricing page before relying on these.

## 1. Web app on Vercel (free)

One-time set-up, in the browser:

1. Push the repository to GitHub.
2. Sign in at <https://vercel.com> with GitHub and choose **Add New… → Project**.
3. Import `plant-leaf-disease-detection`.
4. **Root Directory: `web`** (important). Vercel detects Next.js; keep the default build
   command (`npm run build`, which also copies the TF.js WebAssembly files).
5. Environment variables: none needed. If you later host the API, add
   `NEXT_PUBLIC_API_URL = https://<your-api-host>` and redeploy.
6. **Deploy.** The app is live at `https://<project>.vercel.app` after about two minutes.

Every later `git push` to `main` redeploys automatically; pull requests get preview URLs.

**Check the deployment:**

- open the URL on a phone, take a photo of a leaf on the Detect page;
- in Chrome: menu → **Install app** (or "Add to Home screen" on Android; Share → "Add to
  Home Screen" on iOS);
- turn on airplane mode and analyse another photo: it should still work.

## 2. Docker (any host, or locally)

Both images are built in CI on every push (`.github/workflows/ci.yml`, job `docker`).

```bash
# API: the model files are generated, not committed
(cd ml && .venv/bin/python -m leafdx.train.export_web api-model)      # writes api/model/
docker build --platform linux/amd64 -t leafdx-api api
docker run --rm -p 7860:7860 -e LEAFDX_CORS_ORIGINS=http://localhost:3000 leafdx-api

# Web (NEXT_PUBLIC_API_URL is fixed at build time; omit it to hide the heat-map button)
docker build -t leafdx-web --build-arg NEXT_PUBLIC_API_URL=http://localhost:7860 web
docker run --rm -p 3000:3000 leafdx-web
```

Or both at once: `docker compose up --build` from the repository root (web on
<http://localhost:3000>, API on <http://localhost:7860/docs>).

The API image is CPU-only TensorFlow on Linux x86-64 (`--platform linux/amd64` is needed on
Apple-silicon Macs). Give it at least 1 GB of memory.

## 3. API on Hugging Face Spaces (needs HF PRO)

`api/Dockerfile` follows the Spaces conventions (port 7860, user id 1000), so the API folder
can be pushed as a Docker Space:

```bash
(cd ml && .venv/bin/python -m leafdx.train.export_web api-model)
bash api/scripts/make_space.sh            # assembles api/space/build/ with the Space README
pip install -U huggingface_hub && hf auth login
hf upload <your-hf-user>/leafdx-api api/space/build . --repo-type=space
```

Then, in the Space's **Settings → Variables**, set
`LEAFDX_CORS_ORIGINS = https://<project>.vercel.app`, and in Vercel set
`NEXT_PUBLIC_API_URL = https://<your-hf-user>-leafdx-api.hf.space` and redeploy.

## 4. Running the API locally for a demo

```bash
(cd ml && .venv/bin/python -m leafdx.train.export_web api-model)
cd api && .venv/bin/uvicorn leafdx_api.main:app --port 8000
cd web && NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

## API configuration

| Variable | Default | |
|---|---|---|
| `LEAFDX_CORS_ORIGINS` | `http://localhost:3000` | comma-separated origins allowed to call the API; set it to the web app's URL |
| `LEAFDX_RATE_LIMIT` | `30` | `/predict` calls per client per minute |
| `LEAFDX_MAX_UPLOAD_MB` | `10` | upload size limit |
| `LEAFDX_MAX_PIXELS` | `40000000` | decompression-bomb guard |
| `LEAFDX_MODEL`, `LEAFDX_CLASSES` | `api/model/…` | model and class files (set in the Docker image) |
| `PORT` | `7860` | Docker image only |

The rate limiter is in memory, so it applies per process. Behind a proxy it uses the first
address in `X-Forwarded-For`.
