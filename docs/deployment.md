# Deployment (free)

The web app is the whole user-facing product. It runs the model **in the browser**, so it needs
no server of its own and can be hosted free on Vercel. The Grad-CAM server (`api/`) is optional:
without it the app works fully and simply hides the "Show heat map" button.

## Web app on Vercel (free Hobby plan)

The Hobby plan is free for personal, non-commercial projects.

1. Push the repository to GitHub.
2. Sign in at <https://vercel.com> with your GitHub account.
3. **Add New… → Project**, and import `plant-leaf-disease-detection`.
4. Set **Root Directory** to `web`. Leave the other settings as detected (framework Next.js,
   build command `npm run build`).
5. Click **Deploy**. After about two minutes the app is live at
   `https://<project-name>.vercel.app`.

Every later `git push` to `main` redeploys automatically.

**Check it works:**

- open the URL on your phone and analyse a leaf photo on the Detect page;
- install it: Chrome menu → **Install app** (Android: "Add to Home screen"; iPhone Safari:
  Share → "Add to Home Screen");
- turn on airplane mode and analyse another photo: it should still work.

## Grad-CAM heat maps

The heat-map server loads TensorFlow and needs about 560 MB of memory. As of November 2026 there
is no reliable free host for that: Render's free tier has 512 MB, Hugging Face no longer offers
free CPU Spaces for this kind of app, and Google Cloud Run requires a credit card. So:

- **for demos**, run it on your own computer:

  ```bash
  (cd ml && .venv/bin/python -m leafdx.train.export_web api-model)   # once
  cd api && .venv/bin/uvicorn leafdx_api.main:app --port 8000
  cd web && NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
  ```

- **for the hosted site**, the report's Grad-CAM figures show the heat maps.

If you later host the server somewhere, set `LEAFDX_CORS_ORIGINS=https://<project-name>.vercel.app`
on the server and `NEXT_PUBLIC_API_URL=https://<server>` in Vercel (Settings → Environment
Variables), then redeploy.

## Server settings

| Variable | Default | |
|---|---|---|
| `LEAFDX_CORS_ORIGINS` | `http://localhost:3000` | comma-separated web addresses allowed to call the server |
| `LEAFDX_RATE_LIMIT` | `30` | heat-map requests per user per minute |
| `LEAFDX_MAX_UPLOAD_MB` | `10` | upload size limit |
| `LEAFDX_MODEL`, `LEAFDX_CLASSES` | `api/model/…` | model and class files |
