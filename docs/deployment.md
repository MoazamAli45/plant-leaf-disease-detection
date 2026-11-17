# Deployment

**Everything the app does runs in the user's browser**: the model (`web/public/model/`,
6.6 MB) and the Grad-CAM heat maps. So the whole app is a website, and it can be hosted free
on Vercel. No server, API or separate model hosting is needed.

| Part | Where | Cost |
|---|---|---|
| Web app, model and heat maps (`web/`) | Vercel | free (Hobby plan, personal projects) |
| Server-side API (`api/`), optional, not used by the app | Hugging Face Spaces | see part 2 |

## 1. Web app on Vercel (free)

1. Push the repository to GitHub.
2. Sign in at <https://vercel.com> with your GitHub account.
3. **Add New… → Project**, and import `plant-leaf-disease-detection`.
4. Set **Root Directory** to `web`. Leave the other settings as detected (framework Next.js,
   build command `npm run build`). No environment variables are needed.
5. Click **Deploy**. After about two minutes the app is live at
   `https://<project-name>.vercel.app`.

Every later `git push` to `main` redeploys automatically.

**Check it works:**

- open the URL on your phone and analyse a leaf photo;
- on the result page, press **Show heat map**;
- install it (Android Chrome: menu → "Add to Home screen"; iPhone Safari: Share → "Add to Home
  Screen"), turn on airplane mode, and analyse another photo, heat map included. It should all
  still work.

## 2. Optional: the API on Hugging Face Spaces

The FastAPI service in `api/` offers the same prediction and heat map over HTTP, for scripts
or other apps. The web app does not use it, so skip this part unless you want a public API.

**Cost (November 2026):** since about July 2026, Hugging Face has reportedly required a PRO
subscription (about $9 a month) to create Docker Spaces on its CPU hardware. When you create the
Space, it shows whether "Docker" on "CPU basic" is free or paid for your account.

1. Create the model files: `cd ml && .venv/bin/python -m leafdx.train.export_web api-model`.
2. On <https://huggingface.co>: **New → Space**, name `leafdx-api`, SDK **Docker** (template
   **Blank**), hardware **CPU basic**, **Create Space**.
3. Log in from the terminal with a **Write** access token (Settings → Access Tokens):

   ```bash
   pip install -U huggingface_hub
   hf auth login        # paste the token when asked; never commit it or share it in chat
   ```

4. Upload, from the repository root (`<user>` is your Hugging Face username):

   ```bash
   hf upload <user>/leafdx-api api . --repo-type=space \
     --include "Dockerfile" --include "pyproject.toml" --include "src/*" --include "model/*" \
     --exclude "*__pycache__*" --exclude "*.egg-info*"
   hf upload <user>/leafdx-api api/huggingface/README.md README.md --repo-type=space
   ```

   Hugging Face builds and starts it (5–10 minutes the first time; see the **Logs** tab).
   The `Dockerfile` is Hugging Face's build recipe; you do not need Docker yourself.

5. Check `https://<user>-leafdx-api.hf.space/health` (should show `"model_loaded": true`);
   interactive docs at `/docs`. To call it from a web page, add the page's address in the
   Space's **Settings → Variables**: `LEAFDX_CORS_ORIGINS` = `https://<site>`.

### Also free: publish the trained model on Hugging Face

Model repositories are free. To let others download the weights: **New → Model**, name it
`leafdx-mobilenet-v1`, then:

```bash
hf upload <user>/leafdx-mobilenet-v1 api/model .
hf upload <user>/leafdx-mobilenet-v1 docs/model_card.md README.md
```

## API settings

| Variable | Default | |
|---|---|---|
| `LEAFDX_CORS_ORIGINS` | `http://localhost:3000` | comma-separated web addresses allowed to call the API |
| `LEAFDX_RATE_LIMIT` | `30` | `/predict` requests per client per minute |
| `LEAFDX_MAX_UPLOAD_MB` | `10` | upload size limit |
| `LEAFDX_MODEL`, `LEAFDX_CLASSES` | `api/model/…` | model and class files (set in the Dockerfile) |
