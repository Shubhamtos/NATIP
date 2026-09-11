# NATIP Netlify Deployment

Netlify cannot directly run the Python Streamlit NATIP dashboard as a long-lived
server. This Netlify setup publishes a lightweight static launcher from:

```text
netlify/
```

The full app should still run on Streamlit Cloud, a local tunnel, or a Python
server.

## Netlify Settings

- Build command: leave blank
- Publish directory: `netlify`
- Config file: `netlify.toml`

## GitHub To Netlify

1. Push this repository to GitHub.
2. Open Netlify.
3. Add new site from Git.
4. Select the NATIP repository.
5. Use publish directory `netlify`.
6. Deploy.

## To Host The Full Dashboard

Use Streamlit Cloud with:

```text
Main file: streamlit_app.py
```

or keep running a local tunnel to:

```text
http://localhost:8501
```

