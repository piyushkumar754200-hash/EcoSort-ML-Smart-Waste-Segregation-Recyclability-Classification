import { spawn, ChildProcess } from 'child_process';
import express from 'express';
import { createProxyMiddleware } from 'http-proxy-middleware';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3000;
const FASTAPI_PORT = Number(process.env.BACKEND_PORT || 8009);
const FASTAPI_TARGET = `http://127.0.0.1:${FASTAPI_PORT}`;

let pythonProcess: ChildProcess | null = null;

function checkFastApiAlive(): Promise<boolean> {
  return new Promise((resolve) => {
    const req = http.get(`${FASTAPI_TARGET}/api/health`, { timeout: 1000 }, (res) => {
      res.resume();
      resolve(res.statusCode === 200);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function ensureFastApiRunning(): Promise<void> {
  const alreadyUp = await checkFastApiAlive();
  if (alreadyUp) {
    console.log(`[server] FastAPI backend already active at ${FASTAPI_TARGET}`);
    return;
  }

  const venvUvicorn = path.resolve(__dirname, '.venv', 'bin', 'uvicorn');
  console.log(`[server] Starting FastAPI backend on port ${FASTAPI_PORT}...`);

  pythonProcess = spawn(
    venvUvicorn,
    ['app.main:app', '--host', '127.0.0.1', '--port', String(FASTAPI_PORT)],
    {
      cwd: path.resolve(__dirname, 'backend'),
      env: {
        ...process.env,
        PYTHONPATH: path.resolve(__dirname, 'backend'),
      },
      stdio: 'inherit',
    }
  );

  pythonProcess.on('exit', (code) => {
    console.log(`[server] FastAPI process exited with code ${code}`);
    pythonProcess = null;
  });

  for (let i = 0; i < 16; i++) {
    await new Promise((r) => setTimeout(r, 500));
    if (await checkFastApiAlive()) {
      console.log(`[server] FastAPI backend ready at ${FASTAPI_TARGET}`);
      return;
    }
  }
}

async function startServer() {
  await ensureFastApiRunning();

  const app = express();

  const apiProxy = createProxyMiddleware({
    target: FASTAPI_TARGET,
    changeOrigin: true,
    ws: false,
  });

  app.use('/api', (req, res, next) => {
    req.url = `/api${req.url}`;
    return apiProxy(req, res, next);
  });
  app.use('/docs', (req, res, next) => {
    req.url = `/docs${req.url}`;
    return apiProxy(req, res, next);
  });
  app.use('/openapi.json', (req, res, next) => {
    req.url = '/openapi.json';
    return apiProxy(req, res, next);
  });

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*all', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[server] EcoSort ML Unified Server listening at http://0.0.0.0:${PORT}`);
  });
}

process.on('SIGTERM', () => {
  if (pythonProcess) pythonProcess.kill('SIGTERM');
  process.exit(0);
});

startServer().catch((err) => {
  console.error('[server] Fatal startup error:', err);
  process.exit(1);
});
