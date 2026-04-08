// @llm-context: _rjj/stack.md
// @llm-depends: ../crates/hyverk/src/main.rs

const { app, BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');

let mainWindow = null;
let hyverkProcess = null;

// Resolve path to the hyverk binary
function getHyverkBinaryPath() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'bin', 'hyverk');
  }
  // Development: use cargo build output
  return path.join(__dirname, '..', '..', 'target', 'debug', 'hyverk');
}

// Resolve config path
function getConfigPath() {
  if (app.isPackaged) {
    return path.join(app.getPath('userData'), 'config.toml');
  }
  return path.join(__dirname, '..', '..', 'config.toml');
}

function startHyverkProcess() {
  const binary = getHyverkBinaryPath();
  const configPath = getConfigPath();

  console.log(`Starting hyverk: ${binary}`);
  console.log(`Config: ${configPath}`);

  hyverkProcess = spawn(binary, ['--config', configPath, 'run'], {
    env: {
      ...process.env,
      RUST_LOG: 'info',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  hyverkProcess.stdout.on('data', (data) => {
    const line = data.toString().trim();
    console.log(`[hyverk] ${line}`);
    if (mainWindow) {
      mainWindow.webContents.send('hyverk-log', line);
    }
  });

  hyverkProcess.stderr.on('data', (data) => {
    const line = data.toString().trim();
    console.log(`[hyverk] ${line}`);
    if (mainWindow) {
      mainWindow.webContents.send('hyverk-log', line);
    }
  });

  hyverkProcess.on('exit', (code) => {
    console.log(`Hyverk exited with code ${code}`);
    hyverkProcess = null;
    if (mainWindow) {
      mainWindow.webContents.send('hyverk-status', { running: false, exitCode: code });
    }
  });

  return true;
}

function stopHyverkProcess() {
  if (hyverkProcess) {
    hyverkProcess.kill('SIGINT'); // Triggers graceful shutdown
    hyverkProcess = null;
  }
}

// HTTP helper to call the coordinator API
function apiRequest(method, path, body = null) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: '127.0.0.1',
      port: 17000,
      path,
      method,
      headers: { 'Content-Type': 'application/json' },
      timeout: 5000,
    };

    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => (data += chunk));
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, data: JSON.parse(data) });
        } catch {
          resolve({ status: res.statusCode, data });
        }
      });
    });

    req.on('error', (err) => reject(err));
    req.on('timeout', () => {
      req.destroy();
      reject(new Error('Request timeout'));
    });

    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

// IPC handlers
ipcMain.handle('get-status', async () => {
  try {
    const health = await apiRequest('GET', '/health');
    const nodes = await apiRequest('GET', '/api/v1/nodes');
    return {
      running: hyverkProcess !== null,
      healthy: health.status === 200,
      nodes: nodes.data?.nodes || [],
    };
  } catch {
    return { running: hyverkProcess !== null, healthy: false, nodes: [] };
  }
});

ipcMain.handle('get-nodes', async () => {
  try {
    const resp = await apiRequest('GET', '/api/v1/nodes');
    return resp.data?.nodes || [];
  } catch {
    return [];
  }
});

ipcMain.handle('submit-inference', async (_, { model, prompt, temperature, max_tokens }) => {
  try {
    const resp = await apiRequest('POST', '/api/v1/inference', {
      model,
      prompt,
      temperature,
      max_tokens,
    });
    return resp.data;
  } catch (err) {
    return { error: err.message };
  }
});

ipcMain.handle('get-inference', async (_, taskId) => {
  try {
    const resp = await apiRequest('GET', `/api/v1/inference/${taskId}`);
    return resp.data;
  } catch (err) {
    return { error: err.message };
  }
});

ipcMain.handle('start-hyverk', () => {
  if (hyverkProcess) return { ok: false, error: 'Already running' };
  startHyverkProcess();
  return { ok: true };
});

ipcMain.handle('stop-hyverk', () => {
  stopHyverkProcess();
  return { ok: true };
});

ipcMain.handle('get-metrics', async () => {
  try {
    const resp = await apiRequest('GET', '/api/v1/metrics');
    return resp.status === 200 ? resp.data : null;
  } catch {
    return null;
  }
});

ipcMain.handle('get-training-jobs', async () => {
  try {
    const resp = await apiRequest('GET', '/api/v1/training/jobs');
    return resp.data?.jobs || [];
  } catch {
    return [];
  }
});

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 900,
    height: 700,
    title: 'Hyverk',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'index.html'));

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  createWindow();
  startHyverkProcess();
});

app.on('window-all-closed', () => {
  stopHyverkProcess();
  app.quit();
});

app.on('before-quit', () => {
  stopHyverkProcess();
});
