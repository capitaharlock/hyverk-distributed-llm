const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('hyverk', {
  getStatus: () => ipcRenderer.invoke('get-status'),
  getNodes: () => ipcRenderer.invoke('get-nodes'),
  getMetrics: () => ipcRenderer.invoke('get-metrics'),
  getTrainingJobs: () => ipcRenderer.invoke('get-training-jobs'),
  submitInference: (params) => ipcRenderer.invoke('submit-inference', params),
  getInference: (taskId) => ipcRenderer.invoke('get-inference', taskId),
  startHyverk: () => ipcRenderer.invoke('start-hyverk'),
  stopHyverk: () => ipcRenderer.invoke('stop-hyverk'),
  onLog: (callback) => {
    ipcRenderer.on('hyverk-log', (_, line) => callback(line));
  },
  onStatus: (callback) => {
    ipcRenderer.on('hyverk-status', (_, status) => callback(status));
  },
});
