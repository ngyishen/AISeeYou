globalThis.ort = globalThis.ort || {};
globalThis.ort.env = globalThis.ort.env || {};

globalThis.ort.env.wasm = {
    wasmPaths: chrome.runtime.getURL("libs/wasm/"),
    simd: true,
    numThreads: 1,
    loadLocalFile: true,
    proxy: false
};