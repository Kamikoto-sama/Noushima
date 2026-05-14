const POLL_INTERVAL_MS = 150;
const OFFLINE_POLL_INTERVAL_MS = 5000;
const MIN_CELL_SIZE = 12;
const MAX_CELL_SIZE = 100;

const generationValue = document.getElementById("generationValue");
const botsAliveValue = document.getElementById("botsAliveValue");
const bestEnergyValue = document.getElementById("bestEnergyValue");
const canvas = document.getElementById("simulationCanvas");
const context = canvas.getContext("2d");
const hud = document.querySelector(".hud");

let modeGroup = document.querySelector(".mode-group");
let modeButtons = Array.from(document.querySelectorAll(".mode-button"));

let latestState = null;
let pendingModeValue = null;

function ensureModeControls() {
    if (modeGroup !== null) {
        return;
    }

    modeGroup = document.createElement("div");
    modeGroup.className = "mode-group";
    modeGroup.setAttribute("role", "radiogroup");
    modeGroup.setAttribute("aria-label", "Simulation speed");

    for (const mode of ["Pause", "Slow", "Normal", "Fast"]) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "mode-button";
        button.dataset.mode = mode;
        button.setAttribute("role", "radio");
        button.setAttribute("aria-checked", String(mode === "Normal"));
        button.textContent = mode;
        modeGroup.appendChild(button);
    }

    hud?.appendChild(modeGroup);
    modeButtons = Array.from(modeGroup.querySelectorAll(".mode-button"));
}

function getBestEnergyText(state) {
    const bestEnergy = state.bestEnergy ?? state.BestEnergy;
    if (bestEnergy == null) {
        return "-";
    }

    return String(Math.round(bestEnergy));
}

function getCellSize(state) {
    const totalCells = state.width * state.height;
    const availableWidth = window.innerWidth - 24;
    const availableHeight = window.innerHeight - 96;
    const viewportCellSize = Math.min(availableWidth / state.width, availableHeight / state.height);
    const densityScale = Math.sqrt(1000 / totalCells);
    const scaledCellSize = Math.floor(viewportCellSize * densityScale);
    return Math.max(MIN_CELL_SIZE, Math.min(MAX_CELL_SIZE, scaledCellSize));
}

function resizeCanvas(state) {
    const cellSize = getCellSize(state);
    canvas.width = state.width * cellSize;
    canvas.height = state.height * cellSize;
    return cellSize;
}

function drawGrid(width, height, cellSize) {
    context.fillStyle = "#000000";
    context.fillRect(0, 0, width * cellSize, height * cellSize);

    context.strokeStyle = "rgba(85, 103, 120, 0.35)";
    context.lineWidth = 1;

    for (let x = 0; x <= width; x += 1) {
        const px = x * cellSize + 0.5;
        context.beginPath();
        context.moveTo(px, 0);
        context.lineTo(px, height * cellSize);
        context.stroke();
    }

    for (let y = 0; y <= height; y += 1) {
        const py = y * cellSize + 0.5;
        context.beginPath();
        context.moveTo(0, py);
        context.lineTo(width * cellSize, py);
        context.stroke();
    }
}

function getCellColor(type) {
    switch (type) {
        case "Wall":
            return "#70757d";
        case "Food":
            return "#35c96c";
        case "Poison":
            return "#dd4455";
        case "Bot":
            return "#2d7ff9";
        default:
            return "#000000";
    }
}

function drawBotDirection(cell, cellSize) {
    const x = cell.x * cellSize + 1;
    const y = cell.y * cellSize + 1;
    const size = cellSize - 1;
    const half = size / 2;
    const thickness = Math.max(2, Math.floor(cellSize * 0.12));

    context.fillStyle = "#ffffff";

    switch (cell.direction) {
        case "Up":
            context.fillRect(x, y, size, thickness);
            break;
        case "Right":
            context.fillRect(x + size - thickness, y, thickness, size);
            break;
        case "Down":
            context.fillRect(x, y + size - thickness, size, thickness);
            break;
        case "Left":
            context.fillRect(x, y, thickness, size);
            break;
        case "TopRight":
            context.fillRect(x + half, y, size - half, thickness);
            context.fillRect(x + size - thickness, y, thickness, half);
            break;
        case "BottomRight":
            context.fillRect(x + size - thickness, y + half, thickness, size - half);
            context.fillRect(x + half, y + size - thickness, size - half, thickness);
            break;
        case "BottomLeft":
            context.fillRect(x, y + half, thickness, size - half);
            context.fillRect(x, y + size - thickness, half, thickness);
            break;
        case "TopLeft":
            context.fillRect(x, y, thickness, half);
            context.fillRect(x, y, half, thickness);
            break;
    }
}

function drawState(state) {
    generationValue.textContent = String(state.generation);
    botsAliveValue.textContent = String(state.botsAlive);
    bestEnergyValue.textContent = getBestEnergyText(state);

    const cellSize = resizeCanvas(state);
    drawGrid(state.width, state.height, cellSize);

    const botCells = [];

    for (const cell of state.cells) {
        if (cell.type === "Bot") {
            botCells.push(cell);
            continue;
        }

        context.fillStyle = getCellColor(cell.type);
        context.fillRect(cell.x * cellSize + 1, cell.y * cellSize + 1, cellSize - 1, cellSize - 1);
    }

    for (const cell of botCells) {
        context.fillStyle = getCellColor(cell.type);
        context.fillRect(cell.x * cellSize + 1, cell.y * cellSize + 1, cellSize - 1, cellSize - 1);
        drawBotDirection(cell, cellSize);
    }

    context.fillStyle = "#ffffff";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.font = `bold ${Math.max(9, Math.floor(cellSize * 0.46))}px Arial`;

    for (const cell of botCells) {
        if (cell.energy == null) {
            continue;
        }

        context.fillText(
            String(cell.energy),
            cell.x * cellSize + cellSize / 2,
            cell.y * cellSize + cellSize / 2
        );
    }
}

function updateGeneration(state) {
    generationValue.textContent = String(state.generation);
    botsAliveValue.textContent = String(state.botsAlive);
    bestEnergyValue.textContent = getBestEnergyText(state);
}

function syncMode(mode) {
    if (pendingModeValue !== null && mode !== pendingModeValue) {
        return;
    }

    for (const button of modeButtons) {
        const selected = button.dataset.mode === mode;
        button.setAttribute("aria-checked", String(selected));
    }

    pendingModeValue = null;
    setModeBusy(false);
}

function setModeBusy(isBusy) {
    if (modeGroup === null) {
        return;
    }

    modeGroup.dataset.busy = String(isBusy);

    for (const button of modeButtons) {
        button.disabled = isBusy;
    }
}

async function fetchState() {
    const response = await fetch("/api/simulation/state", { cache: "no-store" });
    if (!response.ok) {
        throw new Error(`Request failed with ${response.status}`);
    }

    return response.json();
}

async function fetchModeState() {
    const response = await fetch("/api/simulation/mode", { cache: "no-store" });
    if (!response.ok) {
        throw new Error(`Request failed with ${response.status}`);
    }

    return response.json();
}

async function setMode(mode) {
    const response = await fetch(`/api/simulation/mode?mode=${encodeURIComponent(mode)}`, {
        method: "POST"
    });

    if (!response.ok) {
        throw new Error(`Request failed with ${response.status}`);
    }
}

async function pollState() {
    let nextPollIntervalMs = POLL_INTERVAL_MS;

    try {
        const state = await fetchState();
        syncMode(state.mode ?? "Normal");

        if (state.cells != null && state.width != null && state.height != null) {
            latestState = state;
            drawState(state);
        } else {
            updateGeneration(state);
        }
    } catch (error) {
        generationValue.textContent = "offline";
        botsAliveValue.textContent = "-";
        bestEnergyValue.textContent = "-";
        nextPollIntervalMs = OFFLINE_POLL_INTERVAL_MS;
        console.error(error);
    } finally {
        window.setTimeout(pollState, nextPollIntervalMs);
    }
}

window.addEventListener("resize", () => {
    if (latestState !== null) {
        drawState(latestState);
    }
});

ensureModeControls();

modeGroup?.addEventListener("click", async event => {
    const button = event.target.closest(".mode-button");
    if (button == null || button.disabled) {
        return;
    }

    const mode = button.dataset.mode;
    if (mode == null) {
        return;
    }

    pendingModeValue = mode;
    syncMode(mode);
    setModeBusy(true);

    try {
        await setMode(mode);
    } catch (error) {
        pendingModeValue = null;
        setModeBusy(false);
        syncMode(latestState?.mode ?? "Normal");
        console.error(error);
    }
});

async function initialize() {
    try {
        const modeState = await fetchModeState();
        syncMode(modeState.mode ?? "Normal");
    } catch (error) {
        console.error(error);
    }

    pollState();
}

initialize();
