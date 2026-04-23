// DOM Elements
const tbody = document.getElementById("crypto-body");
const coinSelect = document.getElementById("coinSelect");
const loader = document.getElementById("loader");
const lstmVal = document.getElementById("lstm-val");
const xgbVal = document.getElementById("xgb-val");
const result = document.getElementById("prediction-result");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");
const pageNumber = document.getElementById("page-number");
const metricsSection = document.getElementById("metrics-section");
const chartSection = document.getElementById("chart-section");
const liveGrid = document.getElementById("live-grid");
const socketStatus = document.getElementById("socket-status");
const liveUpdated = document.getElementById("live-updated");
const liveEmptyState = document.getElementById("live-empty");
const themeToggle = document.getElementById("themeToggle");

// State
let currentPage = 1;
const coinsPerPage = 10;
let allCoins = [];
let chartInstance = null;
let chartCtx = document.getElementById("prediction-chart")?.getContext("2d");
const STREAM_COINS = ["bitcoin", "ethereum", "tether", "solana", "binancecoin", "ripple"];
let socket;
let currentTheme = localStorage.getItem("theme") || "dark";

// Theme functions
function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
    currentTheme = theme;
    themeToggle.textContent = theme === "dark" ? "🌙 Dark" : "☀️ Light";
}

function toggleTheme() {
    const newTheme = currentTheme === "dark" ? "light" : "dark";
    setTheme(newTheme);
}

// Initialize theme
setTheme(currentTheme);

const formatUsd = (value, digits = 2) =>
    value === undefined || value === null
        ? "--"
        : `$${Number(value).toLocaleString("en-US", {
              minimumFractionDigits: digits,
              maximumFractionDigits: digits
          })}`;

// Fetch all coins for table
async function fetchAllCoins() {
    try {
        const res = await fetch("https://api.coinlore.net/api/tickers/?start=0&limit=250");
        allCoins = await res.json();
        // Map to similar format
        allCoins = allCoins.map(coin => ({
            id: coin.id,
            symbol: coin.symbol.toLowerCase(),
            name: coin.name,
            image: `https://api.coinlore.net/img/${coin.id}.png`, // CoinLore has img
            current_price: parseFloat(coin.price_usd),
            price_change_percentage_24h: parseFloat(coin.percent_change_24h),
            market_cap: parseFloat(coin.market_cap_usd),
            // no sparkline
        }));
        renderTable();
    } catch (err) {
        console.error("Failed to fetch coins:", err);
    }
}

// Render table with pagination
function renderTable() {
    if (!tbody) return;
    tbody.innerHTML = "";
    const start = (currentPage - 1) * coinsPerPage;
    const pageCoins = allCoins.slice(start, start + coinsPerPage);
    
    pageCoins.forEach((coin, i) => {
        const changePos = coin.price_change_percentage_24h >= 0;
        const safeId = `spark-${coin.id.replace(/[^a-zA-Z0-9]/g, '')}-${start + i}`;
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${start + i + 1}</td>
            <td><img src="${coin.image}" class="crypto-img"> ${coin.name}</td>
            <td>$${coin.current_price.toLocaleString()}</td>
            <td class="${changePos ? 'green' : 'red'}">${coin.price_change_percentage_24h?.toFixed(2)}%</td>
        `;
        tbody.appendChild(row);
    });
    
    const maxPage = Math.ceil(allCoins.length / coinsPerPage);
    pageNumber.textContent = currentPage;
    prevBtn.disabled = currentPage === 1;
    nextBtn.disabled = currentPage >= maxPage;
}

// Pagination
prevBtn.onclick = () => { if (currentPage > 1) { currentPage--; renderTable(); } };
nextBtn.onclick = () => { if (currentPage < Math.ceil(allCoins.length / coinsPerPage)) { currentPage++; renderTable(); } };

// Load top 10 coins for dropdown
async function loadTopCoins() {
    if (!coinSelect) return;
    try {
        const res = await fetch("https://api.coinlore.net/api/tickers/?start=0&limit=10");
        const coins = await res.json();
        coinSelect.innerHTML = '<option value="">Select a coin...</option>';
        coins.forEach(coin => {
            const option = document.createElement("option");
            option.value = coin.symbol.toLowerCase();
            option.textContent = coin.name;
            coinSelect.appendChild(option);
        });
    } catch (err) {
        console.error("Failed to load coins:", err);
    }
}

// Get prediction
async function getPrediction() {
    if (!coinSelect || !coinSelect.value) {
        resetDisplay();
        return;
    }
    
    if (loader) loader.style.display = "block";
    if (lstmVal) lstmVal.textContent = "Loading...";
    if (xgbVal) xgbVal.textContent = "Loading...";

    try {
        const res = await fetch(`http://localhost:5000/predict?symbol=${coinSelect.value}`);
        const data = await res.json();
        if (loader) loader.style.display = "none";
        
        if (data.error) {
            showError(data.error);
            return;
        }

        if (lstmVal) lstmVal.textContent = formatUsd(data.lstm_prediction);
        if (xgbVal) xgbVal.textContent = formatUsd(data.xgboost_prediction);
        if (result) {
            result.textContent = `✅ Predictions for ${data.symbol}`;
            result.style.color = "#1b7b1b";
        }
        
        updateMetrics(data.lstm_metrics, data.xgb_metrics);
        if (data.history && chartCtx) displayChart(data.history, data.lstm_prediction, data.xgboost_prediction, data.symbol);
    } catch (err) {
        console.error("Prediction error:", err);
        if (loader) loader.style.display = "none";
        showError("Backend not responding. Make sure backend is running on port 5000");
    }
}

// Update metrics
function updateMetrics(lstmMetrics, xgbMetrics) {
    if (!metricsSection) return;
    
    const updateMetric = (prefix, metrics) => {
        if (metrics && Object.keys(metrics).length > 0) {
            document.getElementById(`${prefix}-mae`).textContent = metrics.mae?.toFixed(2) || "--";
            document.getElementById(`${prefix}-rmse`).textContent = metrics.rmse?.toFixed(2) || "--";
            document.getElementById(`${prefix}-mape`).textContent = (metrics.mape?.toFixed(2) || "--") + "%";
            document.getElementById(`${prefix}-r2`).textContent = metrics.r2?.toFixed(4) || "--";
        }
    };
    
    updateMetric("lstm", lstmMetrics);
    updateMetric("xgb", xgbMetrics);
    metricsSection.style.display = "block";
}

// Display chart with smooth continuation
function displayChart(history, lstmPred, xgbPred, symbol) {
    if (!chartSection || !chartCtx) return;
    
    chartSection.style.display = "block";
    if (chartInstance) chartInstance.destroy();
    
    const lastPrice = history[history.length - 1];
    const futureDays = 5;
    const lstmFuture = Array.from({length: futureDays}, (_, i) => {
        const t = (i + 1) / (futureDays + 1);
        return lastPrice + (lstmPred - lastPrice) * t;
    });
    const xgbFuture = Array.from({length: futureDays}, (_, i) => {
        const t = (i + 1) / (futureDays + 1);
        return lastPrice + (xgbPred - lastPrice) * t;
    });
    
    const labels = [...history.map((_, i) => `Day ${i + 1}`), ...Array(futureDays).fill(null).map((_, i) => `Day ${history.length + i + 1}`)];
    
    chartInstance = new Chart(chartCtx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Historical Price (30 Days)",
                    data: [...history, ...Array(futureDays).fill(null)],
                    borderColor: "#007bff",
                    backgroundColor: "rgba(0, 123, 255, 0.1)",
                    borderWidth: 2,
                    pointRadius: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: "LSTM Prediction",
                    data: [...Array(history.length - 1).fill(null), lastPrice, ...lstmFuture, lstmPred],
                    borderColor: "#28a745",
                    backgroundColor: "rgba(40, 167, 69, 0.1)",
                    borderWidth: 3,
                    pointRadius: 3,
                    pointBackgroundColor: "#28a745",
                    tension: 0.4,
                    fill: false
                },
                {
                    label: "XGBoost Prediction",
                    data: [...Array(history.length - 1).fill(null), lastPrice, ...xgbFuture, xgbPred],
                    borderColor: "#ffc107",
                    backgroundColor: "rgba(255, 193, 7, 0.1)",
                    borderWidth: 3,
                    pointRadius: 3,
                    pointBackgroundColor: "#ffc107",
                    tension: 0.4,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                title: { display: true, text: `${symbol.toUpperCase()} Price Prediction Chart`, font: { size: 16, weight: 'bold' } },
                legend: { display: true, position: 'top' },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            if (context.parsed.y === null) return '';
                            return context.dataset.label + ': $' + context.parsed.y.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
                        }
                    }
                }
            },
            scales: {
                x: { display: true, title: { display: true, text: 'Days' }, grid: { display: true, color: 'rgba(0,0,0,0.1)' } },
                y: {
                    display: true,
                    title: { display: true, text: 'Price (USD)' },
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});
                        }
                    }
                }
            },
            interaction: { mode: 'index', intersect: false }
        }
    });
}

// Reset display
function resetDisplay() {
    if (lstmVal) lstmVal.textContent = "--";
    if (xgbVal) xgbVal.textContent = "--";
    if (result) { result.textContent = ""; result.style.color = ""; }
    if (metricsSection) metricsSection.style.display = "none";
    if (chartSection) chartSection.style.display = "none";
}

// Show error
function showError(message) {
    if (lstmVal) lstmVal.textContent = "Error";
    if (xgbVal) xgbVal.textContent = "Error";
    if (result) { result.textContent = `❌ ${message}`; result.style.color = "red"; }
    if (metricsSection) metricsSection.style.display = "none";
}

// Live price websocket handling
function setSocketStatus(status, state) {
    if (!socketStatus) return;
    socketStatus.textContent = status;
    socketStatus.classList.remove("online", "offline");
    if (state) socketStatus.classList.add(state);
}

function renderLiveGrid(coins) {
    if (!liveGrid) return;
    liveGrid.innerHTML = "";

    coins.forEach((coin) => {
        const change = Number(coin.change24h ?? 0);
        const card = document.createElement("div");
        card.className = "live-card";
        card.innerHTML = `
            <div class="live-card-head">
                <span>${coin.label || coin.id}</span>
                <span class="change ${change >= 0 ? "pos" : "neg"}">${change?.toFixed(2) ?? "--"}%</span>
            </div>
            <div class="price">${formatUsd(coin.price)}</div>
        `;
        liveGrid.appendChild(card);
    });

    if (liveEmptyState) {
        liveEmptyState.style.display = coins.length ? "none" : "block";
    }
}

function initSocket() {
    if (typeof io === "undefined" || !socketStatus) return;
    socket = io("https://api-cryptoapp.maadhuavati.in", {
        transports: ["websocket"],
        reconnectionAttempts: 5,
        reconnectionDelay: 2000
    });

    socket.on("connect", () => {
        setSocketStatus("Live", "online");
        socket.emit("subscribe_live", { coins: STREAM_COINS });
    });

    socket.on("disconnect", () => {
        setSocketStatus("Disconnected", "offline");
    });

    socket.on("price_update", (payload) => {
        if (!payload?.coins) return;
        renderLiveGrid(payload.coins);
        if (liveUpdated) {
            const ts = payload.timestamp ? new Date(payload.timestamp * 1000) : new Date();
            liveUpdated.textContent = ts.toLocaleTimeString();
        }
    });

    socket.on("price_error", (err) => {
        setSocketStatus("Feed issue", "offline");
        if (liveEmptyState) {
            liveEmptyState.style.display = "block";
            liveEmptyState.textContent = err?.message || "Unable to stream live prices.";
        }
    });
}

// Event listeners
if (coinSelect) coinSelect.addEventListener("change", getPrediction);
if (themeToggle) themeToggle.addEventListener("click", toggleTheme);

// Initialize
fetchAllCoins();
loadTopCoins();
initSocket();
