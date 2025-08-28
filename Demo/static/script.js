// script.js - CAR T-Cell Digital Twin Frontend Logic

document.addEventListener('DOMContentLoaded', () => {
    // Canvas and rendering constants
    const canvas = document.getElementById('simulationCanvas');
    const ctx = canvas.getContext('2d');
    const GRID_SIZE = 50;
    const CELL_SIZE = canvas.width / GRID_SIZE;

    // UI Elements
    const profileSelect = document.getElementById('profileSelect');
    const runStandardBtn = document.getElementById('runStandardBtn');
    const runAIBtn = document.getElementById('runAIBtn');
    const resetBtn = document.getElementById('resetBtn');
    const simulationStatus = document.getElementById('simulationStatus');
    const currentProfile = document.getElementById('currentProfile');
    const actionLog = document.getElementById('actionLog');

    // Metrics elements
    const totalCellsEl = document.getElementById('totalCells');
    const activatedCellsEl = document.getElementById('activatedCells');
    const potentCellsEl = document.getElementById('potentCells');
    const avgPotencyEl = document.getElementById('avgPotency');

    // Chart setup
    let metricsChart = null;
    const chartData = {
        labels: [],
        datasets: [
            {
                label: 'Total Cells',
                data: [],
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                tension: 0.4
            },
            {
                label: 'Activated Cells',
                data: [],
                borderColor: '#27ae60',
                backgroundColor: 'rgba(39, 174, 96, 0.1)',
                tension: 0.4
            },
            {
                label: 'Avg Potency (×10)',
                data: [],
                borderColor: '#e74c3c',
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                tension: 0.4
            }
        ]
    };

    // Global state
    let socket = null;
    let currentScenario = null;
    let simulationRunning = false;
    let stepCount = 0;

    // --- WebSocket Connection ---
    function initializeSocket() {
        socket = io();

        socket.on('connect', () => {
            console.log('Connected to server!');
            updateStatus('Connected - Ready for simulation', 'status-idle');
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            updateStatus('Disconnected from server', 'status-idle');
            enableControls(true);
        });

        socket.on('profiles_available', (data) => {
            populateProfileDropdown(data.profiles);
            if (data.current) {
                profileSelect.value = data.current;
                updateCurrentProfile(data.profiles[data.current]);
            }
        });

        socket.on('profile_loaded', (data) => {
            updateCurrentProfile(data.profile_display);
            addActionLog(`Profile switched to: ${data.profile_display}`);
        });

        socket.on('scenario_started', (data) => {
            currentScenario = data.scenario;
            simulationRunning = true;
            stepCount = 0;
            clearChart();
            updateStatus(`Running ${getScenarioDisplayName(data.scenario)}...`, 'status-running');
            addActionLog(`Started ${getScenarioDisplayName(data.scenario)}`);
        });

        socket.on('update_state', (data) => {
            drawSimulation(data);
            updateMetrics(data);
            updateChart(data);
            
            if (data.last_action !== undefined) {
                const actionName = getActionName(data.last_action);
                addActionLog(`Step ${data.step || stepCount}: ${actionName}`);
            }
            stepCount++;
        });

        socket.on('episode_complete', (data) => {
            simulationRunning = false;
            enableControls(true);
            updateStatus(`${getScenarioDisplayName(data.scenario)} Complete`, 'status-complete');
            
            const summary = `Simulation complete! Final results: ${data.final_metrics.total_cells} total cells, ${data.potent_cells} potent cells (>0.8 potency)`;
            addActionLog(summary);
            
            // Show completion notification
            setTimeout(() => {
                updateStatus('Simulation Ready', 'status-idle');
            }, 5000);
        });

        socket.on('simulation_reset', () => {
            simulationRunning = false;
            enableControls(true);
            clearCanvas();
            clearChart();
            resetMetrics();
            updateStatus('Simulation Reset', 'status-idle');
            addActionLog('Simulation has been reset');
        });

        socket.on('simulation_error', (data) => {
            console.error('Simulation error:', data.error);
            simulationRunning = false;
            enableControls(true);
            updateStatus('Simulation Error', 'status-idle');
            addActionLog(`Error: ${data.error}`);
        });
    }

    // --- UI Helper Functions ---
    function populateProfileDropdown(profiles) {
        profileSelect.innerHTML = '';
        for (const [key, displayName] of Object.entries(profiles)) {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = displayName;
            profileSelect.appendChild(option);
        }
    }

    function updateCurrentProfile(profileName) {
        currentProfile.textContent = profileName;
    }

    function enableControls(enabled) {
        runStandardBtn.disabled = !enabled;
        runAIBtn.disabled = !enabled;
        profileSelect.disabled = !enabled;
        resetBtn.disabled = false; // Reset is always available
    }

    function updateStatus(message, className) {
        simulationStatus.textContent = message;
        simulationStatus.className = `simulation-status ${className}`;
    }

    function addActionLog(message) {
        const entry = document.createElement('div');
        entry.className = 'action-entry';
        entry.textContent = `${new Date().toLocaleTimeString()}: ${message}`;
        actionLog.insertBefore(entry, actionLog.firstChild);
        
        // Keep only last 10 entries
        while (actionLog.children.length > 10) {
            actionLog.removeChild(actionLog.lastChild);
        }
    }

    function getActionName(action) {
        const actionMap = {
            0: "ADD_BEADS",
            1: "REMOVE_BEADS", 
            2: "SKIP"
        };
        return actionMap[action] || 'UNKNOWN';
    }

    function getScenarioDisplayName(scenario) {
        const scenarioMap = {
            'standard_protocol': 'Standard Protocol',
            'ai_strategy': 'AI Strategy'
        };
        return scenarioMap[scenario] || scenario;
    }

    // --- Canvas Rendering ---
    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw grid
        ctx.strokeStyle = '#ecf0f1';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= GRID_SIZE; i++) {
            const pos = i * CELL_SIZE;
            ctx.beginPath();
            ctx.moveTo(pos, 0);
            ctx.lineTo(pos, canvas.height);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, pos);
            ctx.lineTo(canvas.width, pos);
            ctx.stroke();
        }
    }

    function drawSimulation(data) {
        clearCanvas();

        // Draw cells
        if (data.cells) {
            data.cells.forEach(cell => {
                const x = cell.x * CELL_SIZE;
                const y = cell.y * CELL_SIZE;
                
                if (cell.is_activated) {
                    // Activated cells: blue to yellow gradient based on potency
                    const potency = Math.max(0, Math.min(1, cell.potency));
                    if (potency > 0.8) {
                        // High potency - bright blue
                        ctx.fillStyle = '#3498db';
                    } else if (potency > 0.5) {
                        // Medium potency - blue to orange
                        const factor = (potency - 0.5) / 0.3;
                        const r = Math.round(52 + (243 - 52) * (1 - factor));
                        const g = Math.round(152 + (156 - 152) * (1 - factor));
                        const b = Math.round(219 + (18 - 219) * (1 - factor));
                        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                    } else {
                        // Low potency - orange to red (exhausted)
                        const factor = potency / 0.5;
                        const r = Math.round(243 + (231 - 243) * (1 - factor));
                        const g = Math.round(156 + (76 - 156) * (1 - factor));
                        const b = Math.round(18 + (60 - 18) * (1 - factor));
                        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                    }
                } else {
                    // Naive cells - red
                    ctx.fillStyle = '#e74c3c';
                }
                
                ctx.fillRect(x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2);
                
                // Add a small border for better visibility
                ctx.strokeStyle = '#2c3e50';
                ctx.lineWidth = 0.5;
                ctx.strokeRect(x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2);
            });
        }

        // Draw beads
        if (data.beads) {
            ctx.fillStyle = '#27ae60';
            data.beads.forEach(bead => {
                const centerX = (bead[0] + 0.5) * CELL_SIZE;
                const centerY = (bead[1] + 0.5) * CELL_SIZE;
                
                ctx.beginPath();
                ctx.arc(centerX, centerY, CELL_SIZE / 3, 0, 2 * Math.PI);
                ctx.fill();
                
                // Add border
                ctx.strokeStyle = '#1e8449';
                ctx.lineWidth = 2;
                ctx.stroke();
            });
        }
    }

    // --- Metrics and Charts ---
    function updateMetrics(data) {
        if (data.metrics) {
            const metrics = data.metrics;
            totalCellsEl.textContent = metrics.total_cells || 0;
            activatedCellsEl.textContent = metrics.num_activated || 0;
            avgPotencyEl.textContent = (metrics.avg_potency || 0).toFixed(2);
            
            // Calculate potent cells (>0.8 potency)
            let potentCells = 0;
            if (data.cells) {
                potentCells = data.cells.filter(cell => cell.potency > 0.8).length;
            }
            potentCellsEl.textContent = potentCells;
        }
    }

    function resetMetrics() {
        totalCellsEl.textContent = '0';
        activatedCellsEl.textContent = '0';
        potentCellsEl.textContent = '0';
        avgPotencyEl.textContent = '0.00';
    }

    function initializeChart() {
        const chartCtx = document.getElementById('metricsChart').getContext('2d');
        metricsChart = new Chart(chartCtx, {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            font: { size: 10 }
                        }
                    }
                },
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: '#ecf0f1'
                        }
                    }
                },
                elements: {
                    point: {
                        radius: 1
                    }
                },
                animation: {
                    duration: 200
                }
            }
        });
    }

    function updateChart(data) {
        if (!metricsChart || !data.metrics) return;

        const step = data.step || stepCount;
        
        // Limit chart data to last 50 points for performance
        if (chartData.labels.length > 50) {
            chartData.labels.shift();
            chartData.datasets.forEach(dataset => dataset.data.shift());
        }

        chartData.labels.push(step);
        chartData.datasets[0].data.push(data.metrics.total_cells || 0);
        chartData.datasets[1].data.push(data.metrics.num_activated || 0);
        chartData.datasets[2].data.push((data.metrics.avg_potency || 0) * 10); // Scale for visibility
        
        metricsChart.update('none'); // No animation for real-time updates
    }

    function clearChart() {
        if (!metricsChart) return;
        
        chartData.labels = [];
        chartData.datasets.forEach(dataset => {
            dataset.data = [];
        });
        metricsChart.update();
    }

    // --- Event Handlers ---
    profileSelect.addEventListener('change', (e) => {
        if (!simulationRunning) {
            socket.emit('load_profile', { profile_name: e.target.value });
        }
    });

    runStandardBtn.addEventListener('click', () => {
        if (!simulationRunning) {
            enableControls(false);
            socket.emit('run_scenario', { scenario_name: 'standard_protocol' });
        }
    });

    runAIBtn.addEventListener('click', () => {
        if (!simulationRunning) {
            enableControls(false);
            socket.emit('run_scenario', { scenario_name: 'ai_strategy' });
        }
    });

    resetBtn.addEventListener('click', () => {
        socket.emit('reset_simulation');
    });

    // --- Initialize Application ---
    function initialize() {
        clearCanvas();
        initializeChart();
        initializeSocket();
        updateStatus('Connecting to server...', 'status-idle');
        addActionLog('Application initialized - Connecting to server...');
    }

    // Start the application
    initialize();
});
