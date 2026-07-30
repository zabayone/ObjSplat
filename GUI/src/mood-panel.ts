import { Color } from 'playcanvas';

import { Events } from './events';
import { Splat } from './splat';

type CircumplexPoint = {
    valence: number,
    arousal: number
};

type MoodEntry = {
    ply_path: string,
    circumplex?: CircumplexPoint,
    time_of_day?: string
};

type MoodManifest = {
    active_mood?: string,
    moods?: Record<string, MoodEntry>
};

const fallbackPoints: Record<string, CircumplexPoint> = {
    day: { valence: 0, arousal: 0 },
    neutral: { valence: 0, arousal: 0 },
    serene: { valence: 0.72, arousal: -0.68 },
    joyful: { valence: 0.82, arousal: 0.74 },
    tense: { valence: -0.76, arousal: 0.82 },
    melancholic: { valence: -0.72, arousal: -0.62 },
    night: { valence: -0.15, arousal: -0.35 }
};

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

const pointFor = (name: string, entry: MoodEntry): CircumplexPoint => {
    const point = entry.circumplex ?? fallbackPoints[name] ?? fallbackPoints.neutral;
    return {
        valence: clamp(Number(point.valence) || 0, -1, 1),
        arousal: clamp(Number(point.arousal) || 0, -1, 1)
    };
};

const timeFor = (name: string, entry: MoodEntry) => {
    return entry.time_of_day ?? (name === 'night' ? 'night' : 'day');
};

const displayName = (name: string) => {
    return name.replaceAll('_', ' ').replace(/\b\w/g, char => char.toUpperCase());
};

const initMoodPanel = async (events: Events, manifestUrl: string, moodRootUrl: string) => {
    const response = await fetch(manifestUrl, { cache: 'no-store' });
    if (!response.ok) {
        throw new Error(`Unable to load mood manifest (${response.status})`);
    }
    const manifest = await response.json() as MoodManifest;
    const moods = manifest.moods ?? {};
    const moodNames = Object.keys(moods).filter(name => Boolean(moods[name]?.ply_path));
    if (moodNames.length === 0) {
        return;
    }

    const root = document.createElement('section');
    root.id = 'objsplat-mood-panel';
    root.innerHTML = `
        <div class="mood-panel-header">
            <span>SCENE MOOD</span>
            <button class="mood-panel-collapse" type="button" aria-label="Collapse mood panel">−</button>
        </div>
        <div class="mood-panel-content">
            <div class="mood-panel-coordinates">
                <span>Valence <strong data-value="valence">0.00</strong></span>
                <span>Arousal <strong data-value="arousal">0.00</strong></span>
            </div>
            <canvas class="mood-pad" width="216" height="216" aria-label="Valence and arousal pad"></canvas>
            <label class="mood-slider-row">
                <span>Valence</span>
                <input data-slider="valence" type="range" min="-1" max="1" step="0.01" value="0">
            </label>
            <label class="mood-slider-row">
                <span>Arousal</span>
                <input data-slider="arousal" type="range" min="-1" max="1" step="0.01" value="0">
            </label>
            <label class="mood-time-row">
                <span>Time</span>
                <select data-time></select>
            </label>
            <div class="mood-preset-list"></div>
            <div class="mood-panel-actions">
                <button data-action="nearest" type="button">Load nearest exact mood</button>
                <button data-action="reset" type="button">Reset live grade</button>
            </div>
            <div class="mood-panel-status" role="status"></div>
        </div>
    `;

    const canvasContainer = document.getElementById('canvas-container');
    if (!canvasContainer) {
        throw new Error('SuperSplat canvas container is unavailable');
    }
    canvasContainer.appendChild(root);
    ['pointerdown', 'pointerup', 'pointermove', 'wheel', 'dblclick'].forEach((eventName) => {
        root.addEventListener(eventName, event => event.stopPropagation());
    });

    const canvas = root.querySelector('.mood-pad') as HTMLCanvasElement;
    const context = canvas.getContext('2d');
    const valenceSlider = root.querySelector('[data-slider="valence"]') as HTMLInputElement;
    const arousalSlider = root.querySelector('[data-slider="arousal"]') as HTMLInputElement;
    const valenceValue = root.querySelector('[data-value="valence"]') as HTMLElement;
    const arousalValue = root.querySelector('[data-value="arousal"]') as HTMLElement;
    const timeSelect = root.querySelector('[data-time]') as HTMLSelectElement;
    const presetList = root.querySelector('.mood-preset-list') as HTMLElement;
    const status = root.querySelector('.mood-panel-status') as HTMLElement;
    const content = root.querySelector('.mood-panel-content') as HTMLElement;
    const collapse = root.querySelector('.mood-panel-collapse') as HTMLButtonElement;

    const availableTimes = Array.from(new Set(
        moodNames.map(name => timeFor(name, moods[name]))
    ));
    for (const time of availableTimes) {
        const option = document.createElement('option');
        option.value = time;
        option.textContent = displayName(time);
        timeSelect.appendChild(option);
    }

    const initialLoad = new URL(window.location.href).searchParams.get('load');
    const initialPath = initialLoad ? new URL(initialLoad).pathname : '';
    const matchingInitial = moodNames.find((name) => {
        return new URL(moods[name].ply_path, moodRootUrl).pathname === initialPath;
    });
    let activeMood = matchingInitial ?? manifest.active_mood ?? moodNames[0];
    if (!moods[activeMood]) {
        activeMood = moodNames[0];
    }
    let basePoint = pointFor(activeMood, moods[activeMood]);
    let currentPoint = { ...basePoint };
    let loading = false;

    const drawPad = () => {
        const width = canvas.width;
        const height = canvas.height;
        const gradient = context.createLinearGradient(0, 0, width, height);
        gradient.addColorStop(0, '#314d68');
        gradient.addColorStop(0.48, '#252b30');
        gradient.addColorStop(1, '#d3813b');
        context.fillStyle = gradient;
        context.fillRect(0, 0, width, height);

        const arousalGradient = context.createLinearGradient(0, height, 0, 0);
        arousalGradient.addColorStop(0, 'rgba(130, 160, 180, 0.30)');
        arousalGradient.addColorStop(0.5, 'rgba(0, 0, 0, 0)');
        arousalGradient.addColorStop(1, 'rgba(220, 55, 35, 0.34)');
        context.fillStyle = arousalGradient;
        context.fillRect(0, 0, width, height);

        context.strokeStyle = 'rgba(255, 255, 255, 0.28)';
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(width / 2, 0);
        context.lineTo(width / 2, height);
        context.moveTo(0, height / 2);
        context.lineTo(width, height / 2);
        context.stroke();

        context.font = '11px sans-serif';
        context.fillStyle = 'rgba(255, 255, 255, 0.78)';
        context.fillText('TENSE', 8, 16);
        context.fillText('JOYFUL', width - 48, 16);
        context.fillText('MELANCHOLIC', 8, height - 9);
        context.fillText('SERENE', width - 45, height - 9);

        for (const name of moodNames) {
            const entry = moods[name];
            if (timeFor(name, entry) !== timeSelect.value) continue;
            const point = pointFor(name, entry);
            const x = (point.valence + 1) * 0.5 * width;
            const y = (1 - (point.arousal + 1) * 0.5) * height;
            context.beginPath();
            context.fillStyle = name === activeMood ? '#ffb15b' : 'rgba(255, 255, 255, 0.72)';
            context.arc(x, y, name === activeMood ? 4 : 2.5, 0, Math.PI * 2);
            context.fill();
        }

        const markerX = (currentPoint.valence + 1) * 0.5 * width;
        const markerY = (1 - (currentPoint.arousal + 1) * 0.5) * height;
        context.beginPath();
        context.fillStyle = '#fff';
        context.strokeStyle = '#111';
        context.lineWidth = 2;
        context.arc(markerX, markerY, 7, 0, Math.PI * 2);
        context.fill();
        context.stroke();
    };

    const applyLiveGrade = () => {
        const deltaValence = currentPoint.valence - basePoint.valence;
        const deltaArousal = currentPoint.arousal - basePoint.arousal;
        const contrast = clamp(0.12 * deltaArousal - 0.025 * Math.min(0, deltaValence), -0.16, 0.20);
        const splats = events.invoke('scene.allSplats') as Splat[] ?? [];
        for (const splat of splats) {
            splat.tintClr = Color.WHITE;
            splat.temperature = clamp(0.20 * deltaValence, -0.5, 0.5);
            splat.saturation = clamp(1 + 0.24 * deltaValence + 0.16 * deltaArousal, 0.4, 1.8);
            splat.brightness = clamp(0.11 * deltaValence + 0.06 * deltaArousal, -0.5, 0.5);
            splat.blackPoint = contrast * 0.5;
            splat.whitePoint = 1 - contrast * 0.5;
        }
    };

    const syncControls = (applyGrade = true) => {
        valenceSlider.value = currentPoint.valence.toFixed(2);
        arousalSlider.value = currentPoint.arousal.toFixed(2);
        valenceValue.textContent = currentPoint.valence.toFixed(2);
        arousalValue.textContent = currentPoint.arousal.toFixed(2);
        if (applyGrade) applyLiveGrade();
        drawPad();
    };

    const nearestMood = () => {
        const candidates = moodNames.filter(name => timeFor(name, moods[name]) === timeSelect.value);
        return candidates.reduce((nearest, name) => {
            if (!nearest) return name;
            const point = pointFor(name, moods[name]);
            const nearestPoint = pointFor(nearest, moods[nearest]);
            const distance = (point.valence - currentPoint.valence) ** 2 + (point.arousal - currentPoint.arousal) ** 2;
            const nearestDistance = (nearestPoint.valence - currentPoint.valence) ** 2 + (nearestPoint.arousal - currentPoint.arousal) ** 2;
            return distance < nearestDistance ? name : nearest;
        }, '');
    };

    const loadMood = async (name: string) => {
        if (loading || !moods[name]) return;
        loading = true;
        status.textContent = `Loading ${displayName(name)}…`;
        root.classList.add('loading');
        events.fire('startSpinner');
        try {
            const entry = moods[name];
            const moodUrl = new URL(entry.ply_path, moodRootUrl);
            moodUrl.searchParams.set('moodCache', Date.now().toString());
            events.fire('scene.clear');
            const imported = await events.invoke('import', [{
                filename: entry.ply_path.split('/').pop(),
                url: moodUrl.toString()
            }]) as Splat[];
            if (!imported?.length) {
                throw new Error('SuperSplat did not import the selected mood PLY');
            }
            activeMood = name;
            basePoint = pointFor(name, entry);
            currentPoint = { ...basePoint };
            timeSelect.value = timeFor(name, entry);
            status.textContent = `${displayName(name)} · exact PLY`;
            renderPresetButtons();
            syncControls();
        } catch (error) {
            status.textContent = `Load failed: ${error.message ?? error}`;
        } finally {
            loading = false;
            root.classList.remove('loading');
            events.fire('stopSpinner');
        }
    };

    function renderPresetButtons() {
        presetList.replaceChildren();
        for (const name of moodNames) {
            if (timeFor(name, moods[name]) !== timeSelect.value) continue;
            const button = document.createElement('button');
            button.type = 'button';
            button.textContent = displayName(name);
            button.className = name === activeMood ? 'active' : '';
            button.addEventListener('click', () => loadMood(name));
            presetList.appendChild(button);
        }
    }

    const updateFromInputs = () => {
        currentPoint = {
            valence: Number(valenceSlider.value),
            arousal: Number(arousalSlider.value)
        };
        status.textContent = `${displayName(activeMood)} · live modulation`;
        syncControls();
    };

    let padDragging = false;
    const updateFromPad = (event: PointerEvent) => {
        const bounds = canvas.getBoundingClientRect();
        currentPoint = {
            valence: clamp(((event.clientX - bounds.left) / bounds.width) * 2 - 1, -1, 1),
            arousal: clamp(1 - ((event.clientY - bounds.top) / bounds.height) * 2, -1, 1)
        };
        status.textContent = `${displayName(activeMood)} · live modulation`;
        syncControls();
    };
    canvas.addEventListener('pointerdown', (event) => {
        padDragging = true;
        canvas.setPointerCapture(event.pointerId);
        updateFromPad(event);
    });
    canvas.addEventListener('pointermove', (event) => {
        if (padDragging) updateFromPad(event);
    });
    canvas.addEventListener('pointerup', (event) => {
        padDragging = false;
        canvas.releasePointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointercancel', () => {
        padDragging = false;
    });

    valenceSlider.addEventListener('input', updateFromInputs);
    arousalSlider.addEventListener('input', updateFromInputs);
    timeSelect.addEventListener('change', () => {
        renderPresetButtons();
        drawPad();
        status.textContent = `${displayName(timeSelect.value)} variants`;
    });
    root.querySelector('[data-action="nearest"]').addEventListener('click', () => {
        const nearest = nearestMood();
        if (nearest) loadMood(nearest);
    });
    root.querySelector('[data-action="reset"]').addEventListener('click', () => {
        currentPoint = { ...basePoint };
        status.textContent = `${displayName(activeMood)} · exact PLY`;
        syncControls();
    });
    collapse.addEventListener('click', () => {
        const collapsed = root.classList.toggle('collapsed');
        content.hidden = collapsed;
        collapse.textContent = collapsed ? '+' : '−';
    });

    timeSelect.value = timeFor(activeMood, moods[activeMood]);
    status.textContent = `${displayName(activeMood)} · exact PLY`;
    renderPresetButtons();
    syncControls();
};

export { initMoodPanel };
