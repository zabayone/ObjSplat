import { Color } from 'playcanvas';

import { Events } from './events';
import { Splat } from './splat';

type GradePoint = {
    exposure: number,
    brightness: number,
    contrast: number
};

const EXPOSURE_LIMIT = 2;
const BRIGHTNESS_LIMIT = 0.35;
const CONTRAST_LIMIT = 0.40;

const presets: Record<string, GradePoint> = {
    neutral: { exposure: 0, brightness: 0, contrast: 0 },
    serene: { exposure: 0.25, brightness: 0.03, contrast: -0.12 },
    joyful: { exposure: 0.55, brightness: 0.08, contrast: 0.14 },
    tense: { exposure: -0.35, brightness: -0.04, contrast: 0.20 },
    melancholic: { exposure: -0.50, brightness: -0.06, contrast: -0.10 },
    night: { exposure: -1.25, brightness: -0.10, contrast: 0.06 }
};

const clamp = (value: number, minimum: number, maximum: number) => {
    return Math.max(minimum, Math.min(maximum, value));
};

const displayName = (name: string) => {
    return name.replaceAll('_', ' ').replace(/\b\w/g, char => char.toUpperCase());
};

const initMoodPanel = (events: Events) => {
    if (document.getElementById('objsplat-mood-panel')) return;

    const root = document.createElement('section');
    root.id = 'objsplat-mood-panel';
    root.innerHTML = `
        <div class="mood-panel-header">
            <span>REAL-TIME MOOD MATRIX</span>
            <button class="mood-panel-collapse" type="button" aria-label="Collapse mood matrix">−</button>
        </div>
        <div class="mood-panel-content">
            <div class="mood-panel-coordinates">
                <span>Exposure <strong data-value="exposure">0.00 EV</strong></span>
                <span>Brightness <strong data-value="brightness">0.00</strong></span>
                <span>Contrast <strong data-value="contrast">0.00</strong></span>
            </div>
            <canvas class="mood-pad" width="216" height="216" aria-label="Exposure and contrast mood matrix"></canvas>
            <label class="mood-slider-row">
                <span>Exposure</span>
                <input data-slider="exposure" type="range" min="-${EXPOSURE_LIMIT}" max="${EXPOSURE_LIMIT}" step="0.05" value="0">
            </label>
            <label class="mood-slider-row">
                <span>Brightness</span>
                <input data-slider="brightness" type="range" min="-${BRIGHTNESS_LIMIT}" max="${BRIGHTNESS_LIMIT}" step="0.01" value="0">
            </label>
            <label class="mood-slider-row">
                <span>Contrast</span>
                <input data-slider="contrast" type="range" min="-${CONTRAST_LIMIT}" max="${CONTRAST_LIMIT}" step="0.01" value="0">
            </label>
            <div class="mood-preset-list"></div>
            <div class="mood-panel-actions">
                <button data-action="reset" type="button">Reset live grade</button>
            </div>
            <div class="mood-panel-status" role="status">Neutral · live GPU grade</div>
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
    const exposureSlider = root.querySelector('[data-slider="exposure"]') as HTMLInputElement;
    const brightnessSlider = root.querySelector('[data-slider="brightness"]') as HTMLInputElement;
    const contrastSlider = root.querySelector('[data-slider="contrast"]') as HTMLInputElement;
    const exposureValue = root.querySelector('[data-value="exposure"]') as HTMLElement;
    const brightnessValue = root.querySelector('[data-value="brightness"]') as HTMLElement;
    const contrastValue = root.querySelector('[data-value="contrast"]') as HTMLElement;
    const presetList = root.querySelector('.mood-preset-list') as HTMLElement;
    const status = root.querySelector('.mood-panel-status') as HTMLElement;
    const content = root.querySelector('.mood-panel-content') as HTMLElement;
    const collapse = root.querySelector('.mood-panel-collapse') as HTMLButtonElement;
    const baseTints = new WeakMap<Splat, Color>();
    const gradedTint = new Color();
    let current: GradePoint = { ...presets.neutral };
    let activePreset = 'neutral';

    const drawPad = () => {
        if (!context) return;
        const { width, height } = canvas;
        const exposure = context.createLinearGradient(0, 0, width, 0);
        exposure.addColorStop(0, '#111820');
        exposure.addColorStop(0.5, '#62676a');
        exposure.addColorStop(1, '#f0d8b4');
        context.fillStyle = exposure;
        context.fillRect(0, 0, width, height);

        const contrast = context.createLinearGradient(0, height, 0, 0);
        contrast.addColorStop(0, 'rgba(125, 135, 142, 0.62)');
        contrast.addColorStop(0.5, 'rgba(0, 0, 0, 0)');
        contrast.addColorStop(1, 'rgba(0, 0, 0, 0.48)');
        context.fillStyle = contrast;
        context.fillRect(0, 0, width, height);

        context.strokeStyle = 'rgba(255, 255, 255, 0.34)';
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(width / 2, 0);
        context.lineTo(width / 2, height);
        context.moveTo(0, height / 2);
        context.lineTo(width, height / 2);
        context.stroke();

        context.font = '10px sans-serif';
        context.fillStyle = 'rgba(255, 255, 255, 0.88)';
        context.fillText('TENSE', 7, 15);
        context.fillText('JOYFUL', width - 45, 15);
        context.fillText('MELANCHOLIC', 7, height - 8);
        context.fillText('SERENE', width - 43, height - 8);

        const markerX = (current.exposure / EXPOSURE_LIMIT + 1) * 0.5 * width;
        const markerY = (1 - (current.contrast / CONTRAST_LIMIT + 1) * 0.5) * height;
        context.beginPath();
        context.fillStyle = '#fff';
        context.strokeStyle = '#111';
        context.lineWidth = 2;
        context.arc(markerX, markerY, 7, 0, Math.PI * 2);
        context.fill();
        context.stroke();
    };

    const applyLiveGrade = () => {
        const splats = events.invoke('scene.allSplats') as Splat[] ?? [];
        const exposureGain = 2 ** current.exposure;
        for (const splat of splats) {
            let baseTint = baseTints.get(splat);
            if (!baseTint) {
                const tint = splat.tintClr;
                baseTint = new Color(tint.r, tint.g, tint.b, tint.a);
                baseTints.set(splat, baseTint);
            }
            gradedTint.set(
                baseTint.r * exposureGain,
                baseTint.g * exposureGain,
                baseTint.b * exposureGain,
                baseTint.a
            );
            splat.tintClr = gradedTint;
            splat.brightness = current.brightness;
            splat.blackPoint = current.contrast * 0.5;
            splat.whitePoint = 1 - current.contrast * 0.5;
        }
    };

    function renderPresetButtons() {
        presetList.replaceChildren();
        Object.entries(presets).forEach(([name, point]) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.textContent = displayName(name);
            button.className = name === activePreset ? 'active' : '';
            button.addEventListener('click', () => {
                activePreset = name;
                current = { ...point };
                status.textContent = `${displayName(name)} · live GPU grade`;
                syncControls();
                renderPresetButtons();
            });
            presetList.appendChild(button);
        });
    }

    function syncControls() {
        exposureSlider.value = current.exposure.toFixed(2);
        brightnessSlider.value = current.brightness.toFixed(2);
        contrastSlider.value = current.contrast.toFixed(2);
        exposureValue.textContent = `${current.exposure.toFixed(2)} EV`;
        brightnessValue.textContent = current.brightness.toFixed(2);
        contrastValue.textContent = current.contrast.toFixed(2);
        applyLiveGrade();
        drawPad();
    }

    const updateFromInputs = () => {
        activePreset = '';
        current = {
            exposure: Number(exposureSlider.value),
            brightness: Number(brightnessSlider.value),
            contrast: Number(contrastSlider.value)
        };
        status.textContent = 'Custom · live GPU grade';
        syncControls();
        renderPresetButtons();
    };

    let padDragging = false;
    const updateFromPad = (event: PointerEvent) => {
        const bounds = canvas.getBoundingClientRect();
        activePreset = '';
        current = {
            exposure: clamp(
                (((event.clientX - bounds.left) / bounds.width) * 2 - 1) * EXPOSURE_LIMIT,
                -EXPOSURE_LIMIT,
                EXPOSURE_LIMIT
            ),
            brightness: current.brightness,
            contrast: clamp(
                (1 - ((event.clientY - bounds.top) / bounds.height) * 2) * CONTRAST_LIMIT,
                -CONTRAST_LIMIT,
                CONTRAST_LIMIT
            )
        };
        status.textContent = 'Custom · live GPU grade';
        syncControls();
        renderPresetButtons();
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
    exposureSlider.addEventListener('input', updateFromInputs);
    brightnessSlider.addEventListener('input', updateFromInputs);
    contrastSlider.addEventListener('input', updateFromInputs);
    root.querySelector('[data-action="reset"]').addEventListener('click', () => {
        activePreset = 'neutral';
        current = { ...presets.neutral };
        status.textContent = 'Neutral · live GPU grade';
        syncControls();
        renderPresetButtons();
    });
    collapse.addEventListener('click', () => {
        const collapsed = root.classList.toggle('collapsed');
        content.hidden = collapsed;
        collapse.textContent = collapsed ? '+' : '−';
    });

    renderPresetButtons();
    syncControls();
};

export { initMoodPanel };
