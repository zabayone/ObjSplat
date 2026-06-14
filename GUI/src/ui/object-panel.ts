import { BooleanInput, ColorPicker, Container, Label, SelectInput, SliderInput } from '@playcanvas/pcui';
import { Color } from 'playcanvas';

import { SetSplatColorAdjustmentOp } from '../edit-ops';
import { Events } from '../events';
import { Splat } from '../splat';
import { State } from '../splat-state';
import { Tooltips } from './tooltips';

type MoodPreset = {
    id: string,
    label: string,
    tint: Color,
    temperature: number,
    saturation: number,
    brightness: number,
    blackPoint: number,
    whitePoint: number,
    transparency: number
};

const neutralPreset: MoodPreset = {
    id: 'neutral',
    label: 'Neutral',
    tint: new Color(1, 1, 1),
    temperature: 0,
    saturation: 1,
    brightness: 0,
    blackPoint: 0,
    whitePoint: 1,
    transparency: 1
};

const moodPresets: MoodPreset[] = [
    neutralPreset,
    {
        id: 'warm',
        label: 'Warmth',
        tint: new Color(1, 0.86, 0.64),
        temperature: 0.22,
        saturation: 1.18,
        brightness: 0.06,
        blackPoint: 0.02,
        whitePoint: 0.96,
        transparency: 1
    },
    {
        id: 'calm',
        label: 'Calm',
        tint: new Color(0.74, 0.9, 1),
        temperature: -0.14,
        saturation: 0.86,
        brightness: 0.03,
        blackPoint: 0,
        whitePoint: 1,
        transparency: 1
    },
    {
        id: 'tension',
        label: 'Tension',
        tint: new Color(1, 0.74, 0.72),
        temperature: 0.16,
        saturation: 1.35,
        brightness: -0.06,
        blackPoint: 0.12,
        whitePoint: 0.88,
        transparency: 1
    },
    {
        id: 'melancholy',
        label: 'Melancholy',
        tint: new Color(0.68, 0.76, 0.92),
        temperature: -0.24,
        saturation: 0.68,
        brightness: -0.04,
        blackPoint: 0.06,
        whitePoint: 0.94,
        transparency: 1
    },
    {
        id: 'intimate',
        label: 'Intimate',
        tint: new Color(1, 0.78, 0.66),
        temperature: 0.28,
        saturation: 0.92,
        brightness: -0.02,
        blackPoint: 0.09,
        whitePoint: 0.92,
        transparency: 1
    }
];

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

const mixedPreset = (preset: MoodPreset, intensity: number) => {
    const t = Math.max(0, Math.min(1, intensity));
    return {
        tintClr: new Color(
            lerp(neutralPreset.tint.r, preset.tint.r, t),
            lerp(neutralPreset.tint.g, preset.tint.g, t),
            lerp(neutralPreset.tint.b, preset.tint.b, t)
        ),
        temperature: lerp(neutralPreset.temperature, preset.temperature, t),
        saturation: lerp(neutralPreset.saturation, preset.saturation, t),
        brightness: lerp(neutralPreset.brightness, preset.brightness, t),
        blackPoint: lerp(neutralPreset.blackPoint, preset.blackPoint, t),
        whitePoint: lerp(neutralPreset.whitePoint, preset.whitePoint, t),
        transparency: lerp(neutralPreset.transparency, preset.transparency, t)
    };
};

class ObjectPanel extends Container {
    constructor(events: Events, tooltips: Tooltips, args = {}) {
        args = {
            ...args,
            id: 'object-panel',
            class: 'panel',
            hidden: true
        };

        super(args);

        ['pointerdown', 'pointerup', 'pointermove', 'wheel', 'dblclick'].forEach((eventName) => {
            this.dom.addEventListener(eventName, (event: Event) => event.stopPropagation());
        });

        const header = new Container({
            class: 'panel-header'
        });

        const icon = new Label({
            text: '\uE8A1',
            class: 'panel-header-icon'
        });

        const label = new Label({
            text: 'Object Mode',
            class: 'panel-header-label'
        });

        header.append(icon);
        header.append(label);

        const row = new Container({
            class: 'view-panel-row'
        });

        const rowLabel = new Label({
            text: 'Enable object isolation',
            class: 'view-panel-row-label'
        });

        const toggle = new BooleanInput({
            type: 'toggle',
            class: 'view-panel-row-toggle',
            value: false
        });

        row.append(rowLabel);
        row.append(toggle);

        const sectionSelection = new Label({
            text: 'Selection',
            class: 'object-panel-section'
        });

        const selectedStats = new Label({
            text: 'No object selected',
            class: 'object-panel-stats'
        });

        const glowRow = new Container({
            class: 'view-panel-row'
        });

        const glowLabel = new Label({
            text: 'Selection glow',
            class: 'view-panel-row-label'
        });

        const glowSlider = new SliderInput({
            class: 'view-panel-row-slider',
            min: 0,
            max: 1,
            precision: 2,
            value: 1
        });

        glowRow.append(glowLabel);
        glowRow.append(glowSlider);

        const overlayRow = new Container({
            class: 'view-panel-row'
        });

        const overlayLabel = new Label({
            text: 'Overlay colors',
            class: 'view-panel-row-label'
        });

        const overlayPickers = new Container({
            class: ['view-panel-row-pickers', 'object-panel-pickers']
        });

        const selectedPicker = new ColorPicker({
            class: 'view-panel-row-picker',
            channels: 4,
            value: [1, 0.45, 0.08, 0.75]
        });

        const mutedPicker = new ColorPicker({
            class: 'view-panel-row-picker',
            channels: 4,
            value: [0.08, 0.08, 0.08, 0.65]
        });

        overlayPickers.append(selectedPicker);
        overlayPickers.append(mutedPicker);
        overlayRow.append(overlayLabel);
        overlayRow.append(overlayPickers);

        const actionRow = new Container({
            class: 'object-panel-action-row'
        });

        const clearSelection = new Label({
            text: 'Clear',
            class: 'object-panel-action'
        });

        const hideSelection = new Label({
            text: 'Hide',
            class: 'object-panel-action'
        });

        const restoreHidden = new Label({
            text: 'Restore',
            class: 'object-panel-action'
        });

        const deleteSelection = new Label({
            text: 'Delete',
            class: ['object-panel-action', 'danger']
        });

        actionRow.append(clearSelection);
        actionRow.append(hideSelection);
        actionRow.append(restoreHidden);
        actionRow.append(deleteSelection);

        const sectionMood = new Label({
            text: 'Mood',
            class: 'object-panel-section'
        });

        const moodRow = new Container({
            class: 'view-panel-row'
        });

        const moodLabel = new Label({
            text: 'Context preset',
            class: 'view-panel-row-label'
        });

        const moodSelect = new SelectInput({
            class: 'view-panel-row-select',
            defaultValue: neutralPreset.id,
            options: moodPresets.map((preset) => ({ v: preset.id, t: preset.label }))
        });

        moodRow.append(moodLabel);
        moodRow.append(moodSelect);

        const intensityRow = new Container({
            class: 'view-panel-row'
        });

        const intensityLabel = new Label({
            text: 'Intensity',
            class: 'view-panel-row-label'
        });

        const intensitySlider = new SliderInput({
            class: 'view-panel-row-slider',
            min: 0,
            max: 1,
            precision: 2,
            value: 0.65
        });

        intensityRow.append(intensityLabel);
        intensityRow.append(intensitySlider);

        const tintRow = new Container({
            class: 'view-panel-row'
        });

        const tintLabel = new Label({
            text: 'Manual tint',
            class: 'view-panel-row-label'
        });

        const tintPicker = new ColorPicker({
            class: 'view-panel-row-picker',
            value: [1, 1, 1]
        });

        tintRow.append(tintLabel);
        tintRow.append(tintPicker);

        const fineRow = new Container({
            class: 'object-panel-mini-grid'
        });

        const createMiniSlider = (labelText: string, min: number, max: number, value: number) => {
            const mini = new Container({
                class: 'object-panel-mini-row'
            });
            const miniLabel = new Label({
                text: labelText,
                class: 'object-panel-mini-label'
            });
            const slider = new SliderInput({
                class: 'object-panel-mini-slider',
                min,
                max,
                precision: 2,
                value
            });
            mini.append(miniLabel);
            mini.append(slider);
            fineRow.append(mini);
            return slider;
        };

        const temperatureSlider = createMiniSlider('Temp', -0.5, 0.5, 0);
        const saturationSlider = createMiniSlider('Sat', 0, 2, 1);
        const brightnessSlider = createMiniSlider('Light', -1, 1, 0);
        const contrastSlider = createMiniSlider('Contrast', 0, 1, 0);

        const applyMood = new Label({
            text: 'Apply mood to selected splat',
            class: 'object-panel-apply'
        });

        const resetMood = new Label({
            text: 'Reset mood',
            class: ['object-panel-apply', 'secondary']
        });

        const hint = new Label({
            text: 'Click an object to select same-label gaussians. Mood controls grade the selected splat in realtime, ready for emotional-context automation.',
            class: 'view-panel-hint'
        });

        this.append(header);
        this.append(row);
        this.append(sectionSelection);
        this.append(selectedStats);
        this.append(glowRow);
        this.append(overlayRow);
        this.append(actionRow);
        this.append(sectionMood);
        this.append(moodRow);
        this.append(intensityRow);
        this.append(tintRow);
        this.append(fineRow);
        this.append(applyMood);
        this.append(resetMood);
        this.append(hint);

        let selected: Splat = null;
        let suppress = false;

        const activePreset = () => {
            return moodPresets.find((preset) => preset.id === moodSelect.value) ?? neutralPreset;
        };

        const selectedObjectInfo = (splat: Splat) => {
            if (!splat) {
                return null;
            }

            const state = splat.splatData.getProp('state') as Uint8Array;
            const labels = splat.labels as ArrayLike<number> | null;
            let count = 0;
            let labelValue: number = null;

            for (let i = 0; i < state.length; ++i) {
                if (state[i] === State.selected) {
                    count++;
                    if (labelValue === null && labels && i < labels.length) {
                        labelValue = Number(labels[i]);
                    }
                }
            }

            return { count, labelValue };
        };

        const updateStats = () => {
            const info = selectedObjectInfo(selected);
            if (!selected || !info?.count) {
                selectedStats.text = 'No object selected';
                return;
            }
            const labelText = Number.isFinite(info.labelValue) ? `label ${info.labelValue}` : 'unlabelled';
            selectedStats.text = `${selected.name}: ${info.count.toLocaleString()} gaussians, ${labelText}`;
        };

        const updateMoodControls = (splat: Splat) => {
            if (suppress) {
                return;
            }

            suppress = true;
            tintPicker.value = splat ? [splat.tintClr.r, splat.tintClr.g, splat.tintClr.b] : [1, 1, 1];
            temperatureSlider.value = splat ? splat.temperature : 0;
            saturationSlider.value = splat ? splat.saturation : 1;
            brightnessSlider.value = splat ? splat.brightness : 0;
            contrastSlider.value = splat ? Math.max(0, Math.min(1, splat.blackPoint + (1 - splat.whitePoint))) : 0;
            suppress = false;
        };

        const makeMoodOp = (splat: Splat, nextState: ReturnType<typeof mixedPreset>) => {
            return new SetSplatColorAdjustmentOp({
                splat,
                newState: nextState,
                oldState: {
                    tintClr: splat.tintClr.clone(),
                    temperature: splat.temperature,
                    saturation: splat.saturation,
                    brightness: splat.brightness,
                    blackPoint: splat.blackPoint,
                    whitePoint: splat.whitePoint,
                    transparency: splat.transparency
                }
            });
        };

        const previewMood = () => {
            if (!selected || suppress) {
                return;
            }

            const contrast = contrastSlider.value ?? 0;
            selected.tintClr = new Color(tintPicker.value[0], tintPicker.value[1], tintPicker.value[2]);
            selected.temperature = temperatureSlider.value ?? 0;
            selected.saturation = saturationSlider.value ?? 1;
            selected.brightness = brightnessSlider.value ?? 0;
            selected.blackPoint = contrast * 0.5;
            selected.whitePoint = 1 - contrast * 0.5;
        };

        const applyPresetToControls = () => {
            const state = mixedPreset(activePreset(), intensitySlider.value ?? 0);
            suppress = true;
            tintPicker.value = [state.tintClr.r, state.tintClr.g, state.tintClr.b];
            temperatureSlider.value = state.temperature;
            saturationSlider.value = state.saturation;
            brightnessSlider.value = state.brightness;
            contrastSlider.value = state.blackPoint + (1 - state.whitePoint);
            suppress = false;
            previewMood();
        };

        const commitCurrentMood = () => {
            if (!selected) {
                return;
            }

            const contrast = contrastSlider.value ?? 0;
            events.fire('edit.add', makeMoodOp(selected, {
                tintClr: new Color(tintPicker.value[0], tintPicker.value[1], tintPicker.value[2]),
                temperature: temperatureSlider.value ?? 0,
                saturation: saturationSlider.value ?? 1,
                brightness: brightnessSlider.value ?? 0,
                blackPoint: contrast * 0.5,
                whitePoint: 1 - contrast * 0.5,
                transparency: selected.transparency
            }));
        };

        const setVisible = (visible: boolean) => {
            if (visible === this.hidden) {
                this.hidden = !visible;
                events.fire('objectPanel.visible', visible);
            }
        };

        events.function('objectPanel.visible', () => {
            return !this.hidden;
        });

        events.on('objectPanel.setVisible', (visible: boolean) => {
            setVisible(visible);
        });

        events.on('objectPanel.toggleVisible', () => {
            setVisible(this.hidden);
        });

        events.on('objectPanel.visible', (visible: boolean) => {
            toggle.value = visible;
        });

        toggle.on('change', (value: boolean) => {
            events.fire('view.setObjectMode', value);
        });

        glowSlider.on('change', (value: number) => {
            if (selected) {
                selected.selectionAlpha = value;
                events.fire('objectPanel.glow', value);
            }
        });

        selectedPicker.on('change', (value: number[]) => {
            events.fire('setSelectedClr', new Color(value[0], value[1], value[2], value[3]));
        });

        mutedPicker.on('change', (value: number[]) => {
            events.fire('setUnselectedClr', new Color(value[0], value[1], value[2], value[3]));
        });

        clearSelection.on('click', () => events.fire('select.none'));
        hideSelection.on('click', () => events.fire('select.hide'));
        restoreHidden.on('click', () => events.fire('select.unhide'));
        deleteSelection.on('click', () => events.fire('select.delete'));

        moodSelect.on('change', applyPresetToControls);
        intensitySlider.on('change', applyPresetToControls);

        [temperatureSlider, saturationSlider, brightnessSlider, contrastSlider].forEach((slider) => {
            slider.on('change', previewMood);
        });

        tintPicker.on('change', previewMood);
        applyMood.on('click', commitCurrentMood);

        resetMood.on('click', () => {
            if (selected) {
                events.fire('edit.add', makeMoodOp(selected, mixedPreset(neutralPreset, 1)));
            }
        });

        events.on('selection.changed', (splat: Splat) => {
            selected = splat;
            updateStats();
            updateMoodControls(splat);
        });

        events.on('splat.stateChanged', (splat: Splat) => {
            if (splat === selected) {
                updateStats();
            }
        });

        events.on('selectedClr', (clr: Color) => {
            selectedPicker.value = [clr.r, clr.g, clr.b, clr.a];
        });

        events.on('unselectedClr', (clr: Color) => {
            mutedPicker.value = [clr.r, clr.g, clr.b, clr.a];
        });

        events.on('view.objectMode', (value: boolean) => {
            toggle.value = value;
            if (value) {
                events.fire('tool.eyedropperSelection');
            } else if (events.invoke('tool.active') === 'eyedropperSelection') {
                events.fire('tool.deactivate');
            }
        });

        events.on('colorPanel.visible', (visible: boolean) => {
            if (visible) {
                setVisible(false);
            }
        });

        events.on('viewPanel.visible', (visible: boolean) => {
            if (visible) {
                setVisible(false);
            }
        });

        tooltips.register(label, 'Object-level selection mode', 'left');
        tooltips.register(rowLabel, 'Click once on an object to isolate it and edit all gaussians with its label.', 'left');
        tooltips.register(glowLabel, 'Adjust the strength of selected gaussian highlight.', 'left');
        tooltips.register(overlayLabel, 'Selected and muted overlay colors for object isolation.', 'left');
        tooltips.register(applyMood, 'Commit the current mood grade to undo history.', 'bottom');
        tooltips.register(resetMood, 'Restore neutral color grading on the selected splat.', 'bottom');
    }
}

export { ObjectPanel };
